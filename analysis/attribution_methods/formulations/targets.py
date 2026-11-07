"""Attribution targets: the scalar Captum differentiates. Each returns a detached per-cell
reference and a scalar g(e(x), ref). TargetModule wraps (encoder, target) for Captum."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _proto_tensor(ctx, labels_batch, key):
    refs = ctx.references
    try:
        arr = np.stack([refs.prototypes[key(c)] for c in labels_batch]).astype("float32")
    except KeyError as e:
        raise KeyError(f"no prototype for cluster {e}; references built for "
                       f"{sorted(refs.prototypes)}") from e
    return torch.as_tensor(arr, device=ctx.device)


class SelfRepr:
    name = "self"
    needs_references = False

    def reference(self, ctx, inputs, labels_batch):
        with torch.no_grad():
            return ctx.encoder(inputs).detach()               # e(x0), fixed along IG path

    def scalar(self, e_x, ref):
        return (ref * e_x).abs().sum(dim=1)                   # sum_d |w_d e_d(x)|


class PrototypeSim:
    name = "proto"
    needs_references = True

    def reference(self, ctx, inputs, labels_batch):
        return _proto_tensor(ctx, labels_batch, key=lambda c: str(c))   # home prototype

    def scalar(self, e_x, ref):
        return (e_x * ref).sum(dim=1)                         # cosine (unit-norm e, p)


class Contrastive:
    name = "contrast"
    needs_references = True

    def reference(self, ctx, inputs, labels_batch):
        home = _proto_tensor(ctx, labels_batch, key=lambda c: str(c))
        comp = _proto_tensor(ctx, labels_batch, key=lambda c: ctx.references.competitor[str(c)])
        return torch.stack([home, comp], dim=1)               # [B, 2, latent]

    def scalar(self, e_x, ref):
        return (e_x * ref[:, 0]).sum(1) - (e_x * ref[:, 1]).sum(1)


class DecisionFn:
    """Supervised discriminative target: train a linear probe (logistic regression) on the cell
    embeddings to predict cluster label, and attribute the home cluster's logit. The reference is the
    probe's class weight vector — the embedding-space direction that best SEPARATES that class
    (discriminative), unlike prototype_sim's class-MEAN direction. Its input-gradient should point at
    the genes that distinguish the cell type, if the model's gradient aligns with markers at all.
    Uses a `prepare` hook the engine calls once before the batch loop (no shared ClusterReferences)."""
    name = "decision"
    needs_references = False

    def prepare(self, encoder, X_lognorm, labels, device):
        from sklearn.linear_model import LogisticRegression
        from .prototypes import _embed
        E = _embed(encoder, X_lognorm, device, 512)
        labels = np.asarray(labels).astype(str)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(E, labels)
        W = np.atleast_2d(clf.coef_)
        classes = [str(c) for c in clf.classes_]
        if W.shape[0] == 1 and len(classes) == 2:   # binary: one weight vector for classes_[1]
            weights = {classes[1]: W[0], classes[0]: -W[0]}
        else:
            weights = {c: W[i] for i, c in enumerate(classes)}
        self._weights = {c: w.astype("float32") for c, w in weights.items()}

    def reference(self, ctx, inputs, labels_batch):
        arr = np.stack([self._weights[str(c)] for c in labels_batch]).astype("float32")
        return torch.as_tensor(arr, device=ctx.device)

    def scalar(self, e_x, ref):
        return (e_x * ref).sum(dim=1)          # home-class logit direction (intercept drops out of grad)


class TargetModule(nn.Module):
    def __init__(self, encoder, target):
        super().__init__()
        self.encoder = encoder
        self.target = target

    def forward(self, inputs, ref):
        s = self.target.scalar(self.encoder(inputs), ref)
        return torch.stack([s, s], dim=1)                     # [B, 2] for Captum target=0
