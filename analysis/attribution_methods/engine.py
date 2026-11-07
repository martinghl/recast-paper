"""AttributionEngine: run a Captum attribution method (integrated gradients, saliency or input x gradient)
for a formulation (target x baseline x sign) and return non-negative, row-normalized channels."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from captum.attr import IntegratedGradients, Saliency, InputXGradient
from formulations.base import EngineContext
from formulations.prototypes import build_cluster_references
from formulations.targets import TargetModule
from formulations.sign import normalize_channels


class AttributionEngine:
    def __init__(self, encoder, gene_order, device="cpu"):
        self.encoder = encoder.to(device).eval()
        self.gene_order = list(gene_order)
        self.device = device

    @classmethod
    def from_scimilarity(cls, model_path, use_gpu=True):
        import sys
        _sig_src = "/path/to/SIGnature/src"
        if _sig_src not in sys.path:
            sys.path.insert(0, _sig_src)
        from SIGnature.models.scimilarity import SCimilarityWrapper
        w = SCimilarityWrapper(model_path=model_path, use_gpu=use_gpu)
        device = "cuda" if use_gpu else "cpu"
        return cls(w.model, gene_order=w.gene_order, device=device)

    def attribute(self, X_lognorm, labels, formulation, batch_size=500, n_steps=50,
                  internal_batch_size=None, multiply_by_inputs=True, method="ig"):
        """method: the attribution algorithm — 'ig' (Integrated Gradients, path-integrated),
        'saliency' (|grad| at the input point, pure gradient), 'ixg' (input x grad at the input).
        For 'saliency'/'ixg' the baseline axis has no effect (they are point-gradient, baseline-free).

        multiply_by_inputs (method='ig' only): True (default, SIGnature's recipe) gives
        A = (x - baseline) * IG, so expression is folded in; False gives the pure integrated gradient
        (expression factor removed) — a ratio-based specificity ranker largely cancels the per-gene
        gradient out of the x*g product, so pure gradient tests whether the model adds signal beyond
        expression."""
        labels = np.asarray(labels).astype(str)
        if X_lognorm.shape[1] != len(self.gene_order):
            raise ValueError(f"X has {X_lognorm.shape[1]} genes but engine gene_order has "
                             f"{len(self.gene_order)}; align X to the model gene order first")
        ctx = EngineContext(encoder=self.encoder, device=self.device, references=None)
        if formulation.needs_references:
            ctx.references = build_cluster_references(self.encoder, X_lognorm, labels, self.device)
        if hasattr(formulation.target, "prepare"):
            formulation.target.prepare(self.encoder, X_lognorm, labels, self.device)
        module = TargetModule(self.encoder, formulation.target)
        attributor = _make_attributor(module, method, multiply_by_inputs)

        signed_rows = []
        N = X_lognorm.shape[0]
        for i in range(0, N, batch_size):
            xb = X_lognorm[i:i + batch_size]
            xb = xb.toarray() if sp.issparse(xb) else np.asarray(xb)
            lb = labels[i:i + batch_size]
            inp = torch.as_tensor(xb, dtype=torch.float32, device=self.device).requires_grad_(True)
            ref = formulation.target.reference(ctx, inp, lb)
            base = formulation.baseline.vector(ctx, inp, lb)
            attrs = _attribute_batch(attributor, method, inp, base, ref, n_steps,
                                     internal_batch_size)
            signed_rows.append(attrs.detach().cpu().numpy())
        signed = np.vstack(signed_rows)

        channels = normalize_channels(formulation.sign.apply(signed), target_sum=1000.0)
        return {f"attr__{formulation.name}__{suf}": sp.csr_matrix(v.astype("float32"))
                for suf, v in channels.items()}

    def attach_channels(self, adata, channels: dict) -> None:
        for name, mat in channels.items():
            adata.layers[name] = mat


def _make_attributor(module, method, multiply_by_inputs):
    if method == "ig":
        return IntegratedGradients(module, multiply_by_inputs=multiply_by_inputs)
    if method == "saliency":
        return Saliency(module)                 # |grad| at the input point (pure gradient, no path)
    if method == "ixg":
        return InputXGradient(module)           # input x grad at the input point
    raise ValueError(f"unknown attribution method {method!r}; expected 'ig'|'saliency'|'ixg'")


def _attribute_batch(attributor, method, inp, base, ref, n_steps, internal_batch_size):
    if method == "ig":
        return attributor.attribute(inputs=inp, baselines=base, target=0,
                                    additional_forward_args=(ref,), n_steps=n_steps,
                                    internal_batch_size=internal_batch_size)
    if method == "saliency":
        # abs=False so the downstream SignPolicy owns the sign (|.| for abs, +/- split for signed)
        return attributor.attribute(inputs=inp, target=0, additional_forward_args=(ref,), abs=False)
    if method == "ixg":
        return attributor.attribute(inputs=inp, target=0, additional_forward_args=(ref,))
    raise ValueError(f"unknown attribution method {method!r}")
