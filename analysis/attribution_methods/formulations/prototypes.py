"""Build per-cluster references once per (dataset, labels): embedding prototypes (for
prototype_sim / contrastive targets), nearest-competitor map, and mean-lognorm centroids
(for the counterfactual baseline)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from .base import ClusterReferences


def _embed(encoder, X, device, batch_size):
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            xb = xb.toarray() if sp.issparse(xb) else np.asarray(xb)
            t = torch.as_tensor(xb, dtype=torch.float32, device=device)
            out.append(encoder(t).cpu().numpy())
    return np.vstack(out)


def _clusters_sorted(labels):
    return sorted(set(map(str, labels)), key=lambda x: (len(x), x))


def build_cluster_references(encoder, X_lognorm, labels, device="cpu",
                             batch_size=512) -> ClusterReferences:
    labels = np.asarray(labels).astype(str)
    clusters = _clusters_sorted(labels)
    E = _embed(encoder, X_lognorm, device, batch_size)     # [N, latent]
    prototypes, centroids = {}, {}
    for c in clusters:
        m = labels == c
        p = E[m].mean(0)
        prototypes[c] = p / (np.linalg.norm(p) + 1e-12)
        sub = X_lognorm[m]
        centroids[c] = (np.asarray(sub.mean(0)).ravel() if sp.issparse(sub)
                        else np.asarray(sub).mean(0)).astype("float32")
    competitor = {}
    for c in clusters:
        sims = {c2: float(prototypes[c] @ prototypes[c2]) for c2 in clusters if c2 != c}
        competitor[c] = max(sims, key=sims.get)
    return ClusterReferences(clusters=clusters, prototypes=prototypes,
                             competitor=competitor, centroid_lognorm=centroids)
