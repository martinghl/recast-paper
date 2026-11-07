"""IG baselines (integration start): zero (empty cell) or counterfactual (the nearest
competing cell type's mean-lognorm centroid, per cell)."""
from __future__ import annotations
import numpy as np
import torch


class Zero:
    name = "zero"
    needs_references = False
    def vector(self, ctx, inputs, labels_batch):
        return torch.zeros_like(inputs)


class Counterfactual:
    name = "cf"
    needs_references = True
    def vector(self, ctx, inputs, labels_batch):
        refs = ctx.references
        comp = [refs.competitor[str(c)] for c in labels_batch]
        arr = np.stack([refs.centroid_lognorm[k] for k in comp]).astype("float32")
        return torch.as_tensor(arr, device=ctx.device)
