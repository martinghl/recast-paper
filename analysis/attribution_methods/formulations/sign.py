"""Sign policies: |A| (one channel) or signed (pos/neg channels), with a shared-budget
row-normalization so pos+neg reconstructs the normalized |A| and each channel is non-negative."""
from __future__ import annotations
import numpy as np


class AbsSign:
    name = "abs"
    def apply(self, signed_attr: np.ndarray) -> dict:
        return {"abs": np.abs(signed_attr)}


class SignedSign:
    name = "signed"
    def apply(self, signed_attr: np.ndarray) -> dict:
        return {"pos": np.maximum(signed_attr, 0.0),
                "neg": np.maximum(-signed_attr, 0.0)}


def normalize_channels(channels: dict, target_sum: float = 1000.0) -> dict:
    """Divide every channel by the shared per-row sum of the elementwise total, scaled to
    target_sum. The denominator is the |A| row-sum for both abs (total=|A|) and signed
    (total=pos+neg=|A|). All-zero rows are guarded (left at 0)."""
    total = sum(channels.values())
    row = np.asarray(total.sum(axis=1)).ravel()
    scale = np.where(row > 0, target_sum / np.where(row > 0, row, 1.0), 0.0)
    return {k: v * scale[:, None] for k, v in channels.items()}
