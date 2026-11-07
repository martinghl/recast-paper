"""Reference-free attribution of the four CD8 T subtypes of the Zhao data, every gene (Fig. 2b).

Per gene and subtype s (4 x 28,231 rows):

    reference_free_mean_abs_attr[s, g] = mean over the cells of s of |attribution_ig[cell, g]|
    reference_free_rank[s, g]          = rank of that value within s (1 = largest; ties broken by gene symbol)

`attribution_ig` is the per-cell integrated-gradients layer of atlases/zhao_attribution_scimilarity.h5ad
(atlases/attribution.py: SIGnature's integrated gradients of each cell's own embedding, no reference contrast), which
is what Fig. 2b sets against RECAST.  About half of the genes have exactly zero attribution in a subtype, so the lower half
of each ranking is a tie block ordered by gene symbol; Fig. 2b caps ranks at 100, so the tie block is not visible.

The obs label `CRCR3+ CD8` is a typo in the source data for CXCR3; the table uses CXCR3+ CD8, as the RECAST side does.

    /path/to/envs/SC/bin/python reference_free_attribution.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = "/path/to/research/analysis"
ATLASES = f"{ROOT}/atlases"
OUT = f"{ROOT}/paper/cd8_example/data"
STEM = "zhao"
STATES = ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"]   # raw obs labels
RENAME = {"CRCR3+ CD8": "CXCR3+ CD8"}                              # upstream typo -> gene symbol
TARGET = f"{OUT}/cd8_reference_free_attribution.csv"


def build():
    a = ad.read_h5ad(f"{ATLASES}/{STEM}_attribution_scimilarity.h5ad")
    genes = np.asarray(a.var_names).astype(str)
    lab = a.obs["label"].astype(str).to_numpy()
    A = a.layers["attribution_ig"]
    rows = []
    for s in STATES:
        m = lab == s
        assert m.sum() > 0, f"no cells labelled {s!r}"
        X = A[m]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        v = np.abs(X).mean(axis=0).astype("float64")
        order = np.lexsort((genes, -v))          # value descending, then gene symbol ascending
        rank = np.empty(len(v), dtype="int64")
        rank[order] = np.arange(1, len(v) + 1)
        rows.append(pd.DataFrame({"state": RENAME.get(s, s), "gene": genes,
                                  "reference_free_mean_abs_attr": v, "reference_free_rank": rank}))
        print(f"  [{RENAME.get(s, s):12s}] {int(m.sum()):6d} cells", flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    new = build()
    new.to_csv(TARGET, index=False)
    print(f"\n[wrote] {TARGET}  ({len(new)} rows)")
