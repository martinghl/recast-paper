"""Supplementary Fig. 4c,d: RECAST panels of the 35 cell-cycle comparisons, recomputed with the package.

cell_cycle/select_recast_phase.py ranked genes for each comparison (cycling target cells against noncycling reference
cells, phase_contrasts/<name>/{counts.npz,labels.txt}) with pooled-count profiles (counts summed before
normalization and log1p), the dC > 0 gate and up to 200 genes, and wrote phase_recast/<name>/panel.csv; the
cell-cycle fractions of the figure are scored from those files.

Here the same comparison is one recast.attribute call with centroid="pseudobulk", and its list is compared with
the original panel over the original depth.

    python check_cell_cycle.py
Writes cell_cycle.csv."""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import common as C
import recast

CR = os.path.join(C.SB, "cell_cycle")


def main():
    enc = C.scimilarity()
    genes = C.scimilarity_genes()
    names = sorted(n for n in os.listdir(os.path.join(CR, "phase_contrasts"))
                   if os.path.exists(os.path.join(CR, "phase_contrasts", n, "counts.npz")))
    rows = []
    for n in names:
        d = os.path.join(CR, "phase_contrasts", n)
        X = sp.load_npz(os.path.join(d, "counts.npz")).tocsr()
        y = np.array([l.strip() for l in open(os.path.join(d, "labels.txt")) if l.strip()])
        assert X.shape == (len(y), len(genes)), (n, X.shape)
        a = ad.AnnData(X.astype("float32"))
        a.var_names = genes
        a.obs["grp"] = pd.Categorical(y)
        res = recast.attribute(enc, a, "grp", target="target", reference=["reference"], gate="dC",
                               centroid="pseudobulk", qc="off")
        stored = list(pd.read_csv(os.path.join(CR, "phase_recast", n, "panel.csv")).sort_values("rank").gene.astype(str))
        r = C.compare_lists(res.genes["target"], stored)
        rows.append(dict(contrast=n, n_target=int((y == "target").sum()), n_reference=int((y == "reference").sum()),
                         n_positive=int((res.dC["target"] > 0).sum()), **r))
        print(f"  {n}: {len(y)} cells, depth {r['depth']}, identical {r['identical']}, "
              f"first difference {r['first_diff_rank']}", flush=True)
        C.assert_package_only()
    R = pd.DataFrame(rows)
    print(f"\n{len(R)} comparisons; identical {int(R.identical.sum())}, same top-20 set {int(R.same_set_top20.sum())}, "
          f"same top-200 set {int(R.get('same_set_top200', pd.Series(dtype=bool)).fillna(False).sum())}")
    C.write(R, "cell_cycle.csv")


if __name__ == "__main__":
    main()
