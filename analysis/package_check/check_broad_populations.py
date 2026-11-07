"""Supplementary Fig. 4a,b: RECAST panels of the broad populations of eight atlases, recomputed with the package.

broad_populations/recast_panels.py (seven atlases with expert panels) and recast_auroc.py
(CITE-seq) selected the top 50 genes of every population with at least 10 cells, against all other cells of the
atlas, with recast.cluster_attribution through a wrapper that log-normalized the counts before embedding; the
expert-panel recall (panel a) and the signature AUROC (panel b) are computed from those saved lists.

Here the same call is made with the package's SCimilarityEncoder, which log-normalizes raw counts itself, and the
lists are compared with recast_broad_panels.csv and recast_broad_panels_citeseq.csv.

    python check_broad_populations.py
Writes broad_populations.csv."""
import os

import numpy as np
import pandas as pd
import anndata as ad

import common as C
import recast

RC = os.path.join(C.SB, "broad_populations", "results_realcounts")
BROAD8 = ["pbmc3k", "ren", "stephenson", "citeseq", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]


def main():
    enc = C.scimilarity()
    genes = C.scimilarity_genes()
    stored = pd.concat([pd.read_csv(os.path.join(RC, f)) for f in
                     ("recast_broad_panels.csv", "recast_broad_panels_citeseq.csv")], ignore_index=True)
    rows = []
    for ds in BROAD8:
        names, obs = C.attribution_obs(C.ATLASES, ds)
        lab = obs["label"].astype(str).to_numpy()
        clusters = sorted(c for c in np.unique(lab) if (lab == c).sum() >= 10)
        keep = np.isin(lab, clusters)
        a = ad.AnnData(C.counts(C.ATLASES, ds, names[keep], genes, integer=ds != "lawlor"))
        a.var_names = genes
        a.obs["label"] = lab[keep]
        res = recast.cluster_attribution(enc, a, "label", reference="rest", gate="dC", centroid="mean_lognorm",
                                         qc="silent")
        P = stored[stored.dataset == ds]
        assert sorted(P.cluster.astype(str).unique()) == clusters, ds
        for c in clusters:
            ref = list(P[P.cluster.astype(str) == c].sort_values("rank").gene.astype(str))
            rows.append(dict(dataset=ds, cluster=c, n_cells=int((lab == c).sum()),
                             **C.compare_lists(res.top(c, len(ref)), ref)))
        R = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        print(f"  {ds}: {len(clusters)} populations, {int(keep.sum())} cells; identical {int(R.identical.sum())}, "
              f"same top-20 set {int(R.same_set_top20.sum())}", flush=True)
        C.assert_package_only()
    R = pd.DataFrame(rows)
    print(f"\nall: {len(R)} populations; identical top-50 {int(R.identical.sum())}, "
          f"same top-20 set {int(R.same_set_top20.sum())}")
    if not R.identical.all():
        print(R[~R.identical][["dataset", "cluster", "first_diff_rank", "overlap_top20"]].to_string(index=False))
    C.write(R, "broad_populations.csv")


if __name__ == "__main__":
    main()
