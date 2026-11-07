"""RECAST panels of the broad populations of each atlas and their curated-marker recall.

The same library call as subtype_benchmark/select_recast.py: the SCimilarity encoder wrapper, centroid
"mean_lognorm", raw counts from `<stem>_fullgene.h5ad` (`_load_raw`; `layers["counts"]` of the attribution files
holds log-normalized values).  Cells, labels and clusters come from the attribution file's obs; clusters with
>= 10 cells.  Recall: n = max(|E_c|, 10), recall = |top-n  n  E_c| / |E_c|, expert sets from
expert_markers_long.csv.  The panels are written as well.

Output: results_realcounts/.

    CUDA_VISIBLE_DEVICES=1 /path/to/envs/SC/bin/python recast_panels.py"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path:
        sys.path.insert(0, p)

from subtype_benchmark.select_recast import _load_raw          # noqa: E402
from scoring_benchmark.datasets import _go_gene_order, SCIMILARITY_MODEL  # noqa: E402

ATLASES = f"{SB}/atlases"
OUT = f"{SB}/broad_populations/results_realcounts"
BROAD7 = ["pbmc3k", "ren", "stephenson", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]
TOPN = 50
MIN_CELLS = 10          # clusters with >= 10 cells, as in the broad-population benchmark


def expert_sets(ds):
    e = pd.read_csv(f"{ATLASES}/broad_markers/expert_markers_long.csv")
    e = e[e.dataset == ds]
    out = {ct: {str(g) for g in grp.gene} for ct, grp in e.groupby("cell_type")}
    return {ct: s for ct, s in out.items() if s}


def main():
    import recast
    from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda"))
    go = _go_gene_order()
    os.makedirs(OUT, exist_ok=True)

    panels_all, arm2 = [], []
    for ds in BROAD7:
        att = ad.read_h5ad(f"{ATLASES}/{ds}_attribution_scimilarity.h5ad", backed="r")
        obs = np.asarray(att.obs_names).astype(str)
        lab = att.obs["label"].astype(str).to_numpy()
        del att
        clusters = sorted([c for c in np.unique(lab) if (lab == c).sum() >= MIN_CELLS])
        keep = np.isin(lab, clusters)
        Craw_go = _load_raw(ds, obs[keep])            # REAL counts, aligned to the encoder vocabulary
        a = ad.AnnData(Craw_go)
        a.var_names = go
        a.obs["label"] = lab[keep]
        print(f"[{ds}] {keep.sum()} cells, {len(clusters)} clusters", flush=True)

        ar = recast.cluster_attribution(enc, a, "label", reference="rest", gate="dC",
                                       centroid="mean_lognorm")
        panels = {c: [str(g) for g in ar.top(c, k=TOPN)] for c in clusters}
        panels_all += [dict(dataset=ds, cluster=c, rank=i + 1, gene=g)
                       for c in clusters for i, g in enumerate(panels[c])]

        EXP = expert_sets(ds)
        for c in clusters:
            if c not in EXP:
                continue
            Ec = EXP[c]
            n = max(len(Ec), MIN_CELLS)
            arm2.append(dict(dataset=ds, method="m0", cluster=c, n_expert=len(Ec),
                             recall=len(set(panels[c][:n]) & Ec) / len(Ec)))
        print(f"[{ds}] done, {sum(1 for r in arm2 if r['dataset'] == ds)} scored clusters", flush=True)

    pd.DataFrame(panels_all).to_csv(f"{OUT}/recast_broad_panels.csv", index=False)
    pd.DataFrame(arm2).to_csv(f"{OUT}/recast_broad_marker_recall.csv", index=False)


if __name__ == "__main__":
    main()
