"""Marker-independent metrics of the RECAST panels of the broad populations:
one-vs-rest AUROC, on-target recovery, k-NN and SVM classification and sum-max specificity, at the k the
manuscript reports, with the evaluator of the comparator methods on `Xln = att.X` (the log-normalized matrix).

The eight atlases include the CITE-seq cord-blood data, whose panels are computed here on the GPU with the library
call of recast_panels.py.  Cells, labels, clusters, `Xln`, the >= 10-cell cluster filter, SEED_CV and the k grid are
those of the comparators; only the ranked gene list differs.

Output: results_realcounts/.

    CUDA_VISIBLE_DEVICES=1 /path/to/envs/SC/bin/python recast_auroc.py"""
import os
import sys

# This box has 344 cores and the conda OpenBLAS is precompiled for fewer. Left alone it prints
# "precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata" and then segfaults
# (exit 139) inside the first LinearSVC. Cap the pools BEFORE numpy is imported -- after import the
# variables are ignored.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path:
        sys.path.insert(0, p)

from subtype_benchmark.select_recast import _load_raw                  # noqa: E402
from scoring_benchmark.datasets import _go_gene_order, SCIMILARITY_MODEL    # noqa: E402

ATLASES = f"{SB}/atlases"
OUT = f"{SB}/broad_populations/results_realcounts"
BROAD7 = ["pbmc3k", "ren", "stephenson", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]
BROAD8 = ["pbmc3k", "ren", "stephenson", "citeseq", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]
KS = [3, 5, 10, 20]                 # as in the broad-population benchmark
SEED_CV = 20220502                  # as in the broad-population benchmark
MIN_CELLS = 10
TOPN = 50


def recast_panels(ds, enc, go):
    """Corrected top-50 panels for one atlas -- the same call recast_panels.py makes."""
    import recast
    att = ad.read_h5ad(f"{ATLASES}/{ds}_attribution_scimilarity.h5ad", backed="r")
    obs = np.asarray(att.obs_names).astype(str)
    lab = att.obs["label"].astype(str).to_numpy()
    del att
    clusters = sorted([c for c in np.unique(lab) if (lab == c).sum() >= MIN_CELLS])
    keep = np.isin(lab, clusters)
    a = ad.AnnData(_load_raw(ds, obs[keep]))
    a.var_names = go
    a.obs["label"] = lab[keep]
    ar = recast.cluster_attribution(enc, a, "label", reference="rest", gate="dC",
                                   centroid="mean_lognorm")
    return {c: [str(g) for g in ar.top(c, k=TOPN)] for c in clusters}


def score_arm3(ds, panels):
    """Marker-independent metrics of the RECAST panels, with the comparators' evaluator."""
    att = ad.read_h5ad(f"{ATLASES}/{ds}_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    genes = np.asarray(att.var_names).astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    y = att.obs["label"].astype(str).to_numpy()
    del att
    clusters = sorted([c for c in np.unique(y) if (y == c).sum() >= MIN_CELLS])

    def mf1(yp):
        return float(np.median([f1_score(y == c, yp == c) for c in clusters]))

    def knn_svm(gs, kind):
        idx = sorted({gi[g] for c in clusters for g in gs.get(c, []) if g in gi})
        if not idx:
            return np.nan
        Xf = Xln[:, idx]
        oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xf, y):
            clf = KNeighborsClassifier(n_neighbors=3) if kind == "knn" else LinearSVC(max_iter=5000)
            clf.fit(Xf[tr], y[tr])
            oof[te] = clf.predict(Xf[te])
        return mf1(oof)

    def maxsum(gs):
        cols = {c: [gi[g] for g in gs.get(c, []) if g in gi] for c in clusters}
        if not any(cols.values()):
            return np.nan
        oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xln, y):
            mu = Xln[tr].mean(0); sd = Xln[tr].std(0) + 1e-9
            Zs = (Xln - mu) / sd
            S = np.column_stack([Zs[:, cols[c]].sum(1) if cols[c] else np.full(len(y), -np.inf)
                                 for c in clusters])
            oof[te] = np.asarray(clusters)[S.argmax(1)][te]
        return mf1(oof)

    def ontarget(gs):
        v = [Xln[y == c, j].mean() / (Xln[y == c, j].mean() + Xln[y != c, j].mean() + 1e-12)
             for c in clusters for j in [gi[g] for g in gs.get(c, []) if g in gi]]
        return float(np.mean(v)) if v else np.nan

    def auroc(gs):
        a = []
        for c in clusters:
            idx = [gi[g] for g in gs.get(c, []) if g in gi]
            if idx:
                a.append(roc_auc_score(y == c, Xln[:, idx].mean(1)))
        return float(np.mean(a)) if a else np.nan

    # Scope: 2C consumes `auroc` and `knn` at k=20 and nothing else
    # (broad_tables.py:75-76). auroc and ontarget are cheap and computed across the whole k
    # grid; knn is 5-fold on up to 46k cells, so it runs only at the k that is actually read. svm
    # (LinearSVC over the union panel) and sum_max (which re-standardises all 28k genes per fold)
    # are left out: nothing downstream reads them, and they are what made this job segfault.
    rows = []
    for k in KS:
        gs = {c: v[:k] for c, v in panels.items()}
        rows.append(dict(dataset=ds, method="m0", k=k,
                         knn=knn_svm(gs, "knn") if k == 20 else np.nan,
                         svm=np.nan, sum_max=np.nan,
                         ontarget=ontarget(gs), auroc=auroc(gs)))
        print(f"  [{ds}] k={k:2d}  auroc={rows[-1]['auroc']:.4f}"
              + (f"  knn={rows[-1]['knn']:.4f}" if k == 20 else ""), flush=True)
    return rows


def main():
    import recast
    from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder
    os.makedirs(OUT, exist_ok=True)
    go = _go_gene_order()

    # BROAD-7 panels already exist from recast_panels.py -- reuse rather than recompute, so the
    # two halves of this correction cannot drift apart.
    have = pd.read_csv(f"{OUT}/recast_broad_panels.csv")
    panels = {ds: {c: list(g.sort_values("rank").gene.astype(str))
                   for c, g in d.groupby("cluster")}
              for ds, d in have.groupby("dataset")}
    assert set(panels) == set(BROAD7), sorted(panels)

    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda"))
    for ds in BROAD8:
        if ds not in panels:                       # citeseq -- 2C only, never scored for 2B
            print(f"[{ds}] computing corrected panels (not in the 2B seven)", flush=True)
            panels[ds] = recast_panels(ds, enc, go)
            pd.DataFrame([dict(dataset=ds, cluster=c, rank=i + 1, gene=g)
                          for c, gl in panels[ds].items() for i, g in enumerate(gl)]
                         ).to_csv(f"{OUT}/recast_broad_panels_{ds}.csv", index=False)
    del enc

    rows = []
    for ds in BROAD8:
        rows += score_arm3(ds, panels[ds])
    arm3 = pd.DataFrame(rows)
    arm3.to_csv(f"{OUT}/recast_broad_specificity.csv", index=False)


if __name__ == "__main__":
    main()
