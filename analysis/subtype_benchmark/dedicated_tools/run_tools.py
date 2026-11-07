#!/usr/bin/env python
"""Four dedicated marker-gene selection tools on the 20 subtype lineages (Fig. 3, Supplementary Fig. 3).

Most of the 51 methods are generic differential-expression tests; Cepo and COSG are the purpose-built marker
selectors among them.  These four are purpose-built too, use the cluster labels, and return a ranked gene list per
cluster, as RECAST does.

  nsforest   NS-Forest v4.1   random-forest importance, re-ranked by binary expression score
  scmags     scMAGS 1.7       expression / silhouette filtering, then per-cluster marker selection
  rankcorr   RankCorr         sparse rank-based (Vargo & Gilbert), reference fork of picturedRocks
  scgenefit  scGeneFit 1.0.2  linear-programming label-aware selection, run one-vs-rest per target

Inputs: the per-lineage matrices the other methods were run on (fine_state_export_v2 / sce_export_v2:
<lin>_counts.mtx.gz genes x cells, _genes.txt, _labels.txt).  Each lineage file holds only that lineage's subtypes, so
one-vs-rest within a file is the sibling contrast of the benchmark.  Timing: one CPU thread, wall time of the
selection call only.

->  genesets200/<lin>__<method>.csv   method,cluster,rank,gene   (depth <= TOPN)
->  timing/<lin>__<method>.json       {lineage, method, seconds, n_cells, n_genes, n_clusters, depth}"""
import argparse, gzip, json, os, sys, time
import numpy as np, pandas as pd
import scipy.io as sio
import scipy.sparse as sp

TOPN = 200
HERE = os.path.dirname(os.path.abspath(__file__))
LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]


def load(sce_dir, lin):
    """counts (cells x genes, CSR), gene names, cell labels -- exactly the harness's inputs."""
    with gzip.open(f"{sce_dir}/{lin}_counts.mtx.gz", "rb") as fh:
        m = sio.mmread(fh)                      # genes x cells
    X = sp.csr_matrix(m.T)                      # cells x genes
    genes = np.array([l.strip() for l in open(f"{sce_dir}/{lin}_genes.txt")], dtype=object)
    labels = np.array([l.strip() for l in open(f"{sce_dir}/{lin}_labels.txt")], dtype=object)
    assert X.shape == (len(labels), len(genes)), (X.shape, len(labels), len(genes))
    return X, genes, labels


def lognorm(X):
    """CP10K + log1p, the standard input for the label-supervised selectors."""
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    Y = sp.diags(1e4 / lib) @ X
    Y.data = np.log1p(Y.data)
    return sp.csr_matrix(Y)


# ------------------------------------------------------------------ NS-Forest
def run_nsforest(X, genes, labels, workdir=None):
    import anndata as ad
    import nsforest as ns
    from nsforest import nsforesting
    A = ad.AnnData(X=lognorm(X).astype(np.float32),
                   obs=pd.DataFrame({"cluster": pd.Categorical(labels)},
                                    index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=list(genes)))
    A = ns.pp.prep_medians(A, "cluster")
    A = ns.pp.prep_binary_scores(A, "cluster")
    # NS-Forest ranks in two stages: random-forest importance -> top n_top_genes, then re-rank those
    # by binary expression score -> n_binary_genes.  Both are raised to TOPN so the method's own
    # ranking rule is read out to the depth this benchmark scores; n_genes_eval stays at its default
    # because the final step is an exhaustive 2^n combination search, not a ranking.  NS-Forest is
    # built to emit minimal panels (~6 genes), so its reachable depth is limited by its own
    # BinaryFirst gene pre-filter -- whatever depth it reaches is recorded, never padded.
    out_dir = workdir or os.path.join(HERE, "nsf_tmp")
    os.makedirs(out_dir, exist_ok=True)
    # gene_selection=None is NS-Forest's own documented "include all genes" setting.  Its v4 default
    # (BinaryFirst_high, mean + 2 sd) pre-filters to ~10-20 candidates per cluster, which would cap
    # this method's ranking at a depth the benchmark's top-20 and top-200 endpoints cannot read; the
    # unfiltered setting is the configuration that gives NS-Forest its deepest own-rule ranking.
    res = nsforesting.NSForest(A, "cluster", n_trees=1000, n_jobs=1, gene_selection=None,
                               n_top_genes=TOPN, n_binary_genes=TOPN, n_genes_eval=6,
                               save=False, output_folder=out_dir + "/", outputfilename_prefix="nsf")
    col = next(c for c in ("binary_genes", "NSForest_markers", "markers") if c in res.columns)
    out = {}
    for _, r in res.iterrows():
        g = r[col]
        if isinstance(g, str):
            g = eval(g) if g.strip().startswith("[") else [g]
        out[str(r["clusterName"])] = [str(x) for x in list(g)][:TOPN]
    return out


# ------------------------------------------------------------------ scMAGS
def run_scmags(X, genes, labels, workdir=None):
    from scmags import ScMags
    M = ScMags(data=sp.csr_matrix(X), labels=np.asarray(labels, dtype=str),
               gene_ann=np.asarray(genes, dtype=str), verbose=False)
    M.filter_genes(nof_sel=TOPN + 50)           # candidate pool must exceed the requested depth
    M.sel_clust_marker(nof_markers=TOPN, n_cores=1)
    mk = M.get_markers()
    out = {}
    for cl in mk.index:
        gl = [str(g) for g in mk.loc[cl].dropna().tolist() if str(g) not in ("nan", "")]
        out[str(cl).replace("C_", "", 1)] = gl[:TOPN]
    return out


# ------------------------------------------------------------------ RankCorr
def run_rankcorr(X, genes, labels, workdir=None):
    """RankCorr per cluster, ranked by its own sparse-hyperplane coefficient.

    `Rocks.CSrankMarkers` returns `support2list(marks)` -- the union of every cluster's support,
    de-duplicated and unordered -- so the public call throws away both the cluster assignment and
    the ranking this benchmark needs.  The per-cluster coefficient vector it computes internally
    carries both, so this reproduces that loop (picturedRocks, rocks.py:751-792) and keeps `curr`: the support is
    RankCorr's marker set for that cluster and |curr| is its own weight on each marker.

    `lamb` is RankCorr's sparsity knob; larger lamb keeps more genes.  It is swept upward until
    every cluster's support reaches TOPN.  If the sweep tops out, the natural depth is recorded.
    """
    sys.path.insert(0, f"{HERE}/rankcorr_src")
    import picturedRocks.rocks as rk
    from picturedRocks import Rocks
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels], dtype=int)

    # RankCorr's normalize() densifies (sparse / dense-column -> np.matrix); same arithmetic,
    # kept sparse: scale each cell to the median library size, then log(x+1).
    Xn = sp.csr_matrix(X).astype(float)
    cellsize = np.asarray(Xn.sum(1)).ravel()
    target = np.median(cellsize)
    cellsize[cellsize == 0] = 1.0
    Xn = sp.diags(target / cellsize) @ Xn
    Xn.data = np.log(Xn.data + 1.0)

    R = Rocks(sp.csc_matrix(Xn), y)
    R.CSuseRanks(True)
    R._CSrankConsts()

    best, lamb = None, 8.0
    for _ in range(12):
        cand = {}
        for ci, cl in enumerate(classes):
            consts = np.asarray(R.cs_OvA[ci]).flatten()
            absConsts = np.flipud(np.sort(np.abs(consts)))
            switchInd = R.findNormChanges(consts, absConsts, s=lamb)
            if not switchInd:
                cand = None
                break
            curr, _ = rk.rescale(rk.softThreshold(consts.copy(), absConsts[switchInd[0]]),
                                 np.sqrt(lamb), 1.0)
            sup = curr.nonzero()[0]
            order = sup[np.argsort(-np.abs(curr[sup]))][:TOPN]
            cand[cl] = [str(genes[i]) for i in order]
        if cand is None:
            lamb *= 2.0
            continue
        if best is None or min(len(v) for v in cand.values()) > min(len(v) for v in best.values()):
            best = cand
        if min(len(v) for v in cand.values()) >= TOPN:
            break
        lamb *= 2.0
    return best


# ------------------------------------------------------------------ scGeneFit
def run_scgenefit(X, genes, labels, workdir=None):
    from scGeneFit.functions import get_markers
    D = np.asarray(lognorm(X).todense(), dtype=np.float64)
    out = {}
    for cl in sorted(set(labels)):
        bl = np.array([1 if l == cl else 0 for l in labels])
        idx = get_markers(D, bl, num_markers=min(TOPN, D.shape[1] - 1),
                          method="centers", epsilon=1, verbose=False)
        out[str(cl)] = [str(genes[i]) for i in list(idx)][:TOPN]
    return out


RUNNERS = {"nsforest": run_nsforest, "scmags": run_scmags,
           "rankcorr": run_rankcorr, "scgenefit": run_scgenefit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sce", default=f"{HERE}/sce_export_v2")
    ap.add_argument("--out", default=f"{HERE}/genesets200")
    ap.add_argument("--timing", default=f"{HERE}/timing")
    ap.add_argument("--methods", default=",".join(RUNNERS))
    ap.add_argument("--lineages", default=",".join(LIN20))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.timing, exist_ok=True)

    for lin in a.lineages.split(","):
        X, genes, labels = load(a.sce, lin)
        for meth in a.methods.split(","):
            dst = f"{a.out}/{lin}__{meth}.csv"
            if os.path.exists(dst):
                print(f"[skip] {lin} {meth}", flush=True); continue
            print(f"[run ] {lin} {meth}  cells={X.shape[0]} genes={X.shape[1]} "
                  f"clusters={len(set(labels))}", flush=True)
            t0 = time.perf_counter()
            try:
                res = RUNNERS[meth](X, genes, labels, workdir=f"{a.timing}/../work_{lin}_{meth}")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[FAIL] {lin} {meth}: {type(e).__name__}: {e}", flush=True)
                continue
            sec = time.perf_counter() - t0
            if not res:
                print(f"[FAIL] {lin} {meth}: empty result", flush=True); continue
            rows = [(meth, cl, i + 1, g) for cl, gl in res.items() for i, g in enumerate(gl)]
            pd.DataFrame(rows, columns=["method", "cluster", "rank", "gene"]).to_csv(dst, index=False)
            depth = min((len(v) for v in res.values()), default=0)
            json.dump({"lineage": lin, "method": meth, "seconds": sec, "n_cells": int(X.shape[0]),
                       "n_genes": int(X.shape[1]), "n_clusters": len(set(labels)),
                       "min_depth": depth, "max_depth": max((len(v) for v in res.values()), default=0)},
                      open(f"{a.timing}/{lin}__{meth}.json", "w"), indent=1)
            print(f"[ok  ] {lin} {meth}  {sec:.1f}s  depth {depth}", flush=True)


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    main()
