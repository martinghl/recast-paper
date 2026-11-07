"""Contamination@200 per subtype (Fig. 3a): ribosomal and mitochondrial genes in each method's top-200 list.

Gene sets (HGNC nomenclature):
  ribosomal     = symbols with root RPS / RPL / MRPS / MRPL, excluding the RPS6K* subfamily (HGNC "Ribosomal protein
                  S6 kinases": signalling, not structural ribosomal proteins);
  mitochondrial = ^MT- (the mitochondrial genome's genes).
Depth 200 for every method; seurat_roc returns fewer than 200 genes and is left out.  Lower is cleaner.

Units: the fine-tier units of RECALL_DIR/recall_by_subtype.csv (marker_recall_all.py).  Lists: RECAST's top-200 panels
(PANELS_DIR/topN200_<lineage>_recast_M0.csv) and the comparators' depth-200 lists (GENESETS200_DIR/<lineage>.csv).
->  SCORE_OUT_DIR/contamination_by_subtype.csv   [method, lineage, state, unit, ribo_n, mt_n, contam_n]
    (contam_n = ribo_n + mt_n)
Env analysis.  Run by run_scoring_all.sh.
"""
import os
import re

import pandas as pd

SB = "/path/to/research/analysis"
FULL_BENCHMARK = f"{SB}/subtype_benchmark/full_benchmark"
PAN = os.environ.get("PANELS_DIR", f"{FULL_BENCHMARK}/scoring/in/panels")
GS200 = os.environ.get("GENESETS200_DIR", f"{FULL_BENCHMARK}/scoring/in/genesets200")
OUTW = os.environ.get("SCORE_OUT_DIR", f"{FULL_BENCHMARK}/scoring/out")
RECALL_DIR = os.environ.get("RECALL_DIR", f"{FULL_BENCHMARK}/scoring/out")
K = 200
DROP = {"seurat_roc"}

RIBO = re.compile(r"^(RP[SL]|MRP[SL])")     # ribosomal-protein roots (HGNC)
RPS6K = re.compile(r"^RPS6K")               # ...minus the S6-kinase subfamily (separate HGNC group)
MITO = re.compile(r"^MT-")                  # mitochondrial-genome genes (HGNC prefix)


def make_names(s):
    """R's make.names(): every character outside [A-Za-z0-9._] becomes a dot."""
    return re.sub(r"[^A-Za-z0-9._]", ".", str(s))


def field_sources(lin, clusters=()):
    """-> [(method, {cluster: [ranked genes]})] for every comparator of this lineage.  Cluster keys renamed by R's
    make.names in the harness ("Naive.B.cells") are mapped back to the subtype names in `clusters`."""
    d = pd.read_csv(f"{GS200}/{lin}.csv").sort_values(["method", "cluster", "rank"])
    out = []
    for m, gm in d.groupby("method"):
        g = {str(c): list(gg.gene.astype(str)) for c, gg in gm.groupby("cluster")}
        for want in clusters:
            want = str(want)
            if want in g: continue
            hits = [k for k in g if k == make_names(want)]
            if len(hits) == 1: g[want] = g.pop(hits[0])
        out.append((str(m), g))
    return out


def is_ribo(g): return bool(RIBO.match(g)) and not RPS6K.match(g)
def is_mt(g):   return bool(MITO.match(g))


u = pd.read_csv(f"{RECALL_DIR}/recall_by_subtype.csv")
u = u[u.tier.astype(str).str.startswith("fine")]
LIN = sorted(u.lineage.unique())
CANON = set((u.lineage + "|" + u.state).unique())
print(f"units: {len(CANON)} over {len(LIN)} lineages")


def clusters_for(lin):
    fpan = pd.read_csv(f"{PAN}/topN200_{lin}_recast_M0.csv").sort_values(["cluster", "rank"])
    clusters = sorted(fpan.cluster.astype(str).unique())
    recast_gs = {str(c): list(g.gene.astype(str)) for c, g in fpan.groupby("cluster")}
    return clusters, recast_gs


def count_top(ordered):
    top = ordered[:K]
    rn = sum(1 for g in top if is_ribo(str(g)))
    mn = sum(1 for g in top if is_mt(str(g)))
    return rn, mn


def rows_for(lin, meth, gs, clusters):
    out = []
    for c in clusters:
        unit = f"{lin}|{c}"
        if unit not in CANON: continue
        rn, mn = count_top(gs.get(str(c), []))
        out.append((meth, lin, c, unit, rn, mn, rn + mn))
    return out


def main():
    frows, drows = [], []                   # RECAST rows first, then the comparators
    for lin in LIN:
        clusters, recast_gs = clusters_for(lin)
        frows += rows_for(lin, "recast_M0", recast_gs, clusters)
        for meth, gs in field_sources(lin, clusters):
            if meth not in DROP:
                drows += rows_for(lin, meth, gs, clusters)
    df = pd.DataFrame(frows + drows, columns=["method", "lineage", "state", "unit", "ribo_n", "mt_n", "contam_n"])
    df.to_csv(f"{OUTW}/contamination_by_subtype.csv", index=False)
    print("states:", df.unit.nunique(), "methods:", df.method.nunique(), "rows:", len(df))
    ag = df.groupby("method").contam_n.mean().sort_values()
    print("\n=== mean contamination count @top-200 (lower=cleaner) ===")
    for m in list(ag.index[:6]) + ["..."] + list(ag.index[-6:]):
        if m == "...": print("   ..."); continue
        r = df[df.method == m]
        print(f"  {m:40s} contam={ag[m]:5.2f}  (ribo {r.ribo_n.mean():.2f} + mt {r.mt_n.mean():.2f})")
    for p in ["recast_M0", "random", "cosg"]:
        if p in ag.index:
            print(f"  [{p}] contam={ag[p]:.2f}  rank {list(ag.index).index(p)+1}/{len(ag)} (1=cleanest)")


if __name__ == "__main__":
    main()
