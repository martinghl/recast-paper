"""Marker rank@200 per subtype (Fig. 3a): where a method's top-200 list places the subtype's marker-panel genes.

Score per unit = mean over its panel genes of 100 * (1 - (p - 1) / 200) for a gene at position p of the top-200 list,
and 0 for a gene not in it (censored at rank 201); higher is better.  The mean rank (absent = 201, lower is better) is
written beside it.  Depth 200 for every method; seurat_roc returns fewer than 200 genes and is left out.

Units: the fine-tier units of RECALL_DIR/recall_by_subtype.csv (marker_recall_all.py).  Lists: RECAST's top-200 panels
(PANELS_DIR/topN200_<lineage>_recast_M0.csv) and the comparators' depth-200 lists (GENESETS200_DIR/<lineage>.csv).
Panels: UNIT_PANELS (ground_truth/panels/marker_panels.json), genes in the encoder vocabulary.
->  SCORE_OUT_DIR/marker_rank_by_subtype.csv   [method, lineage, state, unit, mrank200_pct, mean_rank]
Env analysis.  Run by run_scoring_all.sh.
"""
import json
import os
import re

import numpy as np
import pandas as pd

SB = "/path/to/research/analysis"
FULL_BENCHMARK = f"{SB}/subtype_benchmark/full_benchmark"
PAN = os.environ.get("PANELS_DIR", f"{FULL_BENCHMARK}/scoring/in/panels")
GS200 = os.environ.get("GENESETS200_DIR", f"{FULL_BENCHMARK}/scoring/in/genesets200")
OUTW = os.environ.get("SCORE_OUT_DIR", f"{FULL_BENCHMARK}/scoring/out")
RECALL_DIR = os.environ.get("RECALL_DIR", f"{FULL_BENCHMARK}/scoring/out")
UNIT_GT = json.load(open(os.environ.get("UNIT_PANELS", f"{FULL_BENCHMARK}/ground_truth/panels/marker_panels.json")))
GO = set(l.strip() for l in open(f"{SB}/subtype_benchmark/results/go_genes.txt") if l.strip())
K = 200
DROP = {"seurat_roc"}


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


u = pd.read_csv(f"{RECALL_DIR}/recall_by_subtype.csv")
u = u[u.tier.astype(str).str.startswith("fine")]
LIN = sorted(u.lineage.unique())
CANON = set((u.lineage + "|" + u.state).unique())
print(f"units: {len(CANON)} over {len(LIN)} lineages")


def markers_for(lin):
    fpan = pd.read_csv(f"{PAN}/topN200_{lin}_recast_M0.csv").sort_values(["cluster", "rank"])
    clusters = sorted(fpan.cluster.astype(str).unique())
    assert {k.split("|", 1)[1] for k in UNIT_GT if k.split("|", 1)[0] == lin} <= set(clusters), lin
    m = {c: (set(UNIT_GT.get(f"{lin}|{c}", [])) & GO) for c in clusters}
    valid = [c for c in clusters if m[c]]
    recast_gs = {str(c): list(g.gene.astype(str)) for c, g in fpan.groupby("cluster")}
    return valid, m, recast_gs


def score_list(ordered, markers):
    top = ordered[:K]; pos = {g: i + 1 for i, g in enumerate(top)}
    ranks = [pos[g] if g in pos else K + 1 for g in markers]      # censored: absent -> 201
    pct = float(np.mean([100.0 * (1 - (r - 1) / K) for r in ranks]))    # higher better, absent -> 0
    return pct, float(np.mean(ranks))                              # (pct, mean_rank)


def rows_for(lin, meth, gs, valid, markers):
    out = []
    for c in valid:
        unit = f"{lin}|{c}"
        if unit not in CANON: continue
        pct, mr = score_list(gs.get(str(c), []), markers[c])
        out.append((meth, lin, c, unit, pct, mr))
    return out


frows, drows = [], []                       # RECAST rows first, then the comparators
for lin in LIN:
    valid, markers, recast_gs = markers_for(lin)
    frows += rows_for(lin, "recast_M0", recast_gs, valid, markers)
    for meth, gs in field_sources(lin, valid):
        if meth not in DROP:
            drows += rows_for(lin, meth, gs, valid, markers)
df = pd.DataFrame(frows + drows, columns=["method", "lineage", "state", "unit", "mrank200_pct", "mean_rank"])
df.to_csv(f"{OUTW}/marker_rank_by_subtype.csv", index=False)
print("states:", df.unit.nunique(), "methods:", df.method.nunique(), "rows:", len(df))

agg = df.groupby("method").agg(pct=("mrank200_pct", "mean"), mean_rank=("mean_rank", "mean")).sort_values("pct", ascending=False)
print("\n=== marker rank@200 (pct higher = better; mean rank lower = better) ===")
for m in list(agg.index[:8]) + ["..."] + list(agg.index[-6:]):
    if m == "...": print("   ..."); continue
    r = agg.loc[m]; print(f"{m:40s} {r.pct:9.1f} {r.mean_rank:8.1f}")
for probe in ["recast_M0", "random", "cosg", "cepo"]:
    if probe in agg.index:
        r = agg.loc[probe]; rank = list(agg.index).index(probe) + 1
        print(f"  [{probe}] mr@200pct={r.pct:.1f} (rank {rank}/{len(agg)}), meanRank={r.mean_rank:.1f}")
