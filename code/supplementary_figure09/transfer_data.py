"""Data of Supplementary Fig. 9b: the transfer of marker panels between atlases, per method and comparison.

data/transfer_pairs.csv: for each method, endpoint and directed comparison (compartment, source atlas, target atlas),
the value of the source atlas's 20 genes in the target atlas.  load_transfer() keeps the 51 methods, renames RECAST's
row to the id the benchmark module uses (recast_M0) and labels each comparison; transfer_tables() gives the per-method
means and the long table the boxes are drawn from (the shapes of the marker benchmark tables, so the box-drawing code
of common/marker_benchmark_panels.py draws them unchanged); transfer_groups() orders the rows.
"""
import os

import pandas as pd

import marker_benchmark_panels as F2      # method families and display names

HERE = os.path.dirname(os.path.abspath(__file__))

RECAST_SRC, RECAST_ID = "RECAST_M0", "recast_M0"
TR_DROP = ("seurat_roc",)   # 51 methods: the table's 52 minus Seurat roc, which returns about 6 genes

TR_BLOCKS = [("tr_sum_max", "sum_max"), ("tr_auroc", "auroc"),
          ("tr_ontarget", "ontarget"), ("tr_knn", "knn")]

# All four endpoints are rates on (0, 1).  Only subtype AUROC has a chance line: the kNN metric is scored over a varying
# number of target subtypes per comparison, so no single chance value fits that column.
TR_METRICS = {
    "tr_sum_max":  dict(short="sum-max",  title="Sum-max\nspecificity",   higher=True,
                        axis=(0.0, 1.0), box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=None),
    "tr_auroc":    dict(short="AUROC",    title="Subtype\nAUROC",         higher=True,
                        axis=(0.0, 1.0), box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=0.5),
    "tr_ontarget": dict(short="on-target", title="On-target\nrecovery",   higher=True,
                        axis=(0.0, 1.0), box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=None),
    "tr_knn":      dict(short="kNN",      title="kNN label\nprediction",  higher=True,
                        axis=(0.0, 1.0), box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=None),
}

# RECAST's mean and rank (of 51) on the four endpoints, as reported in the text; 01_draw.py asserts them.
TR_PUBLISHED = {"tr_sum_max": (0.725, 5), "tr_auroc": (0.895, 6),
             "tr_ontarget": (0.766, 8), "tr_knn": (None, 14)}


def load_transfer():
    d = pd.read_csv(f"{HERE}/data/transfer_pairs.csv")
    d = d[~d.method.isin(TR_DROP)].copy()
    d["method"] = d.method.replace({RECAST_SRC: RECAST_ID})
    # the table lists every "random" row twice (identical values); keep one, so each box has 42 comparisons
    n0 = len(d)
    d = d.drop_duplicates(["method", "metric", "compartment", "src", "tgt"])
    d["unit"] = d.compartment + "|" + d.src + ">" + d.tgt
    return d, n0 - len(d)


def transfer_tables(d):
    dots = pd.concat([pd.DataFrame(dict(method=s.method, block=key, unit=s.unit, value=s.transfer))
                      for key, met in TR_BLOCKS for s in (d[d.metric == met],)], ignore_index=True)
    master = dots.pivot_table(index="method", columns="block", values="value", aggfunc="mean")
    return master, dots


def transfer_groups(master):
    """Methods grouped by family (as in Supplementary Fig. 3), by mean subtype AUROC within a family."""
    g = {f: [] for f in F2.FAM_ORDER}
    for mid in master.index:
        g[F2.family_of(mid)].append(mid)
    for f in g:
        g[f] = sorted(g[f], key=lambda m: -master.loc[m, "tr_auroc"])
    return [(f, g[f]) for f in F2.FAM_ORDER if g[f]]
