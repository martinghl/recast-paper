#!/usr/bin/env python
"""RECAST against the other methods on every endpoint of the subtype marker benchmark, for all units and for subsets.

For each endpoint and unit subset: RECAST's mean, its rank among all methods (mean over the subset's units), and how
many methods RECAST beats / cannot be told from / loses to.  The comparison rule is the one the open circles of Fig. 3
use (code/common/marker_benchmark_panels.py, is_ns): paired bootstrap of (RECAST - method) over matched units, 10,000
resamples from numpy default_rng(0) shared by every method of an endpoint, 95% percentile interval, selection time on
log10.  "Beats" = the interval excludes 0 in RECAST's favour.

Subsets: all units; units of the original and of the added lineages; tiers (fine_immune / fine_epithelial / fine_brain,
from the recall table).  Selection time is per lineage and covers the original lineages only, so it is reported for the
whole set alone.
Env analysis.  Usage: python -I summary.py [--score DIR]   ->  <score>/summary.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
sys.path.insert(0, f"{FULL_BENCHMARK}/added_lineages")
import added_datasets as D  # noqa: E402

BLOCKS = ["recall20", "auroc", "marker_rank200", "contamination", "ms_per_cell"]
LOWER_BETTER = {"contamination", "ms_per_cell"}
B, SEED = 10000, 0


def compare(piv, block):
    """-> (n_units, recast mean, recast rank, beats, ns, loses) on the units RECAST has."""
    piv = piv.dropna(subset=["recast_M0"])
    means = piv.mean()
    rank = int((means < means["recast_M0"]).sum() + 1) if block in LOWER_BETTER else \
        int((means > means["recast_M0"]).sum() + 1)
    tx = np.log10 if block == "ms_per_cell" else (lambda v: v)
    idx_cache = {}
    beats = ns = loses = 0
    for m in piv.columns:
        if m == "recast_M0":
            continue
        sub = piv[["recast_M0", m]].dropna()
        if len(sub) < 3:
            continue
        d = tx(sub["recast_M0"].to_numpy(float)) - tx(sub[m].to_numpy(float))
        if block in LOWER_BETTER:
            d = -d                                   # d > 0 always favours RECAST
        if np.allclose(d, 0):
            ns += 1
            continue
        n = len(d)
        if n not in idx_cache:
            idx_cache[n] = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        bm = d[idx_cache[n]].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        if lo > 0:
            beats += 1
        elif hi < 0:
            loses += 1
        else:
            ns += 1
    return len(piv), float(means["recast_M0"]), rank, len(means), beats, ns, loses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=f"{FULL_BENCHMARK}/scoring/out")
    a = ap.parse_args()
    dots = pd.read_csv(f"{a.score}/marker_benchmark_by_subtype.csv")
    rec = pd.read_csv(f"{a.score}/recall_by_subtype.csv")
    tier = rec.drop_duplicates("lineage").set_index("lineage")["tier"].to_dict()
    new = set(D.NEW_IDS)
    lin_of = lambda u: u.split("|")[0]
    subsets = {"all": lambda u: True,
               "original lineages": lambda u: lin_of(u) not in new,
               "added lineages": lambda u: lin_of(u) in new}
    for t in ["fine_immune", "fine_epithelial", "fine_brain"]:
        subsets[t] = (lambda t: lambda u: tier.get(lin_of(u)) == t)(t)
    rows = []
    for sname, keep in subsets.items():
        for b in BLOCKS:
            if b == "ms_per_cell" and sname != "all":
                continue
            d = dots[dots.block == b]
            d = d[d.unit.map(keep)]
            if d.empty:
                continue
            piv = d.pivot_table(index="unit", columns="method", values="value")
            n, mu, r, nm, w, t_, l = compare(piv, b)
            rows.append(dict(subset=sname, block=b, n_units=n, recast=round(mu, 4),
                             rank=r, n_methods=nm, beats=w, ns=t_, loses=l))
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.score}/summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
