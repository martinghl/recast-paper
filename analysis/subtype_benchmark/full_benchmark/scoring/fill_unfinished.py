#!/usr/bin/env python
"""Score scGeneFit's unfinished subtypes as the worst value any method attained there.

scGeneFit's linear program did not finish for seven units (MISSING): two of the original lineages
(fs_asc_zhao|cycling ASC, fs_nk_pbmcbmn|NK_CD56bright; the other subtypes there took a median of 233 s on one thread,
these two ran past 6.5 h) and five of the added lineages, stopped at the same cap.  Leaving them empty would put
scGeneFit on a different unit set from every other method.  Instead each missing value is set to the worst value any
method attained on that unit and endpoint, so the fill can only lower scGeneFit's scores and is bounded by observed
values.

The scorers write a placeholder for a subtype whose gene list is absent; for contamination that placeholder is an
empty list scoring 0, the cleanest value.  This script overwrites those rows.

Selection time: for an unfinished subtype on a lineage of the timing block, scGeneFit's lineage time gains the
single-thread seconds the subtype had used when it was stopped (STALL_SECONDS), a lower bound on its cost.  The added
lineages are not in the timing block.

Only rows with method == scgenefit change.
->  rewrites marker_benchmark_by_subtype.csv and the scgenefit row of marker_benchmark_means.csv in --score.
Env analysis.  Run by run_scoring_all.sh.
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = "/path/to/research/analysis/subtype_benchmark/dedicated_tools"
MISSING = ["fs_asc_zhao|cycling ASC", "fs_nk_pbmcbmn|NK_CD56bright",
           "fs37_neuron_gittings|excitatory", "fs37_neuron_maitra|excitatory", "fs37_neuron_rexach|excitatory",
           "fs37_glia_rexach|astrocyte", "fs37_gut_hickey|enterocyte"]
STALL_SECONDS = 24053             # single-thread seconds each unfinished subtype had used when it was stopped
LOWER_BETTER = {"contamination", "ms_per_cell"}
METHOD = "scgenefit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=f"{HERE}/out")
    ap.add_argument("--timing-detail", default=f"{TOOLS}/score_out/timing_detail.csv")
    a = ap.parse_args()
    print(f"units filled: {MISSING}")

    dpath = f"{a.score}/marker_benchmark_by_subtype.csv"
    mpath = f"{a.score}/marker_benchmark_means.csv"
    dots = pd.read_csv(dpath)
    blocks = [b for b in dots.block.unique() if b != "ms_per_cell"]

    print(f"{'block':16s} {'unit':32s} {'worst value':>14s} {'from':>28s}  action")
    added = replaced = 0
    for b in blocks:
        d = dots[dots.block == b]
        for unit in MISSING:
            u = d[d.unit == unit]
            if u.empty:
                print(f"{b:16s} {unit:32s} {'-':>14s} {'-':>28s}  SKIP (unit absent from block)")
                continue
            other = u[u.method != METHOD]
            worst = other.value.max() if b in LOWER_BETTER else other.value.min()
            src = other.loc[other.value.idxmax() if b in LOWER_BETTER else other.value.idxmin(), "method"]
            hit = (dots.block == b) & (dots.unit == unit) & (dots.method == METHOD)
            if hit.any():
                dots.loc[hit, "value"] = worst; replaced += int(hit.sum()); act = "replaced placeholder"
            else:
                dots = pd.concat([dots, pd.DataFrame([dict(method=METHOD, block=b, unit=unit,
                                                           value=worst)])], ignore_index=True)
                added += 1; act = "inserted"
            print(f"{b:16s} {unit:32s} {worst:14.4f} {src:>28s}  {act}")

    # ---- selection time: add the lower bound of each unfinished subtype to its lineage
    tim = dots[dots.block == "ms_per_cell"]
    if not tim.empty:
        detail = pd.read_csv(a.timing_detail)
        for unit in MISSING:
            lin = unit.split("|")[0]
            row = detail[(detail.method == METHOD) & (detail.lineage == lin)]
            hit = (dots.block == "ms_per_cell") & (dots.unit == lin) & (dots.method == METHOD)
            if row.empty or not hit.any():
                print(f"ms_per_cell      {lin:32s} not in the timing block -- skipped")
                continue
            n = float(row.n_cells.iloc[0])
            before = float(dots.loc[hit, "value"].iloc[0])
            after = before + STALL_SECONDS * 1000.0 / n
            dots.loc[hit, "value"] = after
            print(f"{'ms_per_cell':16s} {lin:32s} {before:8.2f} -> {after:8.2f} ms/cell "
                  f"(+{STALL_SECONDS:.0f}s lower bound over {n:.0f} cells)")

    dots.to_csv(dpath, index=False)

    # ---- means: recompute the scgenefit row only
    master = pd.read_csv(mpath).set_index("method")
    for b in master.columns:
        v = dots[(dots.method == METHOD) & (dots.block == b)].value
        if len(v):
            master.loc[METHOD, b] = float(v.mean())
    master.reset_index().to_csv(mpath, index=False)

    print(f"\ninserted {added}, replaced {replaced}")
    n_units = dots[(dots.method == METHOD) & (dots.block == "recall20")].unit.nunique()
    print(f"scgenefit now covers {n_units} subtype units")
    print("scgenefit means:", {b: round(float(master.loc[METHOD, b]), 4) for b in master.columns})


if __name__ == "__main__":
    main()
