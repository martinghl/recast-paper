#!/usr/bin/env python3
"""Donor-clustered intervals of the per-task differences of Fig. 6a.

program_scoring/donor_bootstrap.py computes a donor-clustered percentile bootstrap of the paired difference
(RECAST minus ANS, RECAST minus SIGnature) per task and metric.  The donors of each task come from the source objects
(program_scoring/donor_maps/*_donor.tsv, joined by cell name; make_donor_maps.py names every source).  22 of the 28
tasks reach the floor of 5 donors; the other 6 have 0-4 donors and carry no interval, which the table records.  The
Zhao object has no subject column (its `sample` is individual-by-tissue), so its three individuals are the clusters.

Checks before writing: every difference equals the point estimates of reuse_by_task.csv (ANS) and
reuse_by_task_reference_free.csv (SIGnature); every interval brackets its point and rests on >= 2,000 replicates; an
interval is present exactly when the task has enough donors.
Out: paper/program_scoring_tables/data/reuse_donor_intervals.csv"""
import os
import numpy as np
import pandas as pd

ROOT = "/path/to/research/analysis"
D6   = f"{ROOT}/paper/program_scoring_tables/data"
SRC  = f"{ROOT}/program_scoring/results_donorci/reuse_donor_intervals.csv"
MIN_DONORS = 5                      # the floor donor_bootstrap.py applies
METRICS = ("ba", "f1", "auc")


def main():
    ci  = pd.read_csv(SRC)
    tab = pd.read_csv(f"{D6}/reuse_by_task.csv").set_index("dataset")
    sg  = pd.read_csv(f"{D6}/reuse_by_task_reference_free.csv").set_index("dataset")

    assert len(ci) == len(tab) * 3 * 2, f"expected {len(tab) * 3 * 2} rows, got {len(ci)}"
    assert set(ci.dataset) == set(tab.index), "task set differs from reuse_by_task.csv"
    assert set(ci.metric) == set(METRICS) and set(ci.comparator) == {"ans", "sig"}

    # ---- every difference equals the point estimate of the task tables ------------------------------
    for r in ci.itertuples(index=False):
        recast = tab.loc[r.dataset, f"recast_{r.metric}"]
        other = (tab.loc[r.dataset, f"ans_{r.metric}"] if r.comparator == "ans"
                 else sg.loc[r.dataset, f"sig_gated_{r.metric}"])
        assert abs(r.delta - (recast - other)) < 1e-9, \
            f"{r.dataset} {r.comparator} {r.metric}: delta {r.delta} != {recast - other}"

    # ---- the interval brackets its own point, and rests on enough replicates ----------------------
    fin = ci.ci_lo.notna()
    assert (ci.loc[fin, "ci_lo"] <= ci.loc[fin, "delta"]).all(), "a lower bound sits above its point"
    assert (ci.loc[fin, "delta"] <= ci.loc[fin, "ci_hi"]).all(), "an upper bound sits below its point"
    assert (ci.loc[fin, "n_boot"] >= 2000).all(), "a CI rests on fewer than 2,000 replicates"
    assert int(ci.degenerate.sum()) == 0, "a bootstrap replicate was degenerate"

    # ---- an interval is present exactly when the task has enough donors ---------------------------
    assert (fin == (ci.n_donors >= MIN_DONORS)).all(), \
        "a row has an interval without enough donors, or lacks one despite having them"

    out = f"{D6}/reuse_donor_intervals.csv"
    ci.to_csv(out, index=False)
    n_task = ci.loc[fin, "dataset"].nunique()
    print(f"wrote reuse_donor_intervals.csv ({len(ci)} rows; {n_task} of {ci.dataset.nunique()} tasks carry an "
          f"interval, {ci.dataset.nunique() - n_task} below the {MIN_DONORS}-donor floor)")
    w = ci[fin].copy()
    w["verdict"] = np.where(w.ci_lo > 0, "RECAST higher",
                            np.where(w.ci_hi < 0, "comparator higher", "includes 0"))
    print(w.pivot_table(index=["comparator", "metric"], columns="verdict",
                        values="dataset", aggfunc="count").fillna(0).astype(int).to_string())
    return ci


if __name__ == "__main__":
    main()
