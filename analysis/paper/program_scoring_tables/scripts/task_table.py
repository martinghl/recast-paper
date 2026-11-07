#!/usr/bin/env python3
"""Per-task program-scoring table with pooled calibration, RECAST and ANS (input of task_table_train_only.py).

One row per task (28: 19 fine-subtype lineages + 9 distinct-type datasets), three metrics for RECAST and ANS:

    balanced accuracy  |  macro-F1  |  per-class (one-vs-rest) AUROC

Source (no re-scoring): the per-cell scoring results
    program_scoring/results_recast_pcell_landscape/metrics_all.csv   (21 datasets: 19 fine + 2 distinct)
    program_scoring/results_recast_pcell_distinct/metrics_all.csv    ( 7 distinct datasets)
RECAST = recast_weighted_expr_z (AUROC is rank-invariant, so every weighted_expr variant has the same AUROC);
ANS = ans_baselines.

Writes paper/program_scoring_tables/data/reuse_by_task_pooled_calibration.csv with, per task:
    dataset, regime, reg (fine|distinct), n_cells, n_states, recast_ba, ans_ba, recast_f1, ans_f1, recast_auc, ans_auc
(*_f1 is macro-F1, which weights every subtype equally, as balanced accuracy does)."""
import os
import pandas as pd

ROOT = "/path/to/research/analysis"
SRC = [
    f"{ROOT}/program_scoring/results_recast_pcell_landscape/metrics_all.csv",
    f"{ROOT}/program_scoring/results_recast_pcell_distinct/metrics_all.csv",
]
OUT = f"{ROOT}/paper/program_scoring_tables/data/reuse_by_task_pooled_calibration.csv"

RECAST = "recast_weighted_expr_z"   # RECAST per-cell scorer
COMP = "ANS"                      # signature-scoring comparator


def main():
    parts = []
    for p in SRC:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        parts.append(pd.read_csv(p))
    m = pd.concat(parts, ignore_index=True)

    def slice_method(meth, tag):
        s = m[m.method == meth][
            ["dataset", "regime", "balanced_acc", "macro_f1", "mean_auroc", "n_cells", "n_states"]
        ].copy()
        return s.rename(columns={"balanced_acc": f"{tag}_ba",
                                 "macro_f1": f"{tag}_f1",
                                 "mean_auroc": f"{tag}_auc"})

    f = slice_method(RECAST, "recast")
    a = slice_method(COMP, "ans")[["dataset", "ans_ba", "ans_f1", "ans_auc"]]
    df = f.merge(a, on="dataset", how="inner")
    if len(df) != 28:
        raise AssertionError(f"expected 28 datasets, got {len(df)}: {sorted(df.dataset)}")

    # two-tier grouping used by the panel's shared y-axis: broad collapses into distinct
    df["reg"] = df.regime.replace({"broad": "distinct"})
    df = df[["dataset", "regime", "reg", "n_cells", "n_states",
             "recast_ba", "ans_ba", "recast_f1", "ans_f1", "recast_auc", "ans_auc"]]
    df = df.sort_values(["reg", "recast_ba"], ascending=[True, False]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    # provenance summary for the log
    win = lambda fc, ac: int((df[fc].values >= df[ac].values).sum())
    print(f"wrote {OUT}  ({len(df)} datasets: "
          f"{(df.reg=='fine').sum()} fine + {(df.reg=='distinct').sum()} distinct/broad)")
    print("RECAST >= ANS  |  BA {}/28   macro-F1 {}/28   AUROC {}/28".format(
        win("recast_ba", "ans_ba"), win("recast_f1", "ans_f1"), win("recast_auc", "ans_auc")))
    for c in ("recast_ba", "ans_ba", "recast_f1", "ans_f1", "recast_auc", "ans_auc"):
        print(f"  {c:9s} [{df[c].min():.3f}, {df[c].max():.3f}]")


if __name__ == "__main__":
    main()
