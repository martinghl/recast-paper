"""Concatenate the per-lineage RECAST tables of the label-noise runs into ps_recast_all{tag}.csv, the table
the figure reads.

label_noise_runs.py writes one ps_recast_<lineage>.csv per lineage.

    python merge_recast.py _v2     # ps_recast_all_v2.csv from the per-lineage files with tag _v2

Env SC.  Reads CSVs only."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIN = ["tf_cd8_dom", "tf_b_yosh", "tf_mono_steph", "fs_lung_hlca"]


def build(tag):
    return pd.concat([pd.read_csv(f"{HERE}/ps_recast_{l}{tag}.csv") for l in LIN], ignore_index=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    got = build(tag)
    p = f"{HERE}/ps_recast_all{tag}.csv"
    got.to_csv(p, index=False)
    print(f"wrote {p}  ({len(got)} rows, {got.lineage.nunique()} lineages, {got.state.nunique()} states)")


if __name__ == "__main__":
    main()
