"""Per-atlas table of the comparator methods' marker-independent metrics on the broad populations: the
table of cpu_server/marker_methods_grid/broad_auroc_all_genes.R (figures/arm3_full_csv/arm3_<ds>_allgenes_fast.csv),
expression rows, one row per (method, k) with the metrics as columns.

    python merge_comparator_auroc.py [--datasets pbmc3k,lawlor,...]
-> broad_populations/results/baselines_<ds>.csv, read by paper/benchmark_tables/scripts/broad_tables.py."""
import argparse

import pandas as pd

CSV = "/path/to/research/analysis/figures/arm3_full_csv"
OUT = "/path/to/research/analysis/broad_populations/results"
ATLASES = "pbmc3k,lawlor,ss3_pbmc,citeseq,ren,stephenson,dominguez_conde,yoshida"


def merge(ds):
    real = pd.read_csv(f"{CSV}/arm3_{ds}_allgenes_fast.csv")
    real = real[real.currency == "expression"]
    rp = real.pivot_table(index=["method", "k"], columns="metric", values="value").reset_index()
    rp["dataset"] = ds
    rp.to_csv(f"{OUT}/baselines_{ds}.csv", index=False)
    print(f"[{ds}] {rp.method.nunique()} methods", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=ATLASES)
    for ds in ap.parse_args().datasets.split(","):
        merge(ds)
