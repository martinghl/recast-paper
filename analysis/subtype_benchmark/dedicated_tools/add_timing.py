#!/usr/bin/env python
"""Selection-time rows of the four dedicated marker-selection tools, added to a copy of the per-lineage selection-time
table (paper/benchmark_tables/data/lineage_time_all_methods.csv: 51 methods x 20 lineages, the harness's in-process
time around the selection call on one thread, and RECAST's selection call).  The four CPU tools are timed the same
way: one thread, wall clock around the selection call only, on the CPU server.

scMarkerGene trains an ensemble of neural networks on a GPU, so its time is the training wall time on the local GPU
and carries a host label that says so (RECAST also runs on a GPU).  `--exclude-scmarkergene` drops it.

->  score_out/selection_time_per_cell.csv   method, lineage, ms_per_cell
->  score_out/timing_detail.csv              every dedicated-tool run with its seconds and cell count"""
import argparse, glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SB = "/path/to/research/analysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", default=f"{HERE}/timing")
    ap.add_argument("--base",
                    default=f"{SB}/paper/benchmark_tables/data/lineage_time_all_methods.csv")
    ap.add_argument("--out", default=f"{HERE}/score_out/selection_time_per_cell.csv")
    ap.add_argument("--exclude-scmarkergene", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def cells(lin):
        """Cell count for a lineage, from the same labels file every method was handed."""
        try:
            return sum(1 for l in open(f"{SB}/subtype_benchmark/sce_export_v2/"
                                       f"{lin}_labels.txt") if l.strip())
        except OSError:
            return None

    rows = []
    for f in sorted(glob.glob(f"{a.timing}/*.json")):
        d = json.load(open(f))
        # scMarkerGene's per-stage json does not carry a cell count; take it from the labels file
        # so its wall time can still be put on a per-cell scale in the separate report.
        n = d.get("n_cells") or cells(d["lineage"])
        rows.append(dict(method=d["method"], lineage=d["lineage"], seconds=d["seconds"],
                         n_cells=n, min_depth=d.get("min_depth"), max_depth=d.get("max_depth"),
                         ms_per_cell=(1000.0 * d["seconds"] / n) if n else None,
                         host="cpu_server (1 thread)" if d["method"] != "scmarkergene"
                              else "local GPU (training wall time)"))
    det = pd.DataFrame(rows)
    det.to_csv(f"{HERE}/score_out/timing_detail.csv", index=False)
    print(f"dedicated_tools timing runs: {len(det)}")
    print(det.groupby("method").agg(n=("lineage", "size"), ms_per_cell=("ms_per_cell", "mean"))
             .round(3).to_string())

    keep = det[det.ms_per_cell.notna()]
    if a.exclude_scmarkergene:
        keep = keep[keep.method != "scmarkergene"]
    base = pd.read_csv(a.base)[["method", "lineage", "ms_per_cell"]]
    if len(base) != 1020 or base.method.nunique() != 51:
        raise SystemExit(f"base timing table is not the matched-boundary 51 x 20 table: "
                         f"{len(base)} rows, {base.method.nunique()} methods ({a.base})")
    out = pd.concat([base, keep[["method", "lineage", "ms_per_cell"]]], ignore_index=True)
    out.to_csv(a.out, index=False)
    print(f"\n[wrote {a.out}] {base.method.nunique()} + {keep.method.nunique()} methods, "
          f"{len(out)} rows")


if __name__ == "__main__":
    main()
