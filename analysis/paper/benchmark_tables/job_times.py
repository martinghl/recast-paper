"""Whole-job times of the marker-selection methods on the 20 subtype lineages, from Snakemake's benchmark files.

Every job of the marker_methods harness writes benchmarks/<id>-<method>_benchmark.txt (wall time s, CPU time,
maximum RSS).  Two id sets hold the 20 lineages: the timing run (cpu_server/marker_methods/timing_run.sh, one thread
per job; ids as in LIN20) and the benchmark grid (ids <lineage>_v2, the same cells and SCE dimensions).  The benchmark
files of both are copied from the CPU server to timing_benchmarks/ first.  These are whole-job times (R start-up,
object loading and writing included); selection_times.py compares them with the in-process selection times of the
same jobs.

n_cells per lineage: results/recast_timing_percompartment_e2e.csv (subtype_benchmark/time_recast_stages.py).
-> paper/benchmark_tables/data/selection_time_jobs.csv
   [method, lineage, id_set, s, cpu_time, max_rss_mb, ms_per_cell]   id_set = 'battery' (timing run) or 'grid'
Env analysis.  Usage: python -I job_times.py
"""
import glob
import os
import re

import pandas as pd

ROOT = "/path/to/research/analysis"
FSB = f"{ROOT}/subtype_benchmark/results"
BENCH = f"{ROOT}/subtype_benchmark/timing_benchmarks"
OUT = f"{ROOT}/paper/benchmark_tables/data"
LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]

ft = pd.read_csv(f"{FSB}/recast_timing_percompartment_e2e.csv")
ncell = dict(zip(ft.lineage, ft.n_cells))
rows = []
for f in glob.glob(f"{BENCH}/*_benchmark.txt"):
    base = os.path.basename(f)[:-len("_benchmark.txt")]
    m = re.match(r"^(" + "|".join(LIN20) + r")(_v2)?-(.+)$", base)
    if not m:
        continue
    lin, tag, method = m.group(1), m.group(2), m.group(3)
    try:
        b = pd.read_csv(f, sep="\t").iloc[0]
    except Exception:
        continue
    rows.append(dict(method=method, lineage=lin, id_set="battery" if tag is None else "grid",
                     s=float(b["s"]), cpu_time=float(b.get("cpu_time", float("nan"))),
                     max_rss_mb=float(b.get("max_rss", float("nan"))),
                     ms_per_cell=float(b["s"]) * 1000.0 / ncell[lin]))
de = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
de.to_csv(f"{OUT}/selection_time_jobs.csv", index=False)
for ids in ("battery", "grid"):
    cov = de[de.id_set == ids].groupby("method").lineage.nunique()
    print(f"{ids}: {cov.size} methods; lineages per method min={cov.min()} max={cov.max()}")
print("wrote", f"{OUT}/selection_time_jobs.csv")
