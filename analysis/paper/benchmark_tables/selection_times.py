"""Selection-only times of the 51 marker-selection methods, measured at the same boundary as RECAST's selection call:
every runner of the marker_methods harness (code/run-<method>.R) wraps the method call alone in system.time() and
stores the elapsed seconds as `time` in the result RDS (no R start-up, object loading or writing).  Extracted on the
CPU server (selection_times.R) from

  battery   marker_methods/results/real_data/*.rds        the timing run (one thread per job) on the 20 lineages,
                                                        whose whole-job times are the id_set='battery' rows of
                                                        selection_time_jobs.csv (job_times.py; same jobs)
  grid      marker_methods_grid/results/real_data/*.rds   the benchmark runs on the 20 lineages and the whole-atlas runs
                                                        on six atlases (whole-job times:
                                                        marker_methods_whole_atlas/job_times.csv)

-> paper/benchmark_tables/data/selection_times_all_runs.csv  object x method x run, in-process s, whole-job s
-> paper/benchmark_tables/data/selection_times_summary.csv   per method x setting: median / min / max in-process s
Printed: in-process <= whole-job time for the same job; harness overhead = whole-job - in-process."""
import numpy as np, pandas as pd

ROOT = "/path/to/research/analysis"
OUT = f"{ROOT}/paper/benchmark_tables/data"; FSB = f"{ROOT}/subtype_benchmark/results"
grid = pd.read_csv(f"{FSB}/field_selection_times_inprocess.csv"); grid["provenance"] = "grid"
bat = pd.read_csv(f"{FSB}/field_selection_times_inprocess_battery.csv"); bat["provenance"] = "battery"
f = pd.concat([grid, bat], ignore_index=True)
f["setting"] = np.where(f.data_id.str.endswith("_allgenes"), "whole atlas", "per-lineage")
f["object"] = f.data_id.str.replace("_allgenes", "", regex=False)
f = f.rename(columns={"method_id": "method", "time_s": "inprocess_s"})
# whole-job times: battery (per lineage, same jobs as the battery RDS), whole-atlas runs (same jobs as the grid RDS)
full = pd.read_csv(f"{OUT}/selection_time_jobs.csv"); full = full[full.id_set == "battery"][["method", "lineage", "s"]].rename(columns={"lineage": "object", "s": "wall_s"})
ncl = pd.read_csv(f"{FSB}/recast_timing_percompartment_e2e.csv")[["lineage", "n_cells"]].rename(columns={"lineage": "object"})   # cell counts of the 20 lineage objects
full = full.merge(ncl, on="object", how="left")
t7 = pd.read_csv(f"{ROOT}/marker_methods_whole_atlas/job_times.csv").rename(columns={"dataset": "object", "compute_s": "wall_s"})
ft = pd.read_csv(f"{ROOT}/atlases/recast_timing.csv")[["dataset", "n_cells"]].rename(columns={"dataset": "object"})
t7 = t7.merge(ft, on="object", how="left")
lb = f[(f.setting == "per-lineage") & (f.provenance == "battery")].merge(full, on=["method", "object"], how="left")
lg = f[(f.setting == "per-lineage") & (f.provenance == "grid")].merge(full[["object", "n_cells"]].drop_duplicates(), on="object", how="left"); lg["wall_s"] = np.nan
ag = f[f.setting == "whole atlas"].merge(t7, on=["method", "object"], how="left")
L = pd.concat([lb, lg, ag], ignore_index=True)
L["inprocess_ms_per_cell"] = L.inprocess_s * 1e3 / L.n_cells
L["wall_minus_inprocess_s"] = L.wall_s - L.inprocess_s
cols = ["setting", "object", "method", "provenance", "n_cells", "inprocess_s", "inprocess_ms_per_cell", "wall_s", "wall_minus_inprocess_s", "n_result_rows"]
L = L[cols].sort_values(["setting", "provenance", "object", "inprocess_s"])
L.round(4).to_csv(f"{OUT}/selection_times_all_runs.csv", index=False)
ok = L.dropna(subset=["wall_s", "inprocess_s"])
for (st, pv), g in ok.groupby(["setting", "provenance"]):
    bad = int((g.inprocess_s > g.wall_s).sum())
    print(f"[{st} / {pv}] same-job pairs {len(g)}: in-process > wall in {bad} rows; harness overhead (wall - in-process) median {g.wall_minus_inprocess_s.median():.1f} s, IQR {g.wall_minus_inprocess_s.quantile(.25):.1f}-{g.wall_minus_inprocess_s.quantile(.75):.1f}")
# battery vs grid in-process agreement (same method, same lineage, two runs)
w = f[f.setting == "per-lineage"].pivot_table(index=["object", "method"], columns="provenance", values="inprocess_s").dropna()
r = (w["battery"] / w["grid"])
print(f"per-lineage battery/grid in-process ratio over {len(w)} pairs: median {r.median():.2f}, IQR {r.quantile(.25):.2f}-{r.quantile(.75):.2f}; Spearman {w.corr(method='spearman').iloc[0,1]:.3f}")
S = (L.dropna(subset=["inprocess_s"]).groupby(["setting", "provenance", "method"])
     .agg(n_objects=("object", "nunique"), median_s=("inprocess_s", "median"), min_s=("inprocess_s", "min"), max_s=("inprocess_s", "max"),
          median_ms_per_cell=("inprocess_ms_per_cell", "median")).reset_index().sort_values(["setting", "provenance", "median_s"]))
S.round(4).to_csv(f"{OUT}/selection_times_summary.csv", index=False)
for (st, pv), g in S.groupby(["setting", "provenance"]):
    print(f"\n=== {st} / {pv}: 12 fastest by median in-process selection time"); print(g.head(12)[["method", "n_objects", "median_s", "min_s", "max_s", "median_ms_per_cell"]].round(3).to_string(index=False))
print("wrote selection_times_all_runs.csv + selection_times_summary.csv")
