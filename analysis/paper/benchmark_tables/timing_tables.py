"""Selection-time tables per lineage and per whole atlas (Supplementary Figs. 3 and 10), with every method timed at
the selection call: the harness's in-process system.time() around each method call (selection_times_all_runs.csv:
battery run for the lineages, grid for the atlases; selection_times.py), and RECAST's selection call through the
package (subtype_benchmark/time_recast_package.py: recast_timing_lineages.csv, recast_timing_whole_atlas.csv, median of 3 runs, GPU).  RECAST's own
rows in the harness table are left out.
-> lineage_time_all_methods.csv      [method, lineage, id_set (battery | gpu_local), s, cpu_time, max_rss_mb,
                                     ms_per_cell]; read by subtype_benchmark/dedicated_tools/add_timing.py
-> whole_atlas_time_all_methods.csv  dataset x method selection time in s (method 'recast' = RECAST)"""
import numpy as np, pandas as pd
ROOT = "/path/to/research/analysis"; OUT = f"{ROOT}/paper/benchmark_tables/data"; FSB = f"{ROOT}/subtype_benchmark/results"
IP = pd.read_csv(f"{OUT}/selection_times_all_runs.csv")      # selection_times.py
own = IP.method.str.contains("recast", regex=False)
# Seurat ROC is not one of the 51 methods of the manuscript, so it is left out here too
own = own | (IP.method == "seurat_roc")
L = pd.read_csv(f"{FSB}/recast_timing_lineages.csv"); A = pd.read_csv(f"{FSB}/recast_timing_whole_atlas.csv")
# per lineage
b = IP[(IP.setting == "per-lineage") & (IP.provenance == "battery") & ~own]
pl = pd.DataFrame(dict(method=b.method, lineage=b.object, id_set="battery", s=b.inprocess_s, cpu_time=np.nan, max_rss_mb=np.nan, ms_per_cell=b.inprocess_ms_per_cell))
rc = pd.DataFrame(dict(method="recast_M0", lineage=L.lineage, id_set="gpu_local", s=L.select_s, cpu_time=np.nan, max_rss_mb=np.nan, ms_per_cell=L.select_ms_per_cell))
P = pd.concat([pl, rc], ignore_index=True); P.round(6).to_csv(f"{OUT}/lineage_time_all_methods.csv", index=False)
print(f"per-lineage: {P[P.id_set == 'battery'].method.nunique()} field methods x {P.lineage.nunique()} lineages + RECAST ({len(rc)} rows)")
# whole atlas
a = IP[(IP.setting == "whole atlas") & ~own]
W = pd.DataFrame(dict(dataset=a.object, method=a.method, sel_s=a.inprocess_s, n_cells=a.n_cells))
W = pd.concat([W, pd.DataFrame(dict(dataset=A.dataset, method="recast", sel_s=A.select_s, n_cells=A.n_cells))], ignore_index=True)
W.round(6).to_csv(f"{OUT}/whole_atlas_time_all_methods.csv", index=False)
med = W[~W.method.isin(["recast", "random"])].groupby("method").sel_s.median().sort_values()
print(f"whole atlas: {med.size} field methods; fastest {med.index[0]} {med.iloc[0]:.2f} s, slowest {med.index[-1]} {med.iloc[-1]:.0f} s; RECAST {W[W.method == 'recast'].sel_s.median():.2f} s")
d = W[W.dataset == "dominguez_conde"].set_index("method").sel_s
print("dominguez in-process: " + ", ".join(f"{m} {d[m]:.1f}" for m in ["recast", "scanpy_tover_rankby_raw", "cosg", "cepo", "scanpy_t_rankby_raw", "limma_trend", "edger_ql"] if m in d))
