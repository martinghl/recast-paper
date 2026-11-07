"""Benchmark table of the subtype marker benchmark (Fig. 3, Supplementary Fig. 3): one value per method, endpoint and
unit.

The four subtype-level endpoints share one unit set, the fine-tier units of the recall table (lineage | subtype with a
marker panel); selection time is a per-run quantity and stays per lineage.
  recall20        RESULTS_DIR/recall_by_subtype.csv                       (marker_recall_all.py)
  auroc           AUROC_DIR/fs_cm_arm3_perstate_<lineage><tag>.csv, k = 20 (subtype_auroc.py / auroc.py)
  marker_rank200  TABLES_DIR/marker_rank_by_subtype.csv                   (marker_rank.py)
  contamination   TABLES_DIR/contamination_by_subtype.csv                 (contamination.py)
  ms_per_cell     SELECTION_TIME_FILE (dedicated_tools/add_timing.py), per lineage
seurat_roc is left out (it returns 6-24 genes per subtype).
-> TABLES_DIR/marker_benchmark_by_subtype.csv   [method, block, unit, value]  (unit = lineage|subtype, or lineage for time)
-> TABLES_DIR/marker_benchmark_means.csv        [method x block means]
Env analysis.  Run by run_scoring_all.sh.
"""
import glob
import os

import pandas as pd

SB = "/path/to/research/analysis"
FULL_BENCHMARK = f"{SB}/subtype_benchmark/full_benchmark"
FSB = os.environ.get("RESULTS_DIR", f"{FULL_BENCHMARK}/scoring/out")
PSAU = os.environ.get("AUROC_DIR", f"{FULL_BENCHMARK}/scoring/in/auroc")
OUT = os.environ.get("TABLES_DIR", f"{FULL_BENCHMARK}/scoring/out")
V = os.environ.get("RECAST_RESULT_TAG", "_v3x")      # tag of the per-subtype AUROC files
TIMEF = os.environ.get("SELECTION_TIME_FILE", f"{SB}/subtype_benchmark/dedicated_tools/score_out/selection_time_per_cell.csv")

# ---- unit set = recall fine-tier units (unit = lineage|subtype) ------------------------------
u = pd.read_csv(f"{FSB}/recall_by_subtype.csv")
u = u[u["tier"].astype(str).str.startswith("fine")].copy()
u["unit"] = u.lineage + "|" + u.state
LIN = sorted(u.lineage.unique())
CANON = sorted(u.unit.unique())
print(f"fine units: {len(CANON)} over {len(LIN)} lineages")

# ---- recall (per unit) -------------------------------------------------------------------------
rec = u.groupby(["method", "unit"]).recall.mean().reset_index().rename(columns={"recall": "value"})
rec["block"] = "recall20"

# ---- auroc (per unit, k = 20) ------------------------------------------------------------------
_auf = sorted(glob.glob(f"{PSAU}/fs_cm_arm3_perstate_*{V}.csv"))
assert all(any(f.endswith(f"_{l}{V}.csv") for f in _auf) for l in LIN), "a lineage has no AUROC file"
au = pd.concat([pd.read_csv(f) for f in _auf], ignore_index=True)
au = au[au.k == 20].copy()
au["unit"] = au.lineage + "|" + au.state
au = au[["method", "unit", "auroc"]].rename(columns={"auroc": "value"})
au["block"] = "auroc"

# ---- marker rank @ top-200 (per unit): mean over panel genes of 100*(1-(rank-1)/200), absent -> 0 ----
mr = pd.read_csv(f"{OUT}/marker_rank_by_subtype.csv")
mr = mr[["method", "unit", "mrank200_pct"]].rename(columns={"mrank200_pct": "value"})
mr["block"] = "marker_rank200"

# ---- contamination @ top-200 (per unit): ribosomal + mitochondrial genes; lower = cleaner ---------
ct = pd.read_csv(f"{OUT}/contamination_by_subtype.csv")
ct = ct[["method", "unit", "contam_n"]].rename(columns={"contam_n": "value"})
ct["block"] = "contamination"

# ---- keep the subtype-level endpoints on the unit set ---------------------------------------------
state_parts = []
for name, df in [("recall20", rec), ("auroc", au), ("marker_rank200", mr), ("contamination", ct)]:
    d = df[df.unit.isin(CANON)].copy()
    print(f"  {name:14s}: {d.unit.nunique():3d} states, {d.method.nunique()} methods, {len(d)} rows")
    state_parts.append(d[["method", "block", "unit", "value"]])

# ---- selection time (per lineage) -------------------------------------------------------------
tm = pd.read_csv(TIMEF)
tm = tm[tm.lineage.isin(LIN)].rename(columns={"ms_per_cell": "value", "lineage": "unit"})
tm["block"] = "ms_per_cell"
print(f"  ms_per_cell : {tm.unit.nunique()} lineages (per run)")

dots = pd.concat(state_parts + [tm[["method", "block", "unit", "value"]]], ignore_index=True)
dots = dots[dots.method != "seurat_roc"].copy()
dots.to_csv(f"{OUT}/marker_benchmark_by_subtype.csv", index=False)

# ---- means ------------------------------------------------------------------------------------
master = dots.groupby(["method", "block"]).value.mean().unstack("block")
master = master.reindex(master.sort_values("recall20", ascending=False).index)
master.to_csv(f"{OUT}/marker_benchmark_means.csv")
print("\nwrote marker_benchmark_by_subtype rows", len(dots), "| blocks", sorted(dots.block.unique()))
print("RECAST means:", master.loc["recast_M0"].round(3).to_dict())
