"""Per-task change of each scorer when genes are added to the curated panels (Fig. 6d, Supplementary Fig. 8).

Per task, method and dose k (genes added per curated panel): the metric averaged over the 2 contamination types x 5
draws, minus the same task's value on the curated panel (k = 0).  Donor-grouped folds, matched calibration.  k = 0 is
written once per task with value 0.  Input: program_scoring/added_genes/results/eval_B/*.csv (evaluate.py).

  python added_genes_table.py
writes paper/program_scoring_tables/data/scoring_added_genes.csv: metric, tool, dose, task, value (19 methods, 3 metrics)."""
import glob, re
import pandas as pd

SB = "/path/to/research/analysis"
R = f"{SB}/program_scoring/added_genes/results"
OUT = f"{SB}/paper/program_scoring_tables/data/scoring_added_genes.csv"
MET = ["balanced_acc", "macro_f1", "mean_auroc"]

E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/eval_B/*.csv"))], ignore_index=True)
E = E[(E.scheme == "donor") & (E.calibration == "matched")]
E["dose"] = E.variant.map(lambda v: 0 if v == "orig" else int(re.match(r"\w+_k(\d+)_d\d$", v).group(1)))
assert E.task.nunique() == 28 and E.method.nunique() == 19, (E.task.nunique(), E.method.nunique())
n = E.groupby(["task", "method", "dose"]).size()
assert (n.xs(0, level="dose") == 1).all() and (n.drop(0, level="dose") == 10).all(), "expected 1 curated + 2 types x 5 draws"

rows = []
for met in MET:
    T = E.groupby(["method", "task", "dose"])[met].mean().unstack("dose")
    D = T.sub(T[0], axis=0).stack().rename("value").reset_index()
    rows.append(D.assign(metric=met))
D = pd.concat(rows, ignore_index=True).rename(columns={"method": "tool"})[["metric", "tool", "dose", "task", "value"]]
D.to_csv(OUT, index=False, float_format="%.6f")
print(OUT, len(D), "rows")
for met in MET:
    s = D[(D.metric == met) & (D.dose == 20)].groupby("tool").value.median().sort_values()
    print(met, "median per-task change at k = 20:", s.round(3).to_dict())
