"""Tests behind the significance marks of Fig. 6d and Supplementary Fig. 8.

For every comparator and dose k: RECAST minus the comparator in the per-task change from the curated panel (k minus
k = 0, averaged per task over 2 types x 5 draws: the values the figure plots).  At k = 20 this is the degradation
endpoint `deg_k20` of program_scoring/added_genes/analyze.py; the other doses use the same statistic.  Positive =
RECAST loses less.
Inference as in analyze.py (`boot`): cluster bootstrap over the 11 parent atlases of the 28 tasks, B = 10,000,
default_rng(0), draws shared across comparators, statistic = task-weighted mean over the drawn parents' tasks, 95%
percentile CI; two-sided P = 2 x the smaller bootstrap tail, floored at 2/B.  The k = 20 rows are checked against
results/paired_B.csv (endpoint deg_k20, donor folds, matched calibration).
Also: an exact sign-flip test on the parent sums of the per-task differences (all 2^11 sign patterns; smallest
two-sided P = 2/2048 = 0.00098), and RECAST's own change from the curated panel against 0 (same bootstrap).

  python added_genes_tests.py
writes paper/program_scoring_tables/data/scoring_added_genes_stats.csv"""
import glob, itertools, re
import numpy as np, pandas as pd

SB = "/path/to/research/analysis"
R = f"{SB}/program_scoring/added_genes/results"
OUT = f"{SB}/paper/program_scoring_tables/data/scoring_added_genes_stats.csv"
MET = ["balanced_acc", "macro_f1", "mean_auroc"]
DOSES = [5, 10, 20, 40]
B = 10_000

E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/eval_B/*.csv"))], ignore_index=True)
E = E[(E.scheme == "donor") & (E.calibration == "matched")]
E["dose"] = E.variant.map(lambda v: 0 if v == "orig" else int(re.match(r"\w+_k(\d+)_d\d$", v).group(1)))
parent = pd.read_csv(f"{SB}/paper/program_scoring_tables/data/reuse_by_task.csv").set_index("dataset").parent
methods = sorted(E.method.unique()); comps = [m for m in methods if m != "RECAST"]
assert len(methods) == 19 and E.task.nunique() == 28

tasks = sorted(E.task.unique())                                   # order of analyze.py: sorted tasks, sorted parents
par = sorted({parent[t] for t in tasks})
pidx = np.array([par.index(parent[t]) for t in tasks]); ntask = np.bincount(pidx, minlength=len(par))
draws = np.random.default_rng(0).integers(0, len(par), size=(B, len(par)))     # shared by every comparator
C = np.stack([(draws == j).sum(1) for j in range(len(par))], 1); denom = C @ ntask
SIGNS = np.array(list(itertools.product([1, -1], repeat=len(par))))           # 2^11 sign patterns


def infer(d):
    """d: per-task values (sorted task order). Bootstrap mean / CI / two-sided P, and the exact sign-flip P."""
    means = (C @ np.bincount(pidx, weights=d, minlength=len(par))) / denom
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_boot = max(2 * min((means <= 0).mean(), (means >= 0).mean()), 2 / B)
    S = np.bincount(pidx, weights=d, minlength=len(par))                      # parent sums
    null = SIGNS @ S
    p_flip = (np.abs(null) >= abs(S.sum()) - 1e-12).mean()
    return dict(mean=d.mean(), ci_lo=lo, ci_hi=hi, p_boot=p_boot, p_signflip=p_flip,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()))


rows = []
for met in MET:
    T = E.groupby(["task", "method", "dose"])[met].mean().unstack("dose")
    D = T[DOSES].sub(T[0], axis=0)                                            # per-task change from the curated panel
    for k in DOSES:
        W = D[k].unstack("method").loc[tasks]
        rows.append(dict(metric=met, dose=k, comparison="RECAST vs 0", comparator="RECAST", **infer(W["RECAST"].values)))
        for m in comps:
            rows.append(dict(metric=met, dose=k, comparison="RECAST - comparator", comparator=m,
                             **infer((W["RECAST"] - W[m]).values)))
S = pd.DataFrame(rows)
S["stars"] = pd.cut(S.p_boot, [0, 0.001, 0.01, 0.05, 1.0001], right=False, labels=["***", "**", "*", "n.s."]).astype(str)

# the k = 20 rows equal the deg_k20 analysis of analyze.py
P = pd.read_csv(f"{R}/paired_B.csv")
P = P[(P.scheme == "donor") & (P.calibration == "matched") & (P.endpoint == "deg_k20")].set_index(["metric", "comparator"])
mine = S[(S.dose == 20) & (S.comparison == "RECAST - comparator")].set_index(["metric", "comparator"])
for c in ["mean", "ci_lo", "ci_hi"]:
    ref = P["mean_diff" if c == "mean" else c].reindex(mine.index)
    assert np.allclose(mine[c], ref, atol=1e-12), (c, (mine[c] - ref).abs().max())
S.to_csv(OUT, index=False, float_format="%.6g")
print(OUT, len(S), "rows; k = 20 rows identical to paired_B.csv deg_k20")
