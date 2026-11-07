"""Fig. 6d: summaries and tests of the per-cell program scoring with added genes.

  python analyze.py B|A1|A2
B endpoints: k0 (curated panels), k5 / k10 / k20 / k40 (mean over 2 pool types x 5 draws), <type>_k<k>, and deg_k20
(k20 minus k0 per task; RECAST - comparator > 0 there means RECAST loses less).
Reads program_scoring/added_genes/results/eval_<exp>/*.csv (from the CPU server).  For each condition (fold scheme x
calibration) and each endpoint (B: groups of variants; A: the single long-signature variant), per-task metrics are
averaged over the variants of the endpoint; then RECAST minus each comparator is bootstrapped over the parent atlases
of the included tasks (10,000 draws, default_rng(0), shared across comparators and metrics; statistic = task-weighted
mean over the drawn parents' tasks; 95% percentile interval).
Intersection-union test per metric: "RECAST > all" when the lower bound is > 0 against every comparator.
Diagnostics (primary condition only): the lower bound with every parent atlas weighted equally, and the smallest lower
bound over leave-one-parent-out reruns (diag_* columns).
Writes results/summary_<exp>.csv (means and mean task rank), results/paired_<exp>.csv, results/iut_<exp>.csv."""
import glob, re, sys
import numpy as np, pandas as pd
SB = "/path/to/research/analysis"
R = f"{SB}/program_scoring/added_genes/results"
EXP = sys.argv[1]
MET = ["balanced_acc", "macro_f1", "mean_auroc"]
E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/eval_{EXP}/*.csv"))], ignore_index=True)
parent = pd.read_csv(f"{SB}/paper/program_scoring_tables/data/reuse_by_task.csv").set_index("dataset").parent
methods = sorted(E.method.unique()); comps = [m for m in methods if m != "RECAST"]
assert len(methods) == 19, methods

if EXP == "B":
    def grp(v):
        if v == "orig":
            return ["k0"]
        typ, k, _ = re.match(r"(\w+)_k(\d+)_d(\d)", v).groups()
        return [f"k{k}", f"{typ}_k{k}"]
    E["endpoint"] = E.variant.map(grp)
    E = E.explode("endpoint")
    # secondary endpoint: degradation = per-task mean at k = 20 minus the k = 0 value
    key = ["scheme", "calibration", "task", "method"]
    k20 = E[E.endpoint == "k20"].groupby(key)[MET].mean(); k0 = E[E.endpoint == "k0"].groupby(key)[MET].mean()
    E = pd.concat([E, (k20 - k0).reset_index().assign(endpoint="deg_k20", variant="deg_k20")], ignore_index=True)
    PRIMARY = "k20"
else:
    E["endpoint"] = "long"
    PRIMARY = "long"


def boot_parent_equal(W, tasks, seed=0, B=10_000):
    """Diagnostic: each drawn parent atlas weighted equally (mean of the parents' task means)."""
    order = list(tasks); par = sorted({parent[t] for t in order})
    pidx = np.array([par.index(parent[t]) for t in order]); ntask = np.bincount(pidx, minlength=len(par))
    draws = np.random.default_rng(seed).integers(0, len(par), size=(B, len(par)))
    C = np.stack([(draws == j).sum(1) for j in range(len(par))], 1)
    out = {}
    for m in comps:
        d = (W.loc[order, "RECAST"] - W.loc[order, m]).values
        pm = np.bincount(pidx, weights=d, minlength=len(par)) / ntask
        out[m] = np.percentile((C @ pm) / len(par), 2.5)
    return out


def lopo_min_lo(W, tasks):
    """Diagnostic: the smallest lower bound over leave-one-parent-out reruns, per comparator."""
    order = list(tasks); par = sorted({parent[t] for t in order}); res = {m: np.inf for m in comps}
    for p_ in par:
        keep = [t for t in order if parent[t] != p_]
        if len({parent[t] for t in keep}) < 2:
            continue
        for r in boot(W, keep):
            res[r["comparator"]] = min(res[r["comparator"]], r["ci_lo"])
    return res


def boot(W, tasks, seed=0, B=10_000):
    """W: task x method table of one metric. Returns rows RECAST - comparator."""
    order = list(tasks); par = sorted({parent[t] for t in order})
    pidx = np.array([par.index(parent[t]) for t in order]); ntask = np.bincount(pidx, minlength=len(par))
    draws = np.random.default_rng(seed).integers(0, len(par), size=(B, len(par)))
    C = np.stack([(draws == j).sum(1) for j in range(len(par))], 1); denom = C @ ntask
    out = []
    for m in comps:
        d = (W.loc[order, "RECAST"] - W.loc[order, m]).values
        means = (C @ np.bincount(pidx, weights=d, minlength=len(par))) / denom
        lo, hi = np.percentile(means, [2.5, 97.5])
        out.append(dict(comparator=m, mean_diff=d.mean(), higher=int((d > 0).sum()), lower=int((d < 0).sum()),
                        ci_lo=lo, ci_hi=hi, p_le0=float((means <= 0).mean()),
                        verdict="RECAST better" if lo > 0 else ("RECAST worse" if hi < 0 else "indistinguishable")))
    return out


summ, pair, iut = [], [], []
for (scheme, cal, ep), g in E.groupby(["scheme", "calibration", "endpoint"]):
    T = g.groupby(["task", "method"])[MET].mean()
    tasks = sorted(T.index.get_level_values(0).unique())
    for met in MET:
        W = T[met].unstack()
        rank = W.rank(axis=1, ascending=False, method="average").mean()
        for m in methods:
            summ.append(dict(scheme=scheme, calibration=cal, endpoint=ep, metric=met, method=m, mean=W[m].mean(),
                             mean_task_rank=rank[m], n_tasks=len(tasks)))
        rows = boot(W, tasks)
        primary = scheme == "donor" and cal == "matched"
        eq = boot_parent_equal(W, tasks) if primary else {}
        lo = lopo_min_lo(W, tasks) if primary else {}
        for r in rows:
            pair.append(dict(scheme=scheme, calibration=cal, endpoint=ep, metric=met, **r,
                             diag_lo_parent_equal=eq.get(r["comparator"], np.nan),
                             diag_min_lo_leave_one_parent_out=lo.get(r["comparator"], np.nan)))
        iut.append(dict(scheme=scheme, calibration=cal, endpoint=ep, metric=met, n_tasks=len(tasks),
                        recast_mean=W["RECAST"].mean(), recast_place_by_mean=int((W.drop(columns="RECAST").mean() > W["RECAST"].mean()).sum()) + 1,
                        n_better=sum(r["verdict"] == "RECAST better" for r in rows),
                        n_worse=sum(r["verdict"] == "RECAST worse" for r in rows),
                        not_beaten=",".join(r["comparator"] for r in rows if r["verdict"] != "RECAST better"),
                        IUT_RECAST_beats_all=all(r["verdict"] == "RECAST better" for r in rows),
                        diag_IUT_parent_equal=all(v > 0 for v in eq.values()) if eq else np.nan,   # NaN = not computed (non-primary)
                        diag_IUT_all_leave_one_parent_out=all(v > 0 for v in lo.values()) if lo else np.nan))
pd.DataFrame(summ).to_csv(f"{R}/summary_{EXP}.csv", index=False)
pd.DataFrame(pair).to_csv(f"{R}/paired_{EXP}.csv", index=False)
I = pd.DataFrame(iut); I.to_csv(f"{R}/iut_{EXP}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
prim = I[(I.scheme == "donor") & (I.calibration == "matched") & (I.endpoint == PRIMARY)]
print(f"=== {EXP} PRIMARY (donor folds, matched calibration, endpoint {PRIMARY}) ===")
print(prim.drop(columns=["scheme", "calibration", "endpoint"]).round(4).to_string(index=False))
print("\n=== all conditions ===")
print(I[["scheme", "calibration", "endpoint", "metric", "recast_mean", "recast_place_by_mean", "n_better", "n_worse",
         "IUT_RECAST_beats_all"]].round(4).to_string(index=False))
