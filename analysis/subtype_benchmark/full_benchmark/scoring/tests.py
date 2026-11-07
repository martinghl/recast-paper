#!/usr/bin/env python
"""RECAST against every other method of the subtype marker benchmark under four significance rules.

summary.py uses the first rule only.  The other three read the result with the subtype as the independent unit (a
subtype measured in several datasets counts once):
  unit bootstrap     paired bootstrap of (RECAST - method) over matched units, 10,000 resamples from default_rng(0)
                     shared by every method with the same n, 95% percentile interval (as summary.compare);
  subtype bootstrap  resample subtypes (state names) with replacement, 10,000 draws from default_rng(0); statistic =
                     mean difference over the units of the drawn subtypes;
  paired t           one-sample t-test of the per-unit differences;
  clustered t        intercept-only OLS on the differences, cluster-robust SE by subtype, t with (subtypes - 1) df.
Differences are signed so that > 0 favours RECAST (contamination is lower-better).  The methods are those of
marker_benchmark_by_subtype.csv (56, random included); scGeneFit carries the values fill_unfinished.py wrote; a method
missing a unit (scMAGS on fs37_glia_maitra) is compared on the units it has.  Rank = position of RECAST's mean among all
methods' means, as in summary.py.

Env analysis.  Usage: python -I tests.py [--score DIR] [--blocks ...]
-> <score>/significance_tests.csv (one row per endpoint x subset x method) and a printed digest.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
sys.path.insert(0, f"{FULL_BENCHMARK}/added_lineages")
import added_datasets as D  # noqa: E402

LOWER_BETTER = {"contamination"}
B, SEED = 10000, 0


def rules(d, cl):
    """d: differences (>0 favours RECAST) over matched units; cl: their subtype labels."""
    n = len(d)
    ub = d[np.random.default_rng(SEED).integers(0, n, size=(B, n))].mean(axis=1)
    codes, groups = pd.factorize(cl)
    s = np.bincount(codes, weights=d), np.bincount(codes)
    draw = np.random.default_rng(SEED).integers(0, len(groups), size=(B, len(groups)))
    cb = s[0][draw].sum(axis=1) / s[1][draw].sum(axis=1)
    if np.allclose(d, 0):
        p_t = p_tc = 1.0
    else:
        p_t = float(stats.ttest_1samp(d, 0).pvalue)
        fit = sm.OLS(d, np.ones(n)).fit(cov_type="cluster", cov_kwds={"groups": codes}, use_t=True)
        p_tc = float(fit.pvalues[0])
    return dict(n=n, n_subtypes=len(groups), delta=float(d.mean()),
                u_lo=np.percentile(ub, 2.5), u_hi=np.percentile(ub, 97.5),
                c_lo=np.percentile(cb, 2.5), c_hi=np.percentile(cb, 97.5), p_t=p_t, p_tc=p_tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=f"{FULL_BENCHMARK}/scoring/out")
    ap.add_argument("--blocks", nargs="*", default=["recall20", "marker_rank200", "auroc", "contamination"])
    a = ap.parse_args()
    dots = pd.read_csv(f"{a.score}/marker_benchmark_by_subtype.csv")
    rec = pd.read_csv(f"{a.score}/recall_by_subtype.csv").drop_duplicates("unit").set_index("unit")
    new = set(D.NEW_IDS)
    lin = lambda u: u.split("|")[0]
    subsets = {"all": lambda u: True, "original": lambda u: lin(u) not in new, "added": lambda u: lin(u) in new,
               "immune": lambda u: rec.tier.get(u) == "fine_immune",
               "epithelial": lambda u: rec.tier.get(u) == "fine_epithelial",
               "brain": lambda u: rec.tier.get(u) == "fine_brain",
               "no_brain": lambda u: rec.tier.get(u) in ("fine_immune", "fine_epithelial")}
    rows, digest = [], []
    for b in a.blocks:
        for sname, keep in subsets.items():
            d = dots[(dots.block == b) & dots.unit.map(keep)]
            if d.empty:
                continue
            piv = d.pivot_table(index="unit", columns="method", values="value").dropna(subset=["recast_M0"])
            means = piv.mean()
            sign = -1 if b in LOWER_BETTER else 1
            rank = int((sign * means > sign * means["recast_M0"]).sum() + 1)
            st = rec.loc[piv.index, "state"]
            res = []
            for m in piv.columns.drop("recast_M0"):
                sub = piv[["recast_M0", m]].dropna()
                if len(sub) < 3:
                    continue
                r = rules(sign * (sub.recast_M0 - sub[m]).to_numpy(float), st[sub.index].to_numpy())
                res.append(dict(block=b, subset=sname, method=m, mean=means[m], **r))
            t = pd.DataFrame(res)
            rows.append(t)
            w = lambda col: int((t[col] > 0).sum())
            digest.append(dict(
                block=b, subset=sname, units=len(piv), subtypes=st.nunique(), recast=round(means["recast_M0"], 4),
                rank=f"{rank}/{len(means)}",
                unit_boot=f"{w('u_lo')}-{int(((t.u_lo <= 0) & (t.u_hi >= 0)).sum())}-{int((t.u_hi < 0).sum())}",
                subtype_boot=f"{w('c_lo')}-{int(((t.c_lo <= 0) & (t.c_hi >= 0)).sum())}-{int((t.c_hi < 0).sum())}",
                t=f"{int(((t.p_t < .05) & (t.delta > 0)).sum())}-{int((t.p_t >= .05).sum())}-"
                  f"{int(((t.p_t < .05) & (t.delta < 0)).sum())}",
                t_clustered=f"{int(((t.p_tc < .05) & (t.delta > 0)).sum())}-{int((t.p_tc >= .05).sum())}-"
                            f"{int(((t.p_tc < .05) & (t.delta < 0)).sum())}",
                closest=", ".join(f"{r.method} {r.delta:+.3g} [{r.c_lo:+.3g},{r.c_hi:+.3g}] p_tc={r.p_tc:.2g}"
                                  for r in t.nsmallest(2, "delta").itertuples())))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f"{a.score}/significance_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 200):
        print(f"beats-ns-loses of the other methods; closest = 2 smallest RECAST - method "
              f"differences with the subtype-bootstrap CI and clustered-t P")
        print(pd.DataFrame(digest).to_string(index=False))


if __name__ == "__main__":
    main()
