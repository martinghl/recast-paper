#!/usr/bin/env python
"""Fig. 2c,d tables: marker recovery of RECAST and its comparators on the 97 subtypes, and the tests between them.

Recovery of one subtype = the share of its marker panel (data/marker_panels.json, the panels of Fig. 3) among the 20
genes a method selects.
  RECAST            the trained encoder (data/top20_genes_by_subtype.csv.gz, encoder "trained"); it equals the
                    population x encoder cell of Fig. 2f (data/aggregation_and_encoder.csv), checked below
  no model          the difference of the mean expression profiles (Fig. 2f, population x no encoder)
  reference-free    attribution without a reference ("reference_free")
  random encoders   20 encoders with random weights ("random_0" ... "random_19")
The points of panels c and d are the 31 dataset-lineage groups (mean over each group's subtypes).
Tests, RECAST minus each comparator over the 31 groups: a two-sided sign-flip test (1,000,000 random sign vectors) and
a bootstrap 95% CI (10,000 resamples); against the random encoders the one-sided empirical P = (k + 1) / 21, k = the
seeds whose 31-group mean reaches RECAST's.  The same over the 97 subtypes is reported beside it.

Run     python 01_marker_recovery.py
Output  out/tables/marker_recovery_by_subtype.csv, marker_recovery_by_group.csv, marker_recovery_tests.csv
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "tables")
PANELS = json.load(open(os.path.join(DATA, "marker_panels.json")))
SEED, B, NSIGN_G, NSIGN_U = 20261106, 10_000, 1_000_000, 100_000


def recall(top20, unit):
    g = set(PANELS[unit])
    return len(set(top20.split(";")) & g) / len(g)


def signflip(d, n, seed=SEED):
    rng = np.random.default_rng(seed)
    t, hits, done = abs(d.sum()), 0, 0
    while done < n:
        k = min(100_000, n - done)
        S = rng.choice((1.0, -1.0), size=(k, len(d)))
        hits += int((np.abs(S @ d) >= t - 1e-12).sum()); done += k
    return hits / n


def boot(d, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.percentile(d[rng.integers(0, len(d), (B, len(d)))].mean(1), [2.5, 97.5])


def main():
    R = pd.read_csv(os.path.join(DATA, "top20_genes_by_subtype.csv.gz"))
    assert set(R.unit) == set(PANELS), f"{len(set(R.unit))} of {len(PANELS)} subtypes"
    R["recall"] = [recall(t, u) for t, u in zip(R.top20, R.unit)]
    W = R.pivot(index="unit", columns="encoder", values="recall")
    ft = pd.read_csv(os.path.join(DATA, "aggregation_and_encoder.csv")).set_index("unit").loc[W.index]
    assert np.allclose(W.trained, ft.population_encoder), "RECAST's recovery differs from Fig. 2f"
    units = pd.DataFrame({"group": W.index.str.split("|").str[0], "RECAST": W.trained,
                          "no_model": ft.population_no_encoder, "reference_free": W.reference_free,
                          "random_mean": W[[f"random_{k}" for k in range(20)]].mean(1)})
    for k in range(20):
        units[f"random_{k}"] = W[f"random_{k}"]
    G = units.groupby("group").mean()
    rows = []
    for comp in ("reference_free", "no_model", "random_mean"):
        dg = (G.RECAST - G[comp]).to_numpy(); du = (units.RECAST - units[comp]).to_numpy()
        lo, hi = boot(dg); ulo, uhi = boot(du)
        rows.append(dict(comparison=f"RECAST - {comp}", groups=len(dg), mean_RECAST=G.RECAST.mean(),
                         mean_comp=G[comp].mean(), diff=dg.mean(), ci_lo=lo, ci_hi=hi,
                         g_pos=int((dg > 0).sum()), g_zero=int((dg == 0).sum()), g_neg=int((dg < 0).sum()),
                         p_group=signflip(dg, NSIGN_G), units=len(du), unit_mean_RECAST=units.RECAST.mean(),
                         unit_mean_comp=units[comp].mean(), unit_diff=du.mean(), unit_ci_lo=ulo, unit_ci_hi=uhi,
                         u_pos=int((du > 0).sum()), u_zero=int((du == 0).sum()), u_neg=int((du < 0).sum()),
                         p_unit=signflip(du, NSIGN_U)))
    per_seed = np.array([G[f"random_{k}"].mean() for k in range(20)])
    k = int((per_seed >= G.RECAST.mean()).sum())
    per_seed_u = np.array([units[f"random_{k}"].mean() for k in range(20)])
    rows.append(dict(comparison="RECAST vs 20 random encoders", groups=31,
                     mean_RECAST=G.RECAST.mean(), mean_comp=per_seed.mean(), diff=G.RECAST.mean() - per_seed.mean(),
                     ci_lo=per_seed.min(), ci_hi=per_seed.max(), g_pos=20 - k, p_group=(k + 1) / 21,
                     unit_mean_comp=per_seed_u.mean(), unit_ci_lo=per_seed_u.min(), unit_ci_hi=per_seed_u.max()))
    st = pd.DataFrame(rows)
    st.loc[st.comparison.str.contains("random encoders"), "note"] = (
        f"random SD over seeds {per_seed.std(ddof=1):.4f}; ci_lo/ci_hi = min/max over seeds; "
        f"groups where RECAST exceeds all 20 seeds: "
        f"{int(sum((G.RECAST > G[[f'random_{j}' for j in range(20)]].max(1))))}/31")
    os.makedirs(OUT, exist_ok=True)
    units.to_csv(os.path.join(OUT, "marker_recovery_by_subtype.csv"))
    G.to_csv(os.path.join(OUT, "marker_recovery_by_group.csv"))
    st.to_csv(os.path.join(OUT, "marker_recovery_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(G[["RECAST", "no_model", "reference_free", "random_mean"]].round(3).to_string())
    print(st.drop(columns=[c for c in st.columns if c.startswith("unit_m")]).round(4).T.to_string())


if __name__ == "__main__":
    main()
