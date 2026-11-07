#!/usr/bin/env python3
"""Per-task program-scoring table with train-only calibration (Fig. 6a, Supplementary Data).

reuse_by_task_pooled_calibration.csv (task_table.py) holds RECAST's recast_weighted_expr_z scores, whose
z-calibration was estimated on the pooled out-of-fold scores (label-free but transductive).
program_scoring/train_only_calibration.py (-> results_m7/metrics_all_m7.csv) ran the same folds with every statistic --
weights, reference profiles and the calibration mean and sd -- estimated on the training fold (z_train).  This script
makes z_train the RECAST row of the 28-task table, keeps the ANS columns and task metadata, adds each task's parent
atlas, and gives the RECAST - ANS difference with a cluster bootstrap over parent atlases (several tasks are lineage
subsets of one atlas, so the 28 tasks are not independent).

Writes (paper/program_scoring_tables/data/):
  reuse_by_task.csv            dataset, regime, reg, parent, n_cells, n_states, recast_ba, ans_ba, recast_f1, ans_f1,
                               recast_auc, ans_auc  (recast_* = z_train)
  reuse_parent_means.csv       per parent atlas: n_tasks, mean RECAST - ANS difference per metric, tasks RECAST higher
  reuse_parent_bootstrap.csv   per metric: mean difference over the 28 tasks, cluster-bootstrap 95% CI over the parent
                               atlases (B = 10,000, default_rng(0)), parents with a positive mean difference
"""
import sys
import numpy as np, pandas as pd

ROOT = "/path/to/research/analysis"
D6 = f"{ROOT}/paper/program_scoring_tables/data"
sys.path.insert(0, f"{ROOT}/subtype_benchmark")
from lineage_registry import LINEAGES  # noqa: E402  parent dataset of every fine lineage task

B_BOOT, SEED = 10_000, 0
METRICS = [("ba", "balanced_acc"), ("f1", "macro_f1"), ("auc", "mean_auroc")]


def parent_of(task):
    return LINEAGES[task]["ds"] if task in LINEAGES else task


def master():
    old = pd.read_csv(f"{D6}/reuse_by_task_pooled_calibration.csv")      # task_table.py
    m7 = pd.read_csv(f"{ROOT}/program_scoring/results_m7/metrics_all_m7.csv")
    zt = m7[m7.calibration == "z_train"].set_index("dataset")
    zp = m7[m7.calibration == "z_pooled"].set_index("dataset")
    assert len(old) == 28 and set(old.dataset) == set(zt.index), "task sets differ"
    o = old.set_index("dataset")
    # the pooled-calibration table equals the z_pooled rows of the train-only run, so only the calibration changes
    for a, b in (("recast_ba", "balanced_acc"), ("recast_f1", "macro_f1"), ("recast_auc", "mean_auroc")):
        assert np.allclose(o[a], zp.loc[o.index, b]), a
    assert (o.n_cells == zt.loc[o.index, "n_cells"]).all() and (o.n_states == zt.loc[o.index, "n_states"]).all()
    df = old.copy()
    df["parent"] = df.dataset.map(parent_of)
    for a, b in (("recast_ba", "balanced_acc"), ("recast_f1", "macro_f1"), ("recast_auc", "mean_auroc")):
        df[a] = zt.loc[df.dataset, b].values
    df = df[["dataset", "regime", "reg", "parent", "n_cells", "n_states", "recast_ba", "ans_ba", "recast_f1", "ans_f1", "recast_auc", "ans_auc"]]
    df = df.sort_values(["reg", "recast_ba"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(f"{D6}/reuse_by_task.csv", index=False)
    print(f"wrote reuse_by_task.csv ({len(df)} tasks, {df.parent.nunique()} parent atlases)")
    for tag, _ in METRICS:
        d = df[f"recast_{tag}"] - df[f"ans_{tag}"]
        print(f"  {tag}: RECAST {df[f'recast_{tag}'].mean():.4f} ANS {df[f'ans_{tag}'].mean():.4f} diff {d.mean():+.4f} "
              f"RECAST higher {(d > 0).sum()}/28  | max |z_train - pooled| = {np.abs(df[f'recast_{tag}'].values - o.loc[df.dataset, f'recast_{tag}'].values).max():.4f}")
    return df


def parent_stats(df):
    rng = np.random.default_rng(SEED)
    parents = sorted(df.parent.unique())
    groups = {p: df[df.parent == p] for p in parents}
    rows = []
    for p in parents:
        g = groups[p]; r = dict(parent=p, n_tasks=len(g), tasks=";".join(g.dataset))
        for tag, _ in METRICS:
            d = g[f"recast_{tag}"] - g[f"ans_{tag}"]
            r[f"mean_diff_{tag}"] = d.mean(); r[f"recast_higher_{tag}"] = int((d > 0).sum())
        rows.append(r)
    pm = pd.DataFrame(rows); pm.to_csv(f"{D6}/reuse_parent_means.csv", index=False)
    boot = []
    idx = {p: np.where(df.parent.values == p)[0] for p in parents}
    draws = rng.integers(0, len(parents), size=(B_BOOT, len(parents)))
    for tag, _ in METRICS:
        d = (df[f"recast_{tag}"] - df[f"ans_{tag}"]).values
        means = np.array([np.concatenate([d[idx[parents[j]]] for j in row]).mean() for row in draws])
        lo, hi = np.percentile(means, [2.5, 97.5])
        boot.append(dict(metric=tag, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, n_tasks=len(d), n_parents=len(parents),
                         parents_positive=int((pm[f"mean_diff_{tag}"] > 0).sum()), tasks_positive=int((d > 0).sum()), B=B_BOOT, seed=SEED))
        print(f"  cluster bootstrap {tag}: mean diff {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  parents positive {boot[-1]['parents_positive']}/{len(parents)}")
    pb = pd.DataFrame(boot); pb.to_csv(f"{D6}/reuse_parent_bootstrap.csv", index=False)
    return pm, pb


if __name__ == "__main__":
    df = master(); parent_stats(df)
