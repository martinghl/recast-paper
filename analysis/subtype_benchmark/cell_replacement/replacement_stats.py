"""CPU stage of the cell-replacement experiment: aggregate over the 61 units, paired bootstrap, null P values,
sibling control, the example selection, and the equal-input-change comparison.
usage: python replacement_stats.py RESULTS_DIR
writes RESULTS_DIR/celldel_summary.csv, celldel_null_summary.csv, celldel_lineage_gap.csv, celldel_showcase.json"""
import sys, json
import numpy as np, pandas as pd

D = sys.argv[1]
B = 10000
METHODS = ["RECAST", "COSG", "Wilcoxon"]
main = pd.read_csv(f"{D}/celldel_main.csv")
null = pd.read_csv(f"{D}/celldel_null.csv")
sibc = pd.read_csv(f"{D}/celldel_sibling_ctrl.csv")
units = pd.read_csv(f"{D}/celldel_units.csv")
assert units.unit.nunique() == 61 and len(units) == 61, len(units)


def boot_ci(diff, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (B, len(diff)))
    m = diff[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


rows = []
for (mode, K), g in main.groupby(["mode", "K"]):
    for metric in ("identity_loss", "axis_shift", "orth_shift", "to_partner_state", "mean_abs_dx"):
        piv = g.pivot(index="unit", columns="method", values=metric)
        if piv.isna().all().all():
            continue
        assert len(piv) == 61, (mode, K, metric, len(piv))
        row = dict(mode=mode, K=K, metric=metric, n_units=len(piv))
        for m in METHODS:
            row[f"mean_{m}"] = piv[m].mean()
        for c in ("COSG", "Wilcoxon"):
            d = (piv["RECAST"] - piv[c]).to_numpy()
            lo, hi = boot_ci(d)
            row.update({f"diff_vs_{c}": d.mean(), f"ci_lo_vs_{c}": lo, f"ci_hi_vs_{c}": hi,
                        f"win_vs_{c}": int((d > 0).sum()), f"tie_vs_{c}": int((d == 0).sum()), f"loss_vs_{c}": int((d < 0).sum())})
        rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(f"{D}/celldel_summary.csv", index=False)

# null: per unit x method empirical P for identity loss (replace, K=20)
obs = main[(main["mode"] == "replace") & (main.K == 20)].set_index(["unit", "method"])
nrows = []
for (unit, m), g in null.groupby(["unit", "method"]):
    o = obs.loc[(unit, m)]
    for metric in ("identity_loss", "axis_shift"):
        v = g[metric].to_numpy()
        nrows.append(dict(unit=unit, method=m, metric=metric, observed=o[metric], null_mean=v.mean(),
                          null_q95=np.percentile(v, 95), n_null=len(v),
                          p_emp=(1 + (v >= o[metric]).sum()) / (len(v) + 1)))
nsum = pd.DataFrame(nrows)
nsum.to_csv(f"{D}/celldel_null_summary.csv", index=False)

# rule for choosing the example units
k20 = main[(main["mode"] == "replace") & (main.K == 20)].pivot(index="unit", columns="method", values="identity_loss")
gap = (k20["RECAST"] - k20[["COSG", "Wilcoxon"]].max(1)).rename("gap").reset_index()
gap = gap.merge(units[["unit", "lineage", "state", "n_anchored"]], on="unit")
lin = gap.groupby("lineage").agg(mean_gap=("gap", "mean"), min_anchored=("n_anchored", "min"), n_units=("unit", "size"))
lin["eligible"] = lin.min_anchored >= 100
lin = lin.sort_values("mean_gap", ascending=False)
lin.to_csv(f"{D}/celldel_lineage_gap.csv")
el = lin[lin.eligible]
best = el.index[0]
hl = gap[gap.lineage == best].sort_values("gap", ascending=False).iloc[0]
med = el.index[int(np.argsort(el.mean_gap.to_numpy())[len(el) // 2])]
show = dict(showcase_lineage=best, showcase_mean_gap=float(el.mean_gap.iloc[0]), highlight_unit=hl.unit,
            highlight_gap=float(hl.gap), median_lineage=med, median_mean_gap=float(el.loc[med, "mean_gap"]),
            n_eligible=int(len(el)), n_lineages=int(len(lin)))
json.dump(show, open(f"{D}/celldel_showcase.json", "w"), indent=1)

# console report
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
s = summary[summary.metric.isin(["identity_loss", "axis_shift"])]
print(s[["mode", "K", "metric"] + [f"mean_{m}" for m in METHODS] +
        ["diff_vs_COSG", "ci_lo_vs_COSG", "ci_hi_vs_COSG", "win_vs_COSG", "tie_vs_COSG", "loss_vs_COSG",
         "diff_vs_Wilcoxon", "ci_lo_vs_Wilcoxon", "ci_hi_vs_Wilcoxon", "win_vs_Wilcoxon", "tie_vs_Wilcoxon",
         "loss_vs_Wilcoxon"]].round(3).to_string(index=False))
il = nsum[nsum.metric == "identity_loss"]
print("\nnull (replace, K=20, identity loss): units with P<0.05 / mean observed / mean null")
for m in METHODS:
    x = il[il.method == m]
    print(f"  {m:9s} {int((x.p_emp < 0.05).sum())}/61  obs {x.observed.mean():.3f}  null {x.null_mean.mean():.3f}")
print("\nsibling control (zero T's K=20 genes in sibling cells):")
print(sibc.groupby("method")[["identity_loss", "to_target"]].mean().round(4).to_string())
print("\nlineage gap (RECAST - best comparator, K=20 replace identity loss):")
print(lin.round(3).to_string())
print("\nshowcase:", json.dumps(show))

# ---- magnitude-matched null (post hoc) ----
import os
if os.path.exists(f"{D}/celldel_dxnull.csv"):
    dx = pd.read_csv(f"{D}/celldel_dxnull.csv")
    agg = dx.groupby(["unit", "method"]).agg(null_loss=("identity_loss", "mean"), null_axis=("axis_shift", "mean"),
                                             null_dx=("mean_abs_dx", "mean")).reset_index()
    o = main[(main["mode"] == "replace") & (main.K == 20)][["unit", "method", "identity_loss", "axis_shift", "mean_abs_dx"]]
    agg = agg.merge(o, on=["unit", "method"])
    pe = dx.merge(o[["unit", "method", "identity_loss"]].rename(columns={"identity_loss": "obs"}), on=["unit", "method"])
    pv = pe.assign(ge=pe.identity_loss >= pe.obs).groupby(["unit", "method"]).ge.agg(["sum", "size"]).reset_index()
    pv["p_emp"] = (1 + pv["sum"]) / (1 + pv["size"])
    agg = agg.merge(pv[["unit", "method", "p_emp"]], on=["unit", "method"])
    agg["excess_loss"] = agg.identity_loss - agg.null_loss
    agg["excess_axis"] = agg.axis_shift - agg.null_axis
    agg["dx_ratio"] = agg.mean_abs_dx / agg.null_dx
    agg.to_csv(f"{D}/celldel_dxnull_summary.csv", index=False)
    print("\nmagnitude-matched null (post hoc) (replace, K=20): per method means over 61 units")
    print(agg.groupby("method")[["identity_loss", "null_loss", "excess_loss", "axis_shift", "null_axis", "excess_axis",
                                 "dx_ratio"]].mean().round(4).to_string())
    print("units with P<0.05:", agg[agg.p_emp < 0.05].groupby("method").size().to_dict())
    rows2 = []
    for metric in ("excess_loss", "excess_axis"):
        piv = agg.pivot(index="unit", columns="method", values=metric)
        for c in ("COSG", "Wilcoxon"):
            d = (piv["RECAST"] - piv[c]).to_numpy(); lo, hi = boot_ci(d)
            rows2.append(dict(metric=metric, vs=c, diff=d.mean(), ci_lo=lo, ci_hi=hi,
                              win=int((d > 0).sum()), tie=int((d == 0).sum()), loss=int((d < 0).sum())))
    r2 = pd.DataFrame(rows2); r2.to_csv(f"{D}/celldel_dxnull_paired.csv", index=False)
    print(r2.round(4).to_string(index=False))

# ---- equal-input-change comparison and robustness of the primary contrast (post hoc) ----
rp = main[main["mode"] == "replace"]
curve = {(u_, m): np.r_[[[0.0, 0.0]], g.sort_values("K")[["mean_abs_dx", "identity_loss"]].to_numpy()]
         for (u_, m), g in rp.groupby(["unit", "method"])}
for c_ in curve.values():
    assert (np.diff(c_[:, 0]) > 0).all(), "summed |dx| must increase with K"
k20 = rp[rp.K == 20].set_index(["unit", "method"])
eq_rows = []
for u_ in sorted(units.unit):
    r = k20.loc[(u_, "RECAST")]
    for c in ("COSG", "Wilcoxon"):
        o = k20.loc[(u_, c)]
        cr, cc = curve[(u_, "RECAST")], curve[(u_, c)]
        a2 = np.interp(o.mean_abs_dx, cr[:, 0], cr[:, 1])     # RECAST at the comparator's K=20 input change
        a1 = np.interp(r.mean_abs_dx, cc[:, 0], cc[:, 1])     # comparator at RECAST's K=20 input change
        eq_rows.append(dict(unit=u_, vs=c, dx_recast=r.mean_abs_dx, dx_comp=o.mean_abs_dx,
                            a2_diff=a2 - o.identity_loss, a1_diff=r.identity_loss - a1,
                            a2_clamped=bool(o.mean_abs_dx > cr[-1, 0]), a1_clamped=bool(r.mean_abs_dx > cc[-1, 0])))
eq = pd.DataFrame(eq_rows); eq.to_csv(f"{D}/celldel_equal_dx.csv", index=False)
print("\nequal input change (post hoc) (replace, K=20): A2 = RECAST interpolated to the comparator's |dx| (conservative),"
      " A1 = comparator interpolated to RECAST's |dx|")
for c in ("COSG", "Wilcoxon"):
    e = eq[eq.vs == c]
    for col in ("a2_diff", "a1_diff"):
        d = e[col].to_numpy(); m_, (lo, hi) = d.mean(), boot_ci(d)
        print(f"  vs {c:8s} {col}: {m_:+.4f} [{lo:+.4f}, {hi:+.4f}]  higher/tie/lower {(d > 0).sum()}/{(d == 0).sum()}/{(d < 0).sum()}"
              f"  clamped {int(e[col.replace('diff', 'clamped')].sum())}")

lin_of = units.set_index("unit").lineage
anc = units.set_index("unit").n_anchored
rob = []
k20p = rp[rp.K == 20].pivot(index="unit", columns="method", values="identity_loss")
for c in ("COSG", "Wilcoxon"):
    d = (k20p.RECAST - k20p[c])
    lins = lin_of.loc[d.index].to_numpy(); ul = np.unique(lins)
    rng = np.random.default_rng(0); cl = []
    for _ in range(B):                                       # lineage-cluster bootstrap
        pick_ = rng.choice(ul, len(ul))
        cl.append(np.concatenate([d.to_numpy()[lins == l] for l in pick_]).mean())
    big = d[anc.loc[d.index] >= 100].to_numpy()
    lo_b, hi_b = boot_ci(big)
    lm = d.groupby(lins).mean()
    rob.append(dict(vs=c, mean=d.mean(), median=d.median(), weighted_by_anchored=np.average(d, weights=anc.loc[d.index]),
                    cluster_ci_lo=np.percentile(cl, 2.5), cluster_ci_hi=np.percentile(cl, 97.5),
                    n_ge100=len(big), mean_ge100=big.mean(), ci_lo_ge100=lo_b, ci_hi_ge100=hi_b,
                    lineages_positive=int((lm > 0).sum()), n_lineages=len(lm),
                    loo_lineage_min=min(d[lins != l].mean() for l in ul)))
rob = pd.DataFrame(rob); rob.to_csv(f"{D}/celldel_robustness.csv", index=False)
print("\nrobustness of the primary contrast (replace, K=20):"); print(rob.round(4).to_string(index=False))
fl = k20p.max(1)
print(f"floor: subtypes where every method loses <5%: {(fl < 0.05).sum()}/61; <2%: {(fl < 0.02).sum()}/61")
print("method medians:", k20p.median().round(4).to_dict())
print("units with <100 anchored cells:", sorted(anc[anc < 100].index.tolist()))
