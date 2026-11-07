"""Supplementary Fig. 5: sensitivity to the settings of RECAST and to the cell annotation.

  a-c  marker recovery against the number of integration steps (a), target cells (b) and reference cells (c), for the
       16 Zhao subtypes: one faint line per subtype, the mean (bold) and +-1 s.d. (band); "All" = all cells
  d    standard deviation of recovery over repeated sampling of target and reference cells (16 subtypes)
  e    recovery with five choices of the integration baseline (box plots, one point per subtype)
  f,g  recovery (f) and top-20 overlap with the unperturbed list (g) when a fraction of the cells is relabelled, at
       subtype boundaries or at random, RECAST against Wilcoxon (mean over subtypes of the mean over three seeds)
  h    loss of recovery against the effective shift of the target profile, 10% boundary relabelling, per subtype

Run     python 01_draw.py
Output  out/S5_sensitivity_and_annotation.{pdf,svg,png}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S                    # noqa: E402
import supplementary_common as K                   # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402
from matplotlib.patches import Patch      # noqa: E402
import matplotlib.text as mtext           # noqa: E402

# ------------------------------------------------------------------ a-e: settings of RECAST (16 Zhao subtypes)
integration_steps = pd.read_csv(f"{HERE}/data/sensitivity_integration_steps.csv")
cell_number = pd.read_csv(f"{HERE}/data/sensitivity_cell_number.csv")
reference_size = pd.read_csv(f"{HERE}/data/sensitivity_reference_size.csv")
seeds = pd.read_csv(f"{HERE}/data/sensitivity_seeds.csv")
baseline_choice = pd.read_csv(f"{HERE}/data/baseline_choice.csv")
# one row per subtype per setting (repeated draws already averaged within a subtype)
for d_, x_ in ((integration_steps, "n_steps"), (cell_number, "n_cells"), (reference_size, "ref_size")):
    assert d_.target.nunique() == 16 and d_.lineage.str.startswith("zhao_").all()
    assert not d_.duplicated(["lineage", "target"] + ([x_] if "is_full" not in d_ else ["is_full", x_])).any()
for d_ in (cell_number, reference_size):
    assert (d_[d_.is_full].jac_vs_full == 1.0).all() and d_.groupby("target").size().eq(5).all()
assert seeds.target.nunique() == 16 and len(seeds) == 16 and (seeds.n_cells == 100).all() and (seeds.ref_size == 200).all()
MEDIAN_SEED_SD = float(np.median(seeds.recall20_sd))
assert round(MEDIAN_SEED_SD, 5) == 0.04741
assert baseline_choice.groupby("mode").size().eq(16).all() and set(baseline_choice["mode"]) == {
    "mean_lognorm", "ref_cpm", "median_lognorm", "global_cpm", "zero"}
BASELINE_MODES = [("mean_lognorm", "Mean log-normalized profiles (default)"), ("ref_cpm", "Pooled-count profiles"),
                  ("median_lognorm", "Median profiles"), ("global_cpm", "Whole-dataset pooled baseline"),
                  ("zero", "Zero baseline")]

# ------------------------------------------------------------------ f-h: relabelled cells (16 subtypes, 14 with a panel)
perturbed_recast = pd.read_csv(f"{HERE}/data/annotation_perturbation_recast.csv")
perturbed_wilcoxon = pd.read_csv(f"{HERE}/data/annotation_perturbation_comparators.csv")
effective_shift = pd.read_csv(f"{HERE}/data/annotation_perturbation_effective_shift.csv")
assert len(perturbed_recast) == 496 and len(perturbed_wilcoxon) == 496 and len(effective_shift) == 32
for d_ in (perturbed_recast, perturbed_wilcoxon):
    assert d_.groupby(["lineage", "state"]).ngroups == 16
    assert d_[d_["mode"] == "none"].groupby(["lineage", "state"]).size().eq(1).all()
    assert d_[d_["mode"] != "none"].groupby(["mode", "eps", "lineage", "state"]).size().eq(3).all()   # 3 seeds
    assert d_.dropna(subset=["recall20"]).groupby(["lineage", "state"]).ngroups == 14
    assert d_.jaccard20.notna().all()                                  # g keeps all 16 subtypes
clean_recast = perturbed_recast[perturbed_recast["mode"] == "none"].recall20.mean()
clean_wilcoxon = perturbed_wilcoxon[perturbed_wilcoxon["mode"] == "none"].recall20.mean()
effective_shift_10 = effective_shift[np.isclose(effective_shift.eps, 0.10)].dropna(subset=["drop"]).copy()
assert len(effective_shift_10) == 14

S.setup()

STEM = "S5_sensitivity_and_annotation"
W = 180.0


def tukey_stats(v):
    """Tukey box-plot summary: quartiles and whiskers at the most extreme points within 1.5 x IQR."""
    v = np.sort(np.asarray(v, float))
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    return q1, med, q3, float(inside.min()), float(inside.max())


def sweep(ax, df, xcol, ycol, show_key=False):
    """One sensitivity panel (a-c): the 16 subtypes as faint lines, a +-1 s.d. band, the mean as a bold line with
    white-rimmed points; in b and c a dotted connector and a separator before the "All" diamond (all cells)."""
    has_full = "is_full" in df.columns
    nom = df[~df["is_full"]] if has_full else df
    g = nom.groupby(xcol)[ycol].agg(["mean", "std", "size"]).reset_index().sort_values(xcol)
    assert (g["size"] == 16).all()

    # 1. Faint per-subtype trajectories (16 Zhao subtypes)
    piv = nom.pivot(index="target", columns=xcol, values=ycol)
    xs_nom = [int(v) for v in g[xcol]]
    for _, row in piv.iterrows():
        ax.plot(xs_nom, row[xs_nom].to_numpy(float), color=S.RECAST_L, alpha=0.42, lw=0.42, zorder=1.5)

    # 2. +-1 SD shaded band + mean curve + white-rimmed circles
    lo, hi = np.clip(g["mean"] - g["std"], 0, 1), np.clip(g["mean"] + g["std"], 0, 1)
    ax.fill_between(g[xcol], lo, hi, color=S.RECAST_XL, alpha=0.85, lw=0, zorder=1)
    ax.plot(g[xcol], g["mean"], color=S.RECAST, lw=1.25, zorder=3)
    ax.scatter(g[xcol], g["mean"], s=15, color=S.RECAST, edgecolors="white", linewidths=0.45, zorder=4)

    ticks = list(xs_nom)
    labs = [f"{t:,}" for t in ticks]
    xhi = ticks[-1]
    fx = None
    if has_full:
        fx = float(df[df["is_full"]][xcol].median())
        # the "All" tick sits one log-step past the last fixed setting
        x_all = ticks[-1] * 2.2
        x_sep = np.sqrt(ticks[-1] * x_all)
        ax.axvline(x_sep, color=S.HAIR, lw=S.HAIR_LW, ls=(0, (3, 3)), zorder=0.8)
        ax.plot([ticks[-1], x_all], [g["mean"].iloc[-1], 1.0], color=S.RECAST, lw=1.0, ls=(0, (2, 2)), zorder=2.5)
        ax.scatter([x_all], [1.0], s=26, marker="D", color=S.RECAST_D, edgecolors="white", linewidths=0.5, zorder=5)
        ticks, labs, xhi = ticks + [x_all], labs + ["All"], x_all
    else:
        # In panel a, highlight the 50-step default reference with a dark-blue diamond
        ax.scatter([50], [1.0], s=24, marker="D", color=S.RECAST_D, edgecolors="white", linewidths=0.5, zorder=5)

    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    ax.set_xlim(ticks[0] * 0.82, xhi * 1.24)
    ax.set_ylim(0.0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    S.frame(ax)

    if show_key:
        # Compact key in the lower blank region of panel a (where all curves sit above y = 0.94)
        h_mu = Line2D([0], [0], color=S.RECAST, lw=1.25, marker="o", ms=3.6, mec="white", mew=0.4,
                      label="Mean \u00b1 1 s.d.")
        h_sub = Line2D([0], [0], color=S.RECAST_L, lw=0.65, label="16 subtypes")
        h_ref = Line2D([0], [0], color="none", marker="D", ms=4.0, mfc=S.RECAST_D, mec="white", mew=0.4,
                       label="Reference (1.0)")
        ax.legend(handles=[h_mu, h_sub, h_ref], loc="lower left", bbox_to_anchor=(0.03, 0.03),
                  frameon=False, handlelength=1.4, handletextpad=0.45, labelspacing=0.28, borderaxespad=0.0)

    return g.assign(lo=lo, hi=hi), fx


def pooled_with_sem(df, metric, n_expect):
    """Seed mean within subtype, then mean and SEM across subtypes (for panels f and g)."""
    out = {}
    for mode in ("boundary", "random"):
        per = df[df["mode"] == mode].groupby(["eps", "lineage", "state"])[metric].mean().dropna()
        assert per.groupby("eps").size().eq(n_expect).all()
        g = per.groupby("eps").agg(["mean", lambda s: s.std(ddof=1) / np.sqrt(len(s))])
        g.columns = ["mean", "sem"]
        out[mode] = (g.index.to_numpy() * 100, g["mean"].to_numpy(), g["sem"].to_numpy())
    return out


def curves(ax, dat_f, dat_d):
    """Row 3 perturbation curves (f, g): mean across subtypes (seed mean within subtype first)."""
    for mode, ls in (("boundary", "-"), ("random", ":")):
        for dat, col, lw, ssz, z in ((dat_f, S.RECAST, 1.2, 14, 5), (dat_d, S.DE, 0.95, 11, 4)):
            x, y, _ = dat[mode]
            ax.plot(x, y, ls=ls, color=col, lw=lw, zorder=z)
            ax.scatter(x, y, s=ssz, color=col, edgecolors="white", linewidths=0.4, zorder=z + 2)


def noise_axis(ax):
    S.frame(ax)
    ax.set_xscale("log")
    ax.set_xticks([2.5, 5, 10, 20, 40])
    ax.set_xticklabels(["2.5", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlim(2.1, 48)
    ax.set_xlabel("Cells relabelled (% of lineage)")


def title(P, ax, s):
    bb = ax.get_position()
    return P.text((bb.x0 + bb.x1) / 2 * W, P.H - bb.y1 * P.H - 1.3, s, ha="center", va="bottom")


def main():
    R1, H1 = 6.5, 31.0                    # Row 1 (a-c): y = 6.5..37.5 mm
    R2, H2 = 49.0, 25.0                   # Row 2 (d-e): y = 49.0..74.0 mm
    R3, H3 = 92.5, 30.5                   # Row 3 (f-h): y = 92.5..123.0 mm
    H = R3 + H3 + 8.5                     # Total height = 131.5 mm (ultra-compact vertical spacing)
    P = S.Page(W, H)
    fig = P.fig
    checks = {}

    # ============================================================== Row 1: a-c (shared y axis, minimal 2.0 mm gap)
    W1 = 51.5
    G1 = 2.0
    X_A = 19.5
    X_B = X_A + W1 + G1                   # 73.0
    X_C = X_B + W1 + G1                   # 126.5 (ends at 178.0)
    ax_a = P.ax(X_A, R1, W1, H1)
    ax_b = P.ax(X_B, R1, W1, H1)
    ax_c = P.ax(X_C, R1, W1, H1)
    ga, _ = sweep(ax_a, integration_steps, "n_steps", "jac_vs50", show_key=True)
    gb, fxb = sweep(ax_b, cell_number, "n_cells", "jac_vs_full")
    gc, fxc = sweep(ax_c, reference_size, "ref_size", "jac_vs_full")
    ax_a.set_ylabel("Jaccard overlap\nwith the 50-step,\nfull-population selection", linespacing=1.10)
    for ax in (ax_b, ax_c):
        ax.set_yticklabels([])
    ax_a.set_xlabel("Integration steps")
    ax_b.set_xlabel("Target cells sampled")
    ax_c.set_xlabel("Reference cells sampled")
    for ax, s in ((ax_a, "Numerical integration"),
                  (ax_b, "Target-cell subsampling"),
                  (ax_c, "Reference-cell subsampling")):
        title(P, ax, s)
    P.letter(0.5, 0.8, "a")
    P.letter(X_B, 0.8, "b")
    P.letter(X_C, 0.8, "c")
    assert abs(ga.set_index("n_steps").loc[50, "mean"] - 1.0) < 1e-12

    # ============================================================== Row 2: d (joint sampling) & e (profiles/baselines)
    # d: horizontal Tukey box plot, the 16 subtypes as jittered points, mean diamond, dashed median line
    sd = seeds["recall20_sd"].to_numpy(float)
    D_X, D_W = 4.0, 69.5
    ax_d = P.ax(D_X, R2, D_W, H2)
    q1, med, q3, wlo, whi = tukey_stats(sd)
    mu_sd = float(np.mean(sd))

    # Draw horizontal Tukey box at y = 0 (spanning y = -0.38..0.38 so all 5 x=0 points sit inside the Q1 edge)
    BW_D, CAP_D = 0.38, 0.20
    ax_d.fill_between([q1, q3], -BW_D, BW_D, facecolor=S.RECAST_XL, alpha=0.9, edgecolor="none", zorder=1.5)
    ax_d.plot([q1, q3, q3, q1, q1], [-BW_D, -BW_D, BW_D, BW_D, -BW_D], color=S.RECAST_D, lw=0.7, zorder=2.5)
    ax_d.plot([wlo, q1], [0, 0], color=S.RECAST_D, lw=0.7, zorder=2.5)
    ax_d.plot([q3, whi], [0, 0], color=S.RECAST_D, lw=0.7, zorder=2.5)
    for xx in (wlo, whi):
        ax_d.plot([xx, xx], [-CAP_D, CAP_D], color=S.RECAST_D, lw=0.7, zorder=2.5)
    ax_d.plot([med, med], [-BW_D, BW_D], color=S.RECAST_D, lw=1.2, zorder=3.5)

    # Deterministic swarm positions for the 16 subtypes (compact vertical step so x=0 stack fits cleanly on Q1)
    XL_D = (-0.008, 0.235)
    mmx, mmy = D_W / (XL_D[1] - XL_D[0]), H2 / 2.2
    dia = np.sqrt(14) / 72 * 25.4 + 0.14
    slots = [0.0] + [v for k in range(1, 12) for v in (k * 0.095, -k * 0.095)]
    order = list(np.argsort(sd, kind="stable"))
    jy = np.zeros(len(sd))
    for i in order:
        for y in slots:
            if all(np.hypot((sd[i] - sd[j]) * mmx, (y - jy[j]) * mmy) > dia for j in order[:order.index(i)]):
                jy[i] = y
                break
        else:
            raise RuntimeError("no free slot")
    assert np.abs(jy).max() <= BW_D + 0.02

    ax_d.scatter(sd, jy, s=14, color=S.RECAST, alpha=0.88, edgecolors="white", linewidths=0.35, zorder=4)
    ax_d.scatter([mu_sd], [-0.66], s=22, marker="D", color=S.RECAST_D, edgecolors="white", linewidths=0.45, zorder=5)
    ax_d.text(mu_sd + 0.006, -0.66, f"mean {mu_sd:.3f}", va="center", ha="left", color=S.RECAST_D)
    S.refline(ax_d, x=MEDIAN_SEED_SD, zorder=2)
    med_t = ax_d.annotate(f"median {MEDIAN_SEED_SD:.3f}", (MEDIAN_SEED_SD, 0.72), xytext=(3.0, 0), textcoords="offset points",
                          va="center", ha="left")
    ax_d.set_ylim(-1.05, 1.08)
    ax_d.set_yticks([])
    ax_d.set_xlim(*XL_D)
    ax_d.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    ax_d.set_xticklabels(["0", "0.05", "0.1", "0.15", "0.2"])
    S.frame(ax_d, spines=("bottom",))
    ax_d.set_xlabel("SD of marker recovery across 10 joint draws")
    title(P, ax_d, "Joint cell sampling (100 target / 200 reference)")
    P.letter(0.5, R2 - 5.5, "d")

    # e: horizontal bars (mean), +-1 s.d. error bars, the 16 subtypes as points
    gm = baseline_choice.groupby("mode")["jac_vs_default"].agg(["mean", "std"])
    assert abs(float(gm.loc["mean_lognorm", "mean"]) - 1.0) < 1e-9
    E_X, E_W = 121.0, 57.0
    ax_e = P.ax(E_X, R2, E_W, H2)
    ys = np.arange(len(BASELINE_MODES))
    vals = [float(gm.loc[m, "mean"]) for m, _ in BASELINE_MODES]
    stds = [float(gm.loc[m, "std"]) for m, _ in BASELINE_MODES]

    bar_fills = [S.RECAST_XL if m == "mean_lognorm" else S.NULL_L for m, _ in BASELINE_MODES]
    bar_edges = [S.RECAST_D if m == "mean_lognorm" else S.DE for m, _ in BASELINE_MODES]
    ax_e.barh(ys, vals, height=0.62, color=bar_fills, edgecolor=bar_edges, lw=0.55, zorder=2)

    # Deterministic vertical jitter for the 16 subtype points per mode
    rng_e = np.random.default_rng(42)
    for idx, (m, _) in enumerate(BASELINE_MODES):
        sub_v = baseline_choice[baseline_choice["mode"] == m]["jac_vs_default"].to_numpy(float)
        assert len(sub_v) == 16
        if m == "mean_lognorm":
            ax_e.scatter([1.0], [idx], s=22, marker="D", color=S.RECAST_D, edgecolors="white", linewidths=0.45, zorder=5)
        else:
            yj = idx + rng_e.uniform(-0.18, 0.18, size=len(sub_v))
            ax_e.scatter(sub_v, yj, s=9.5, color=S.DE, alpha=0.75, edgecolors="white", linewidths=0.25, zorder=3.5)
            # +-1 SD error bar on the mean
            mu_v, sd_v = vals[idx], stds[idx]
            ax_e.errorbar([mu_v], [idx], xerr=[[min(sd_v, mu_v)], [min(sd_v, 1.0 - mu_v)]],
                          fmt="none", ecolor=S.BLACK, elinewidth=0.65, capsize=2.0, capthick=0.65, zorder=4.5)

    ax_e.axhline(2.5, color=S.HAIR, lw=S.HAIR_LW, zorder=1)  # profiles above, baseline-only below
    ax_e.set_yticks(ys)
    ax_e.set_yticklabels([lab for _, lab in BASELINE_MODES])
    ax_e.set_ylim(len(BASELINE_MODES) - 0.5, -0.5)
    ax_e.set_xlim(0, 1.05)
    ax_e.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_e.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    S.frame(ax_e)
    ax_e.tick_params(axis="y", length=0, pad=2)
    ax_e.set_xlabel("Jaccard overlap with default (mean \u00b1 1 s.d.)")
    title(P, ax_e, "Expression profiles and baselines")

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = K.px_per_mm(fig)
    e_lab_x0 = min(t.get_window_extent(rend).x0 for t in ax_e.get_yticklabels()) / k
    d_x1 = max([t.get_window_extent(rend).x1 for t in ax_d.get_xticklabels() + [ax_d.xaxis.label, med_t]]) / k
    checks["d_right_mm"], checks["e_labels_left_mm"] = round(d_x1, 2), round(e_lab_x0, 2)
    assert e_lab_x0 - d_x1 > 1.5, (d_x1, e_lab_x0)
    P.letter(e_lab_x0 - 3.5, R2 - 5.5, "e")

    # ============================================================== Row 3: f-h (minimal inter-panel gaps & aligned letters)
    ax_f = P.ax(13.5, R3, 46.5, H3)
    rec_f, rec_d = pooled_with_sem(perturbed_recast, "recall20", 14), pooled_with_sem(perturbed_wilcoxon, "recall20", 14)
    curves(ax_f, rec_f, rec_d)
    S.refline(ax_f, y=clean_recast, color=S.RECAST, alpha=0.75, zorder=1)
    S.refline(ax_f, y=clean_wilcoxon, alpha=0.9, zorder=1)
    yv = [v for d_ in (rec_f, rec_d) for _, y_, _ in d_.values() for v in y_] + [clean_recast, clean_wilcoxon]
    lo, hi = min(yv), max(yv)
    pad = 0.10 * (hi - lo)
    ax_f.set_ylim(lo - pad, hi + pad)
    ax_f.set_yticks([t / 100 for t in range(0, 101, 5) if lo - pad <= t / 100 <= hi + pad])
    ax_f.set_yticklabels([f"{t / 100:.2f}" for t in range(0, 101, 5) if lo - pad <= t / 100 <= hi + pad])
    noise_axis(ax_f)
    ax_f.set_ylabel("Marker recovery (20 genes)")
    title(P, ax_f, "Marker recovery (14 subtypes)")

    ax_g = P.ax(72.5, R3, 46.5, H3)
    jac_f, jac_d = pooled_with_sem(perturbed_recast, "jaccard20", 16), pooled_with_sem(perturbed_wilcoxon, "jaccard20", 16)
    curves(ax_g, jac_f, jac_d)
    ax_g.set_ylim(0.45, 1.0)
    ax_g.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_g.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    noise_axis(ax_g)
    ax_g.set_ylabel("Jaccard overlap with\nclean selection", linespacing=1.10)
    title(P, ax_g, "Selection overlap (16 subtypes)")

    ax_h = P.ax(131.5, R3, 46.5, H3)
    Cm = effective_shift_10
    ax_h.axhline(0.0, color=S.HAIR, lw=S.HAIR_LW, zorder=1)
    ax_h.scatter(Cm.eps_eff, Cm["drop"], s=17, color=S.RECAST, edgecolors="white", linewidths=0.45, alpha=0.92, zorder=5)

    S.frame(ax_h)
    xmax = max(1.03, float(Cm.eps_eff.max()) * 1.05)
    ax_h.set_xlim(-0.03, xmax)
    ax_h.set_ylim(-0.13, 0.08)
    ax_h.set_yticks([-0.1, -0.05, 0, 0.05])
    ax_h.set_yticklabels(["\u22120.1", "\u22120.05", "0", "0.05"])
    assert float(Cm["drop"].max()) < 0.06 and float(Cm["drop"].min()) > -0.12
    ax_h.set_xticks([t for t in (0, 0.25, 0.5, 0.75, 1.0) if t <= xmax])
    ax_h.set_xticklabels([f"{t:g}" for t in ax_h.get_xticks()])

    # Direct annotations clear of all points and lines (including gdT and ncMono defined in the SI caption)
    short = {"gdT": "gdT", "Non-classical monocytes": "ncMono", "basal": "basal", "Memory B cells": "Memory B"}
    offs = {
        "basal": (5, 0, "left", "center"),
        "gdT": (0, 4.5, "center", "bottom"),
        "ncMono": (5, 0, "left", "center"),
        "Memory B": (-5, -2.5, "right", "top"),
    }
    labs_ = []
    for _, r in Cm.iterrows():
        lab = short.get(r.state)
        if lab:
            ex, ey = round(float(r.eps_eff), 3), round(float(r["drop"]), 4)
            dx, dy, ha, va = offs[lab]
            labs_.append(ax_h.annotate(lab, (ex, ey), xytext=(dx, dy), textcoords="offset points",
                                       ha=ha, va=va))

    # Note the 11 unchanged subtypes sitting on y = 0 (in the empty upper-right region)
    labs_.append(ax_h.text(0.98, 0.74, "11/14 at 0", transform=ax_h.transAxes,
                           ha="right", va="bottom", color=S.DE))

    ax_h.set_xlabel("Relative membership change")
    ax_h.set_ylabel("Loss of marker recovery")
    title(P, ax_h, "RECAST, 10% boundary-focused")

    # Shared single-line key centered across Row 3 above f-h, so f, g, h panel letters sit on ONE horizontal line!
    leg_kw = dict(frameon=False, handlelength=2.0, borderaxespad=0, borderpad=0, handletextpad=0.45, columnspacing=1.35)
    h_rec = Line2D([0], [0], color=S.RECAST, lw=1.2, marker="o", ms=3.6, mec="white", mew=0.4, label="RECAST")
    h_wil = Line2D([0], [0], color=S.DE, lw=0.95, marker="o", ms=3.2, mec="white", mew=0.4,
                   label="Wilcoxon (tie-corrected)")
    h_bnd = Line2D([0], [0], color=S.BLACK, lw=0.85, ls="-", label="Boundary-focused")
    h_rnd = Line2D([0], [0], color=S.BLACK, lw=0.85, ls=":", label="Random relabelling")
    h_cln = Line2D([0], [0], color=S.BLACK, lw=S.LW, ls=S.DASH, label="Clean labels (f)")
    KY = R3 - 8.8
    leg1 = fig.legend(handles=[h_rec, h_wil, h_bnd, h_rnd, h_cln], loc="center", ncol=5,
                      bbox_to_anchor=(0.50, 1 - KY / H), **leg_kw)

    # All three Row 3 panel letters sit on the exact same horizontal line (R3 - 5.3 mm)!
    P.letter(0.5, R3 - 5.3, "f")
    P.letter(59.8, R3 - 5.3, "g")
    P.letter(119.0, R3 - 5.3, "h")

    # ============================================================== checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax_h.transData.transform(np.c_[Cm.eps_eff, Cm["drop"]])
    r_px = np.sqrt(18) / 2 * fig.dpi / 72
    for t in labs_:
        bb = mtext.Text.get_window_extent(t, rend).padded(r_px)
        n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
        assert n == 0, f"label {t.get_text()} covers {n} points"
    top_titles = min((fig.bbox.y1 - t.get_window_extent(rend).y1) / k for t in fig.texts
                     if t.get_position()[1] > 1 - (R3 - 1.0) / H and t.get_position()[1] < 1 - (R3 - 8.5) / H)
    key_bottom = (fig.bbox.y1 - leg1.get_window_extent(rend).y0) / k
    checks["row3_key_bottom_mm"], checks["row3_title_top_mm"] = round(key_bottom, 2), round(top_titles, 2)
    assert top_titles - key_bottom > 0.8, (key_bottom, top_titles)
    checks.update(
        a_mean=ga.set_index("n_steps")["mean"].round(3).to_dict(),
        b_mean=gb.set_index("n_cells")["mean"].round(3).to_dict(), b_all_median_cells=fxb,
        c_mean=gc.set_index("ref_size")["mean"].round(3).to_dict(), c_all_median_cells=fxc,
        d_median_sd=round(MEDIAN_SEED_SD, 5), d_mean_sd=round(mu_sd, 5),
        e_means={lab: round(v, 3) for (_, lab), v in zip(BASELINE_MODES, vals)},
        f_clean={"RECAST": round(float(clean_recast), 4), "Wilcoxon": round(float(clean_wilcoxon), 4)},
        h_points=len(Cm), page_mm=(W, H))
    P.save(f"{HERE}/out", STEM)
    print(STEM, checks)

if __name__ == "__main__":
    main()
