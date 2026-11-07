"""Fig. 2d, f, g: what RECAST's marker recovery comes from.

  d  panel_d  random encoders, no model (the mean-expression profile) and the trained encoder, over the 31 dataset-
              lineage groups (out/tables/marker_recovery_by_group.csv and _tests.csv, from 01_marker_recovery.py)
  f  panel_f  2 x 2 factorial: target aggregation (per cell / population) x representation (no encoder / encoder),
              mean +- s.e.m. over the 97 subtypes (data/aggregation_and_encoder.csv)
  g  panel_g  target sample size: the target cells split into groups of n, each group with its own profile and
              contrast; n = 1 and n = all are two cells of f (data/target_sample_size.csv)
Each panel is one 44.2 x 48 mm slot of the row; the x axes of the row sit on one line.  Panels are saved to OUT (set
by 02_draw.py).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import panel_style                                                            # noqa: E402,F401  (house style)
from method_colours import (apply_style, mm, save_fig, INK, G700, G550, G400, G100,   # noqa: E402
                            RECAST, RECAST_D, DE_L, REF_FREE, NULLC)
import recovery_axes as RA                                                    # noqa: E402
from panels_a_to_c import tick_values                                         # noqa: E402

DATA = os.path.join(HERE, "data")
TABLES = os.path.join(HERE, "out", "tables")
OUT = os.path.join(HERE, "out", "panels")
apply_style()

DOSE_YLIM = (0.255, 0.400)                  # f and g share one recovery axis
DOSE_YTICK = [0.26, 0.30, 0.34, 0.38]
SLOT_W, SLOT_H = 44.2, 48.0                 # mm


def _bare(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2, labelsize=5.6)


def panel_d():
    """Per method a box over the 31 groups, drawn as c's: box = interquartile range, line = median, whiskers to the
    last group within 1.5 x IQR, diamond = mean (printed above the plot), groups as points joined across the methods.
    Brackets: trained vs no model (group-level sign-flip test), trained vs random encoders (empirical rank)."""
    G = pd.read_csv(os.path.join(TABLES, "marker_recovery_by_group.csv"), index_col=0)
    st = pd.read_csv(os.path.join(TABLES, "marker_recovery_tests.csv")).set_index("comparison")
    p_nm = st.loc["RECAST - no_model", "p_group"]
    p_rand = st.loc["RECAST vs 20 random encoders", "p_group"]
    top = float(max(G[["RECAST", "reference_free", "no_model", "random_mean"]].max()))   # shared with panel c
    ticks = tick_values(top)
    groups = list(G.index)
    random_mean = G[[f"random_{k}" for k in range(20)]].mean(1)
    rows = [({g: float(random_mean[g]) for g in groups}, "random\nencoder", NULLC),
            ({g: float(G.loc[g, "no_model"]) for g in groups}, "no model\n(profile)", DE_L),
            ({g: float(G.loc[g, "RECAST"]) for g in groups}, "trained\nencoder", RECAST)]
    fig = plt.figure(figsize=(mm(SLOT_W), mm(SLOT_H)))
    ax = fig.add_axes([0.235, 0.170, 0.740, 0.775])
    xs = np.arange(len(rows))
    UNIT = 0.7792                               # c's x unit / d's: c's box, cap and point-offset widths on the page
    BW, CAP = 0.30 * UNIT, 0.17 * UNIT
    offs = np.linspace(-0.22, 0.22, len(groups)) * UNIT     # one fixed x offset per group, shared by its points
    top_all = max(max(r[0].values()) for r in rows)
    off = top_all * 0.05                        # the means' row sits this far above the highest point
    for i, g in enumerate(groups):              # each group's values, joined
        ax.plot(xs + offs[i], [r[0][g] for r in rows], "-", color=G400, lw=0.3, zorder=2.5)
    for x0, (vals, lab, col) in zip(xs, rows):
        v = np.array([vals[g] for g in groups], float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        hero = col == RECAST                    # c's RECAST style; the baselines take c's reference-free style
        fill, alpha, edge, medc, pc, dc = ((RECAST, 0.20, RECAST_D, RECAST_D, RECAST, RECAST_D) if hero else
                                           (G100, 1.0, G550, G700, REF_FREE, REF_FREE))
        ax.fill_between([x0 - BW, x0 + BW], q1, q3, facecolor=fill, alpha=alpha, edgecolor="none", zorder=2)
        ax.plot([x0 - BW, x0 + BW, x0 + BW, x0 - BW, x0 - BW], [q1, q1, q3, q3, q1], color=edge, lw=0.7, zorder=3)
        ax.plot([x0, x0], [inside.min(), q1], color=edge, lw=0.7, zorder=3)
        ax.plot([x0, x0], [q3, inside.max()], color=edge, lw=0.7, zorder=3)
        for yy in (inside.min(), inside.max()):
            ax.plot([x0 - CAP, x0 + CAP], [yy, yy], color=edge, lw=0.7, zorder=3)
        ax.plot([x0 - BW, x0 + BW], [med, med], color=medc, lw=1.1, zorder=4)
        ax.scatter(x0 + offs, v, s=5, color=pc, edgecolors="none", zorder=4.5)
        mu = v.mean()
        ax.scatter([x0], [mu], s=30, marker="D", color=dc, edgecolors="white", linewidths=0.6, zorder=5)
        ax.text(x0, top_all + off, f"{mu:.3f}", ha="center", va="bottom", fontsize=5.4, color=INK,
                fontweight="bold" if hero else "normal", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9))
    ax.set_xlim(-0.441, 2.441)
    ax.set_xticks(xs); ax.set_xticklabels([r[1] for r in rows], fontsize=5.6)
    for t, r in zip(ax.get_xticklabels(), rows):
        if r[2] == RECAST:
            t.set_color(RECAST_D); t.set_fontweight("bold")
    ax.set_ylabel(RA.LABEL, fontsize=6.2, labelpad=1.6)
    ax.set_ylim(0, round(top * 1.52, 4))
    tick = round(top * 0.025, 4)
    RA.bracket(ax, 1, 2, round(top * 1.20, 4), tick, RA.ptxt(p_nm))
    RA.bracket(ax, 0, 2, round(top * 1.36, 4), tick, RA.ptxt(p_rand))
    _bare(ax); ax.tick_params(axis="x", length=0, pad=3.5 + 2.2)
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v:.2f}" for v in ticks])
    RA.ylabel(fig, ax)
    return save_fig(fig, "panel_d", OUT)


def panel_f():
    """The 2 x 2 factorial drawn as an interaction plot: one line per representation across the two aggregations."""
    T = pd.read_csv(os.path.join(DATA, "aggregation_and_encoder.csv"))
    cells = ["per_cell_no_encoder", "per_cell_encoder", "population_no_encoder", "population_encoder"]
    mu, se = T[cells].mean(), T[cells].sem()
    fig = plt.figure(figsize=(mm(SLOT_W), mm(SLOT_H)))
    ax = fig.add_axes([0.315, 0.170, 0.590, 0.775])
    xs = [0, 1]
    rows = [(["per_cell_no_encoder", "population_no_encoder"], "no encoder", DE_L, 1.2, "white"),
            (["per_cell_encoder", "population_encoder"], "encoder", RECAST, 1.9, None)]
    for keys, lab, col, lw, mfc in rows:
        ax.errorbar(xs, [mu[k] for k in keys], yerr=[se[k] for k in keys], color=col, lw=lw, marker="o", ms=4.6,
                    capsize=2.0, elinewidth=0.8, capthick=0.8, mfc=mfc or col, mec=col, mew=1.0,
                    zorder=4 if col == RECAST else 3, label=lab)
    # the population x encoder cell is RECAST
    ax.annotate("RECAST", xy=(1, mu.population_encoder), xytext=(1.09, mu.population_encoder),
                fontsize=5.8, color=RECAST_D, fontweight="bold", ha="left", va="center")
    ax.set_xlim(-0.30, 1.46)
    ax.set_ylim(*DOSE_YLIM); ax.set_yticks(DOSE_YTICK)
    ax.set_xticks(xs); ax.set_xticklabels(["per-cell", "population"], fontsize=6.0)
    ax.set_xlabel("target aggregation", fontsize=6.2, labelpad=1.8)
    ax.set_ylabel(RA.LABEL, fontsize=6.0, linespacing=1.35, labelpad=1.6)
    # a categorical axis draws no tick marks: pad by the missing tick length, so the x labels of the row share one line
    _bare(ax); ax.tick_params(axis="x", length=0, pad=3.5 + 2.2)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=5.6, frameon=False,
                    handlelength=1.5, handletextpad=0.5, labelspacing=0.35, borderaxespad=0.0)
    for t in leg.get_texts():
        if t.get_text() == "encoder":
            t.set_color(RECAST_D); t.set_fontweight("bold")
    RA.ylabel(fig, ax)
    return save_fig(fig, "panel_f", OUT)


def _log_axis_need(xs, labels, lim, fs, gap_mm):
    """Axis width (mm) at which neighbouring tick labels of a log axis keep gap_mm between them (measured)."""
    pf = plt.figure(figsize=(mm(20), mm(10)))
    r = pf.canvas.get_renderer()
    w = [pf.text(0, 0, s, fontsize=fs).get_window_extent(r).width / pf.dpi * 25.4 for s in labels]
    plt.close(pf)
    lx = np.log10(np.asarray(xs, float))
    per_dec = max(((a + b) / 2 + gap_mm) / (x1 - x0) for a, b, x0, x1 in zip(w, w[1:], lx, lx[1:]))
    return per_dec * np.log10(lim[1] / lim[0])


def panel_g():
    """Recovery against the number of target cells per group (mean over random partitions per subtype, then mean +-
    s.e.m. over the 97 subtypes); the dashed line is the per-cell level (n = 1), the diamond n = all (RECAST)."""
    D = pd.read_csv(os.path.join(DATA, "target_sample_size.csv"))
    units = set(pd.read_csv(os.path.join(DATA, "aggregation_and_encoder.csv")).unit)
    D = D[D.unit.isin(units)]
    assert D.unit.nunique() == 97, D.unit.nunique()
    pm = D.groupby(["unit", "n_cells"]).recall.mean().unstack("n_cells")      # partition mean per subtype
    ends = D.groupby("unit")[["recall_n1", "recall_full"]].first()
    pm.insert(0, 1, ends.recall_n1)
    grid = pd.DataFrame({"mean": pm.mean(0), "sem": pm.sem(0), "count": pm.count(0)})
    full = ends.recall_full
    ns = list(grid.index)
    x_full = max(ns) * 2.6                      # a decade clear of the last fixed n
    # the axis is as wide as neighbouring tick labels need to stay >= 0.5 mm apart; margins and height are the slot's
    AXW = max(0.660 * SLOT_W, _log_axis_need(ns + [x_full], [str(n) for n in ns] + ["all"],
                                             (0.72, x_full * 1.9), 6.2, 0.5))
    FW = SLOT_W + AXW - 0.660 * SLOT_W
    rect = [0.315 * SLOT_W / FW, 0.170, AXW / FW, 0.775]
    fig = plt.figure(figsize=(mm(FW), mm(SLOT_H)))
    ax = fig.add_axes(rect)
    ax.fill_between(ns, grid["mean"] - grid["sem"], grid["mean"] + grid["sem"], color=RECAST, alpha=0.16, lw=0,
                    zorder=1)
    ax.plot(ns, grid["mean"], color=RECAST, lw=1.8, marker="o", ms=3.8, mfc=RECAST, mec=RECAST, zorder=3)
    ax.errorbar([x_full], [full.mean()], yerr=[full.sem()], color=RECAST_D, marker="D", ms=4.6,
                capsize=2.0, elinewidth=0.8, capthick=0.8, mfc=RECAST_D, mec="white", mew=0.6, zorder=5)
    ax.plot([ns[-1], x_full], [grid["mean"].iloc[-1], full.mean()], color=RECAST, lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.axhline(grid.loc[1, "mean"], color=G400, lw=0.7, ls=(0, (3, 2.5)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.72, x_full * 1.9)
    ax.set_xticks(ns + [x_full])
    ax.set_xticklabels([str(n) for n in ns] + ["all"], fontsize=5.6)
    ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.minorticks_off()
    ax.set_ylim(*DOSE_YLIM); ax.set_yticks(DOSE_YTICK)
    ax.set_xlabel("target cells per group  (n, log)", fontsize=6.2, labelpad=1.8)
    ax.set_ylabel(RA.LABEL, fontsize=6.0, linespacing=1.35, labelpad=1.6)
    _bare(ax)
    # "RECAST" above the band beside its point (the empty second line keeps it there)
    ax.annotate("RECAST\n", xy=(x_full, full.mean()), xytext=(-9, 10), textcoords="offset points",
                fontsize=5.3, color=RECAST_D, fontweight="bold", ha="right", va="bottom", linespacing=1.4)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bbs = [t.get_window_extent(rend) for t in ax.get_xticklabels() if t.get_text()]
    assert min((b.x0 - a.x1) / fig.dpi * 25.4 for a, b in zip(bbs, bbs[1:])) >= 0.45, "x tick labels touch"
    RA.ylabel(fig, ax)
    return save_fig(fig, "panel_g", OUT)
