"""Fig. 6a-c: reusing RECAST programs to score single cells.

  a  panel_a  per task, the change in balanced accuracy and in macro-F1 when cells are assigned with RECAST
              programs instead of ANS or reference-free programs, with donor-clustered 95% intervals; the 22 tasks
              with a donor-level interval are shown (data/reuse_by_task*.csv, data/reuse_donor_intervals.csv)
  b  panel_b  CD8 T cells (Dominguez Conde) on one embedding: the annotated cells of one subtype, and the same
              program scored by RECAST, ANS and the reference-free method on one shared colour scale
  c  panel_c  per subtype, the program score of the subtype's own cells and of all other cells, by method
Data: data/cd8_cells.csv (embedding, annotation, RECAST and ANS scores), data/cd8_cells_reference_free.csv,
data/cd8_summary.json (per-program AUROC), data/cd8_auroc_by_method.csv.  Panels are saved to OUT (set by 01_draw.py).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from panel_style import LW as RULE_LW                                   # noqa: E402
from method_colours import (apply_style, mm, save_fig, neutral_cmap, INK, SUB, G700, G550, G400,  # noqa: E402
                            G300, G200, BIO, RECAST, RECAST_D, RECAST_XL, DE)

D = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "panels")
apply_style()

REFFREE = "#7E9E77"      # reference-free method: sage
REFFREE_XL = "#DCE6D9"   # light sage: its "other cells" boxes
REFFREE_D = "#4F6B49"    # dark sage: edges and titles
NAME = {"RECAST": "RECAST", "ANS": "ANS", "SIGnature": "Reference-free"}   # the reference-free method's data key
                                                                           # is "SIGnature"

CD8 = dict(states=["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"],
           sdisp={"Tcm/Naive cytotoxic T cells": "Tcm/naive", "Tem/Temra cytotoxic T cells": "Tem/Temra",
                  "Tem/Trm cytotoxic T cells": "Tem/Trm", "gdT": "γδ T"},
           scol={"Tcm/Naive cytotoxic T cells": BIO[2], "Tem/Temra cytotoxic T cells": BIO[6],
                 "Tem/Trm cytotoxic T cells": BIO[1], "gdT": BIO[4]})

ROW_H = 58.0        # mm: height of panels b and c
FS_BODY = 5.9       # pt: body, tick and label text
FS_NOTE = 5.8       # pt: keys
FS_HEAD = 6.4       # pt: map titles


# ===================================================================================================== panel a
BLOCKS = [dict(fcol="recast_ba", acol="ans_ba", scol="sig_gated_ba", mkey="ba", title="Δ balanced accuracy"),
          dict(fcol="recast_f1", acol="ans_f1", scol="sig_gated_f1", mkey="f1", title="Δ macro-F1")]
PAD = 0.016          # Δ units past the shared window's min / max
STEP = 0.05          # tick pitch
REGIME = {"fine": "Fine", "distinct": "Distinct"}
PITCH = 3.00         # mm per table row
DOT = 5.82           # marker area, pt^2 (0.96 mm across)
DODGE = 0.205        # vertical offset of the two comparators, row units
BAND = 0.82          # zebra band height, row units
CI_W = 17.30         # mm, one interval column
CI_GAP = 1.00        # mm, gutter between the forest and a column, and between the two columns
WHISKER = 0.7        # pt
REG_COLOUR = G700
KEY_X0 = 60.955833   # mm: left end of the key row above the table
COLUMN_SHIFT = {"ans": 3.0, "sig": 0.4}   # pt: the ANS and reference-free interval columns sit this far left of their
                                          # computed right edges
LINEAGE = {"cd8": "CD8 T", "cd4": "CD4 T", "mono": "Mono", "b": "B cell", "nk": "NK", "asc": "ASC"}
DATASET = {"steph": "Stephenson", "yosh": "Yoshida", "dom": "Domínguez", "ren": "Ren", "zhao": "Zhao",
           "pbmcbmn": "pbmcbmn"}
DISTINCT = {"baron_pancreas": "Baron pancreas", "pelka_crc_epi": "Pelka colon", "dominguez_conde": "Domínguez",
            "ss3_pbmc": "SS3 PBMC", "pbmc3k": "PBMC 3k", "lawlor": "Lawlor", "yoshida": "Yoshida",
            "stephenson": "Stephenson", "ren": "Ren"}


def _task_label(name):
    """(lineage, dataset) for a task id: 'tf_cd8_steph' -> ('CD8 T', 'Stephenson'); distinct-type tasks are one
    dataset each and carry no lineage."""
    if name in DISTINCT:
        return "", DISTINCT[name]
    p = name.split("_")
    if p[0] in ("tf", "fs") and len(p) >= 3 and p[1] in LINEAGE:
        rest = "_".join(p[2:])
        return LINEAGE[p[1]], DATASET.get(rest, rest)
    return "", name


def _tick_step(fig, lims, fw_mm, fs):
    """The smallest tick pitch whose labels fit the forest width with 1.6 mm between them."""
    r = fig.canvas.get_renderer()
    for step in (STEP, 2 * STEP, 4 * STEP):
        tk = [t * step for t in range(-9, 10) if lims[0] < t * step < lims[1]]
        if not tk:
            continue
        w = 0.0
        for t in tk:
            tx = fig.text(0, 0, ("0" if abs(t) < 1e-9 else f"{t:.2f}".replace("-", "−")), fontsize=fs)
            w += tx.get_window_extent(renderer=r).width / fig.dpi * 25.4
            tx.remove()
        if w + 1.6 * (len(tk) - 1) <= fw_mm:
            return step, tk
    return 4 * STEP, [t * 4 * STEP for t in range(-9, 10) if lims[0] < t * 4 * STEP < lims[1]]


def panel_a():
    df = pd.read_csv(f"{D}/reuse_by_task.csv").set_index("dataset")
    sg = pd.read_csv(f"{D}/reuse_by_task_reference_free.csv").set_index("dataset")
    cols = ["recast_ba", "recast_f1", "recast_auc", "ans_ba", "ans_f1", "ans_auc"]
    assert not (df[cols] - sg[cols]).abs().to_numpy().any(), "the two tables come from different runs"
    ci = pd.read_csv(f"{D}/reuse_donor_intervals.csv").set_index(["dataset", "comparator", "metric"])
    # the tasks with a donor-level interval: all four (2 comparators x 2 metrics) or none
    n = (ci[ci.index.get_level_values("metric").isin([b["mkey"] for b in BLOCKS])]
         .ci_lo.notna().groupby(level="dataset").sum())
    assert set(n.unique()) <= {0, 4}, f"task with a partial set of intervals: {n[~n.isin([0, 4])]}"
    shown = list(n[n == 4].index)
    df, sg = df.loc[df.index.isin(shown)], sg.loc[sg.index.isin(shown)]
    ci = ci[ci.index.get_level_values("dataset").isin(shown)]
    df["dba"] = df["recast_ba"] - df["ans_ba"]
    order, regs = [], []                                 # fine tasks, then distinct, each by Δ balanced accuracy
    for gkey in ("fine", "distinct"):
        ids = list(df[df.reg == gkey].sort_values("dba", ascending=False).index)
        order += ids
        regs += [REGIME[gkey]] * len(ids)
    rowmap = list(enumerate(order))
    nrow = len(rowmap)
    dvals = [(df[b["fcol"]] - df[b["acol"]]) for b in BLOCKS]
    svals = [(sg[b["fcol"]] - sg[b["scol"]]) for b in BLOCKS]
    # one window for both blocks, wide enough for every point and interval
    lo = min(min(float(v.min()), float(w.min())) for v, w in zip(dvals, svals))
    hi = max(max(float(v.max()), float(w.max())) for v, w in zip(dvals, svals))
    mk = [b["mkey"] for b in BLOCKS]
    lo = min(lo, float(ci[ci.index.get_level_values("metric").isin(mk)].ci_lo.min()))
    hi = max(hi, float(ci[ci.index.get_level_values("metric").isin(mk)].ci_hi.max()))
    lims = [(lo - PAD, hi + PAD)] * len(BLOCKS)

    W = 177.0
    H = 20.5 + (nrow + 4.6) * PITCH
    FS_ROW, FS_TICK, FS_HDR, FS_KEY = 6.5, 6.5, 6.8, 6.5
    fig = plt.figure(figsize=(mm(W), mm(H)))
    YB, YT = 15.0 / H, (H - 5.5) / H                     # 15 mm bottom margin: tick labels, arrows, caption
    X0, X1 = 0.225, 0.992                                # X0: the two label columns
    gap = 0.034
    nb = len(BLOCKS)
    avail = (X1 - X0) - gap * (nb - 1)
    bws = [avail / nb] * nb
    xs = [X0 + (bws[0] + gap) * i for i in range(nb)]
    ciw, cig = CI_W / W, CI_GAP / W
    fws = [b - 2 * (ciw + cig) for b in bws]             # forest width inside each block
    assert fws[0] * W > 25.0, f"forest collapsed to {fws[0] * W:.1f} mm"
    nxs = [[xs[i] + fws[i] + cig + j * (ciw + cig) + ciw for j in range(2)] for i in range(nb)]

    # table coordinates in row units (the axes are inverted: smaller y is higher on the page)
    Y_KEY, Y_TOP, Y_HDR, Y_MID = -3.45, -2.60, -1.50, -0.55
    Y_BOT = nrow - 1 + 0.55
    ymin, ymax = Y_KEY - 0.75, Y_BOT + 0.85

    # zebra bands and three rules, under every column
    axbg = fig.add_axes([0.0, YB, X1, YT - YB]); axbg.set_xlim(0, 1); axbg.set_ylim(ymax, ymin)
    axbg.axis("off"); axbg.patch.set_visible(False)
    for y, _ds in rowmap:
        if y % 2 == 1:
            axbg.add_patch(Rectangle((0, y - BAND / 2), 1, BAND, color=G200, lw=0, zorder=0))
    for yy in (Y_TOP, Y_MID, Y_BOT):
        axbg.plot([0, 1], [yy, yy], color="#000000", lw=RULE_LW, zorder=3, solid_capstyle="butt")
    # key: each glyph is the row glyph (a whisker through a dot)
    rend = fig.canvas.get_renderer()

    def tw(txt, fs=FS_KEY):
        t = axbg.text(0, 0, txt, fontsize=fs); w_ = t.get_window_extent(renderer=rend).width / fig.dpi * 25.4
        t.remove(); return w_
    GLYPH, GPAD, SEP = 4.2, 1.2, 6.0                     # mm: glyph run, glyph-to-text pad, item gap
    kx = KEY_X0
    for c, lab in ((DE, "RECAST − ANS"), (REFFREE, "RECAST − Reference-free")):
        x0, x1_ = kx / W / X1, (kx + GLYPH) / W / X1
        axbg.plot([x0, x1_], [Y_KEY, Y_KEY], color=c, lw=WHISKER, zorder=3, solid_capstyle="butt", clip_on=False)
        axbg.scatter([(x0 + x1_) / 2], [Y_KEY], s=DOT, marker="o", color=c, lw=0, zorder=4, clip_on=False)
        axbg.text((kx + GLYPH + GPAD) / W / X1, Y_KEY, lab, fontsize=FS_KEY, color=INK, ha="left", va="center")
        kx += GLYPH + GPAD + tw(lab) + SEP
    assert kx - SEP <= X1 * W, "the key row runs past the table"

    # label columns: regime, then "dataset – lineage (n)"
    axl = fig.add_axes([0.0, YB, X0, YT - YB]); axl.set_xlim(0, X0 * W); axl.set_ylim(ymax, ymin)
    axl.axis("off"); axl.patch.set_visible(False)
    XR, XT = 0.8, 12.2                                   # mm
    axl.text(XR, Y_HDR, "Regime", fontsize=FS_HDR, fontweight="bold", color=INK, ha="left", va="center")
    axl.text(XT, Y_HDR, "Dataset – lineage (n)", fontsize=FS_HDR, fontweight="bold", color=INK,
             ha="left", va="center")
    for (y, ds), rg in zip(rowmap, regs):
        lin, dsn = _task_label(ds)
        k = int(df.loc[ds, "n_states"])                  # number of cell types the task tells apart
        lab = f"{dsn} – {lin} (n={k})" if lin else f"{dsn} (n={k})"
        axl.text(XR, y, rg, fontsize=FS_ROW, color=REG_COLOUR, ha="left", va="center")
        axl.text(XT, y, lab, fontsize=FS_ROW, color=INK, ha="left", va="center")

    # the two metric blocks: forest, then one interval column per comparator
    for bi, b in enumerate(BLOCKS):
        ax = fig.add_axes([xs[bi], YB, fws[bi], YT - YB])
        ax.patch.set_visible(False)
        ax.set_ylim(ymax, ymin); ax.set_xlim(*lims[bi])
        ax.plot([0, 0], [Y_MID, Y_BOT], color=G550, ls=(0, (2.2, 1.8)), lw=0.6, zorder=1)
        for y, ds in rowmap:
            for key, col, off, vals in (("ans", DE, -DODGE, dvals[bi]), ("sig", REFFREE, +DODGE, svals[bi])):
                r = ci.loc[(ds, key, b["mkey"])]
                pt, yy = float(vals[ds]), y + off
                assert abs(pt - float(r.delta)) < 1e-9, f"{ds} {key} {b['mkey']}: point != interval centre"
                ax.plot([float(r.ci_lo), float(r.ci_hi)], [yy, yy], color=col, lw=WHISKER, zorder=3,
                        solid_capstyle="butt")
                ax.scatter([pt], [yy], s=DOT, marker="o", color=col, lw=0, zorder=4)
        ax.set_yticks([])
        _step, tk = _tick_step(fig, lims[bi], fws[bi] * W, FS_TICK)
        ax.set_xticks(tk)
        ax.set_xticklabels([("0" if abs(t) < 1e-9 else f"{t:.2f}".replace("-", "−")) for t in tk])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(G550); ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(axis="x", labelsize=FS_TICK, length=2, width=0.5, colors=G550, labelcolor=SUB, pad=1.6)
        axbg.text((xs[bi] + fws[bi] / 2) / X1, Y_HDR, b["title"], fontsize=FS_HDR,
                  fontweight="bold", color=INK, ha="center", va="center")
        for j, (hd, col, key) in enumerate((("ANS", DE, "ans"), ("Reference-free", REFFREE, "sig"))):
            xr = (nxs[bi][j] - COLUMN_SHIFT[key] * 25.4 / 72 / W) / X1
            axbg.text(xr, Y_HDR, hd, fontsize=FS_HDR, fontweight="bold", color=col, ha="right", va="center")
            for y, ds in rowmap:
                r = ci.loc[(ds, key, b["mkey"])]
                txt = f"({float(r.ci_lo):+.3f}, {float(r.ci_hi):+.3f})".replace("-", "−")
                axbg.text(xr, y, txt, fontsize=FS_ROW, color=INK, ha="right", va="center", zorder=4)

    # one arrow under each forest, pointing to the RECAST-better end, and the caption under both
    deco = fig.add_axes([0, 0, 1, 1]); deco.axis("off"); deco.patch.set_visible(False)
    deco.set_xlim(0, 1); deco.set_ylim(0, 1)
    y_arrow, y_cap = 9.4 / H, 6.4 / H
    for bi in range(nb):
        deco.annotate("", xy=(xs[bi] + fws[bi], y_arrow), xytext=(xs[bi], y_arrow),
                      arrowprops=dict(arrowstyle="-|>", color=G550, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=8))
    deco.text(X1 / 2, y_cap, "Δ = RECAST − comparator · higher values favour RECAST",
              ha="center", va="center", fontsize=7.2, color=INK)
    fig.patch.set_facecolor("white")
    return save_fig(fig, "panel_a", OUT)


# ===================================================================================================== panel b
FLOOR_NEUTRAL = 0.53
MAP_TOP = 0.80


def _grey_floor_cmap(cmap, lo, anchor=G300, blend=0.25):
    """cmap from `lo` to its top, with its bottom `blend` fraction cross-faded into `anchor` (the grey of the cells
    outside the annotated subtype), so that the lowest scores print in that grey."""
    from matplotlib.colors import to_rgb
    cols = cmap(np.linspace(lo, 1.0, 256))[:, :3].copy()
    a = np.asarray(to_rgb(anchor), dtype=float)
    k = max(2, int(round(blend * cols.shape[0])))
    w = np.linspace(1.0, 0.0, k)[:, None]
    cols[:k] = w * a + (1.0 - w) * cols[:k]
    return LinearSegmentedColormap.from_list(cmap.name + "_greyfloor", cols)


def map_cmap():
    """The one ramp every score map of panel b is drawn through (grey to dark grey)."""
    base = _grey_floor_cmap(neutral_cmap(), FLOOR_NEUTRAL)
    return LinearSegmentedColormap.from_list("map_seq", base(np.linspace(0.0, MAP_TOP, 256)))


def _bar_ticks(lo, hi, n=4):
    """Round tick values strictly inside a colour bar's [lo, hi] window."""
    from matplotlib.ticker import MaxNLocator
    t = [float(v) for v in MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
         if lo <= v <= hi]
    return t or [lo, hi]


def panel_b(w_mm=34.2):
    """2 x 2 maps on the CD8 T embedding: the annotated cells of the subtype where RECAST's AUROC most exceeds ANS's,
    then that subtype's program scored by RECAST, ANS and the reference-free method (z-scores, one colour scale: the
    1st-99th percentile of all three)."""
    STATES, SDISP, SCOL = CD8["states"], CD8["sdisp"], CD8["scol"]
    d = pd.read_csv(f"{D}/cd8_cells.csv")
    au = json.load(open(f"{D}/cd8_summary.json"))["per_program_auroc"]
    d_state = max(STATES, key=lambda s: au["RECAST"][s] - au["ANS"][s])
    xy = d[["umap_x", "umap_y"]].values
    padx = (xy[:, 0].max() - xy[:, 0].min()) * 0.04; pady = (xy[:, 1].max() - xy[:, 1].min()) * 0.04
    xlim = (xy[:, 0].min() - padx, xy[:, 0].max() + padx)
    ylim = (xy[:, 1].min() - pady, xy[:, 1].max() + pady)
    sdt = SDISP[d_state]
    dsig = pd.read_csv(f"{D}/cd8_cells_reference_free.csv")
    assert (dsig.gt_state.values == d.gt_state.values).all(), "the two cell tables are in different orders"
    maps = [("RECAST", d, f"RECAST::{d_state}", map_cmap(), RECAST_D),
            ("ANS", d, f"ANS::{d_state}", map_cmap(), SUB),
            ("SIGnature", dsig, f"SIGnature::{d_state}", map_cmap(), REFFREE_D)]
    fig = plt.figure(figsize=(mm(w_mm), mm(ROW_H)))
    ASPECT = 0.312 * 65.5 / (0.385 * ROW_H)             # the embedding's width : height
    W = 15.97 / w_mm                                    # map width
    h = (W * w_mm / ASPECT) / ROW_H
    GX = 1.2 / w_mm                                     # column gap
    xs = [0.5 / w_mm, 0.5 / w_mm + W + GX]
    Y_TOP, Y_BOT = 35.15 / ROW_H, 9.2 / ROW_H           # map bottoms
    rects = [(xs[0], Y_TOP, W, h), (xs[1], Y_TOP, W, h), (xs[0], Y_BOT, W, h), (xs[1], Y_BOT, W, h)]
    cb_y = 3.45 / ROW_H
    axes = [fig.add_axes(list(r)) for r in rects]
    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off"); ax.set_aspect("equal")

    gtm = d.gt_state.values == d_state                  # the annotated subtype; the other subtypes in grey
    axes[0].scatter(xy[~gtm, 0], xy[~gtm, 1], s=1.6, color=G300, linewidths=0, alpha=0.85, zorder=2)
    axes[0].scatter(xy[gtm, 0], xy[gtm, 1], s=1.6, color=SCOL[d_state], linewidths=0, alpha=0.95, zorder=3)
    axes[0].set_title(f"Annotated\n{sdt} cells", fontsize=FS_HEAD, color=INK, pad=2.5, linespacing=1.15)

    zs = []
    for mth, src, col, cmap, tcol in maps:
        v = src[col].values.astype(float)
        zs.append((v - v.mean()) / (v.std() + 1e-12))
    zall = np.concatenate(zs)
    lo, hi = np.percentile(zall, [1, 99]); hi = max(hi, lo + 1e-9)
    nrm = Normalize(vmin=lo, vmax=hi)
    for (ax, (mth, src, col, cmap, tcol)), z in zip(zip(axes[1:], maps), zs):
        order = np.argsort(z)
        ax.scatter(xy[order, 0], xy[order, 1], s=1.6, c=z[order], cmap=cmap, norm=nrm, linewidths=0, zorder=2)
        ax.set_title(f"{NAME[mth]}\n{sdt}", fontsize=FS_HEAD, color=tcol, pad=2.5, linespacing=1.15)
    # one colour bar under both columns, both ends labelled
    span = (xs[1] + W) - xs[0]
    xmid = 0.5 * (xs[0] + xs[1] + W)
    CB1 = 0.86 * span
    cax = fig.add_axes([xmid - 0.5 * CB1, cb_y, CB1, 1.5 / ROW_H])
    cb = fig.colorbar(ScalarMappable(norm=nrm, cmap=maps[0][3]), cax=cax, orientation="horizontal")
    pad = 0.16 * (hi - lo)
    ticks = [lo] + [t for t in _bar_ticks(lo, hi, n=3) if lo + pad < t < hi - pad] + [hi]
    cb.set_ticks(ticks)
    cb.set_ticklabels([(f"{t:.1f}" if i in (0, len(ticks) - 1) else f"{t:g}").replace("-", "−")
                       for i, t in enumerate(ticks)])
    cb.ax.tick_params(length=1.4, width=0.4, color=G400, labelsize=FS_NOTE, pad=0.97)
    cb.outline.set_linewidth(0.4)
    cax.set_title("score  (z, shared scale)", fontsize=FS_NOTE, color=SUB, pad=3.8)
    return save_fig(fig, "panel_b", OUT)


# ===================================================================================================== panel c
LEGEND_C_X0 = 0.257024   # figure fraction: where the key row under panel c is anchored (lower left)


def _box(ax, xc, v, hw, face, edge):
    """One box at x = xc: interquartile body, median bar, whiskers at the 5th and 95th percentiles, no outliers."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size < 8:
        return
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    w_lo, w_hi = np.percentile(v, [5, 95])
    ax.plot([xc, xc], [w_lo, w_hi], color=edge, lw=0.45, zorder=3, solid_capstyle="butt")
    ax.add_patch(Rectangle((xc - hw, q1), 2 * hw, q3 - q1, facecolor=face, edgecolor=edge, lw=0.4, zorder=4))
    ax.plot([xc - hw, xc + hw], [med, med], color=edge, lw=0.7, zorder=5, solid_capstyle="butt")


def panel_c(w_mm=68.5):
    """Per CD8 T subtype (ordered by RECAST's AUROC minus ANS's), each method's z-scored program score: a box for
    the subtype's own cells beside a box for all other cells."""
    SDISP = CD8["sdisp"]
    d = pd.read_csv(f"{D}/cd8_cells.csv")
    au = pd.read_csv(f"{D}/cd8_auroc_by_method.csv").pivot(index="state", columns="method", values="auroc")
    dsig = pd.read_csv(f"{D}/cd8_cells_reference_free.csv")
    assert len(dsig) == len(d), "the two cell tables differ in length"
    for st in CD8["states"]:
        d[f"SIGnature::{st}"] = dsig[f"SIGnature::{st}"].values
    gt = d.gt_state.values
    order = (au["RECAST"] - au["ANS"]).sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=(mm(w_mm), mm(ROW_H)))
    ax = fig.add_axes([0.125, 0.115, 0.865, 0.875])
    series = (("RECAST", -0.27), ("ANS", 0.0), ("SIGnature", +0.27))
    DX = HW = 0.052                                      # a pair's two boxes share an edge
    styles = {"RECAST": (RECAST, RECAST_XL, RECAST_D), "ANS": (DE, G200, INK),
              "SIGnature": (REFFREE, REFFREE_XL, REFFREE_D)}          # own cells / other cells / edge
    for i, st in enumerate(order):
        for m, off in series:
            z = d[f"{m}::{st}"].values.astype(float)
            z = (z - z.mean()) / (z.std() + 1e-9)
            face, rest, edge = styles[m]
            _box(ax, i + off - DX, z[gt == st], HW, face, edge)
            _box(ax, i + off + DX, z[gt != st], HW, rest, edge)
    ax.axhline(0, color=G300, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SDISP[s] for s in order], fontsize=FS_BODY)
    ax.set_xlim(-0.50, len(order) - 0.50)
    ax.margins(y=0.10)
    ax.set_ylim(bottom=-1.6)
    ax.set_ylabel("program score  (z)", fontsize=6.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2, labelsize=5.6)
    ax.tick_params(axis="x", length=0, labelsize=FS_BODY, pad=3.40)
    ax.tick_params(axis="y", labelsize=FS_BODY)
    handles = [Patch(fc=RECAST, ec=INK, lw=0.3, label="RECAST"), Patch(fc=DE, ec=INK, lw=0.3, label="ANS"),
               Patch(fc=REFFREE, ec=INK, lw=0.3, label="Reference-free")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(LEGEND_C_X0, -0.0227), ncol=len(handles),
               frameon=False, fontsize=FS_BODY, handletextpad=0.35, columnspacing=1.0, handlelength=1.1)
    return save_fig(fig, "panel_c", OUT)
