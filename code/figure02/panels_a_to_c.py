"""Fig. 2a-c: four CD8 T-cell subtypes, the genes reference-free attribution and RECAST rank first, and marker recovery.

  a  panel_a  the four CD8 subtypes in the SCimilarity embedding (data/cd8_embedding.csv)
  b  panel_b  ranks of shared-lineage and subtype-associated genes, reference-free attribution beside RECAST
              (data/cd8_reference_free_ranks.csv, data/cd8_recast_top50.csv)
  c  panel_c  marker recovery of reference-free attribution and RECAST over the 31 dataset-lineage groups
              (out/tables/marker_recovery_by_group.csv and _tests.csv, written by 01_marker_recovery.py)
The three panels share one 64-mm canvas height: a title line, a column-header line and one plot band at the same
heights.  Panels are saved to OUT (set by 02_draw.py).
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import panel_style                                                            # noqa: E402,F401  (house style)
from method_colours import (apply_style, mm, thin, save_fig, INK, HAIR, G700, G550, G400, G100,   # noqa: E402
                            RECAST, RECAST_D, REF_FREE, BIO, recast_cmap, neutral_cmap, rank_intensity)
import recovery_axes as RA                                                    # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out", "panels")
apply_style()

CD8_STATES = ["CCR7+ CD8", "CXCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"]      # sibling order
SUBTYPE_COLOURS = {"CCR7+ CD8": "#4EBAE8", "CXCR3+ CD8": "#B55968", "CX3CR1+ CD8": "#BDCB46", "CXCR6+ CD8": "#36257F"}
GITAL = dict(style="italic")                                                 # gene symbols in italics

ROW_H_MM = 64.0            # every panel of the row: one canvas height ...
ROW_TITLE_Y = 62.4         # ... one title line (va = top) ...
ROW_HDR_Y = 56.4           # ... one column-header line (va = bottom) ...
ROW_PLOT = (8.5, 55.5)     # ... and one plot band (bottom, top), all in millimetres
ROWA_W, ROWB_W, ROWC_W = 32.6, 107.9, 44.0
COL_LABEL_ANGLE = 35       # degrees: the subtype labels under the heat maps
COL_LABEL_Y = 3.4583       # mm above the canvas bottom: the middle of each label's baseline, under its column


def _tw(fig, txt, fs, **kw):
    """Rendered width of `txt` at `fs` points, in millimetres."""
    t = fig.text(0, 0, txt, fontsize=fs, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def _row_fig(W):
    """A canvas of the row and its mm -> figure-fraction maps."""
    fig = plt.figure(figsize=(mm(W), mm(ROW_H_MM)))
    return fig, (lambda v: v / W), (lambda v: v / ROW_H_MM)


def _rank_rows():
    """Rows and rank intensities of the two heat maps of b.

    Rows: nine shared-lineage genes (four named ones first, then the genes whose worst reference-free rank over the four
    subtypes is best), then each subtype's two highest RECAST genes that are subtype-specific (RECAST rank <= 15 in the
    subtype, absent from every other subtype's RECAST top 50; immunoglobulin, mitochondrial, ribosomal and haemoglobin
    genes left out).  Intensity = 1 - log10(rank) / log10(100); a gene outside RECAST's top 50 is white."""
    df = pd.read_csv(os.path.join(DATA, "cd8_reference_free_ranks.csv"))
    rk = df.pivot(index="gene", columns="state", values="rank").reindex(columns=CD8_STATES).dropna()
    worst = rk.max(axis=1)
    force = [g for g in ["CD3D", "CD52", "PTPRC", "HCST"] if g in rk.index]
    shared = list(dict.fromkeys(force + [g for g in worst.sort_values().index if g not in force]))[:9]
    top = pd.read_csv(os.path.join(DATA, "cd8_recast_top50.csv"))
    FR = {s: dict(zip(top.loc[top.state == s, "gene"], top.loc[top.state == s, "rank"])) for s in CD8_STATES}
    frank = {(s, g): r for s in CD8_STATES for g, r in FR[s].items()}
    amb = re.compile(r"^(IGH|IGK|IGL|MT-|RPS|RPL|HB[AB])")
    state_rows = []
    for st in CD8_STATES:
        others = [c for c in CD8_STATES if c != st]
        picks = []
        for g, r in sorted(FR[st].items(), key=lambda kv: kv[1]):
            if r > 15 or g in shared or amb.match(g) or g not in rk.index:
                continue
            if any(g in FR[o] for o in others):
                continue
            picks.append(g)
            if len(picks) == 2:
                break
        state_rows += [(g, st) for g in picks]
    genes = shared + [g for g, _ in state_rows]
    grp = ["shared"] * len(shared) + ["state"] * len(state_rows)
    fmat = np.array([[frank.get((st, g), 10 ** 6) for st in CD8_STATES] for g in genes], dtype=float)
    return genes, grp, rank_intensity(rk.loc[genes].values, cap=100.0), rank_intensity(fmat, cap=100.0), \
        len(shared), len(state_rows)


def panel_a():
    """The four CD8 sibling subtypes in the embedding, with a 2 x 2 key and the kNN macro-F1 under the cloud."""
    df = pd.read_csv(os.path.join(DATA, "cd8_embedding.csv"))
    order = sorted(CD8_STATES, key=lambda s: -df["state"].eq(s).sum())      # big first, small on top
    W = ROWA_W
    fig, fx, fy = _row_fig(W)
    ax = fig.add_axes([fx(2.0), fy(13.2), fx(W - 4.0), fy(ROW_PLOT[1] - 13.2)])
    for s in order:
        d = df[df["state"] == s]
        ax.scatter(d["umap1"], d["umap2"], s=1.5, c=SUBTYPE_COLOURS[s], alpha=0.55, edgecolors="none",
                   rasterized=False)
    ax.axis("off"); ax.margins(0.02)
    fig.text(fx(W / 2), fy(ROW_TITLE_Y), "CD8 subtypes", fontsize=6.2, color=INK, ha="center", va="top")
    for i, s in enumerate(CD8_STATES):
        x, y = 2.8 + (i % 2) * 15.4, 9.6 - (i // 2) * 3.6
        fig.add_artist(Line2D([fx(x)], [fy(y)], marker="o", linestyle="none", markersize=3.4,
                              markerfacecolor=SUBTYPE_COLOURS[s], markeredgecolor="none", transform=fig.transFigure))
        fig.text(fx(x + 2.2), fy(y), s.replace(" CD8", ""), fontsize=6.0, color=INK, ha="left", va="center")
    fig.text(fx(W / 2), fy(1.2), "kNN macro-F1 = 0.907", fontsize=5.8, color=INK, ha="center", va="bottom")
    return save_fig(fig, "panel_a", OUT)


def panel_b():
    """Gene ranks: the group strip, the reference-free and RECAST heat maps, and the two rank scales stacked."""
    genes, grp, nint, fint, ns, nt = _rank_rows()
    ny = len(genes)
    W = ROWB_W
    fig, fx, fy = _row_fig(W)
    TXT_W, BAR_W, BGAP = 3.0, 2.4, 0.8          # rotated group text | coloured bar | bar -> gene labels
    SCGAP, CBB, CBT, RM = 1.2, 2.2, 8.0, 0.3    # map -> scale column | bar body | its tick labels and name | margin
    GAP = 2.2                                   # between the two maps
    LM = max(_tw(fig, g, 6.0, **GITAL) for g in genes) + 0.8
    CW = (W - (TXT_W + BAR_W + BGAP + LM + SCGAP + CBB + CBT + RM) - GAP) / 2
    assert CW > 14.0, f"only {CW:.1f} mm per heat map"
    Y0, H = fy(ROW_PLOT[0]), fy(ROW_PLOT[1] - ROW_PLOT[0])
    xN = fx(TXT_W + BAR_W + BGAP + LM)
    xF = fx(TXT_W + BAR_W + BGAP + LM + CW + GAP)
    xcb = fx(TXT_W + BAR_W + BGAP + LM + 2 * CW + GAP + SCGAP)
    axL = fig.add_axes([0.0, Y0, fx(TXT_W + BAR_W), H])            # group text + coloured bar
    axN = fig.add_axes([xN, Y0, fx(CW), H])                        # reference-free heat map (carries the genes)
    axF = fig.add_axes([xF, Y0, fx(CW), H])                        # RECAST heat map
    genes_top, grp_top = genes[::-1], grp[::-1]
    xs = [s.replace(" CD8", "") for s in CD8_STATES]

    def draw(ax, x0, mat, cmap, ylabels):
        ax.pcolormesh(mat[::-1], cmap=cmap, vmin=0, vmax=1, edgecolors=HAIR, linewidth=0.5)
        ax.set_xlim(0, 4); ax.set_ylim(0, ny)
        ax.set_xticks(np.arange(4) + 0.5)
        ax.set_xticklabels([""] * 4)
        ax.tick_params(length=0, pad=1.8)
        if ylabels:
            ax.set_yticks(np.arange(ny) + 0.5); ax.set_yticklabels(genes_top, fontsize=6.0, **GITAL)
        else:
            ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for j, s in enumerate(xs):                                 # subtype labels, centred under their columns
            fig.text(x0 + (j + 0.5) * fx(CW) / 4, fy(COL_LABEL_Y), s, fontsize=6.2, rotation=COL_LABEL_ANGLE,
                     rotation_mode="anchor", ha="center", va="baseline")

    draw(axN, xN, nint, neutral_cmap(), True)
    draw(axF, xF, fint, recast_cmap(), False)
    fig.text(xN + fx(CW) / 2, fy(ROW_HDR_Y), "Reference-free", fontsize=6.2, color=REF_FREE, ha="center", va="bottom")
    fig.text(xF + fx(CW) / 2, fy(ROW_HDR_Y), "RECAST", fontsize=6.2, color=RECAST_D, ha="center", va="bottom",
             fontweight="bold")
    fig.text((xN + xF + fx(CW)) / 2, fy(ROW_TITLE_Y), "Gene ranks across CD8 subtypes", fontsize=6.2, color=INK,
             ha="center", va="top")

    # group strip (shared lineage / subtype-associated), left of the gene column
    axL.set_xlim(0, TXT_W + BAR_W); axL.set_ylim(0, ny); axL.axis("off")    # x in millimetres
    gc = {"shared": BIO[4], "state": BIO[6]}
    for i, gk in enumerate(grp_top):
        axL.add_patch(Rectangle((TXT_W, i), BAR_W, 1, color=gc[gk], lw=0))
    axL.text(TXT_W / 2, ny - ns / 2, "Shared lineage", rotation=90, rotation_mode="anchor",
             fontsize=6.0, color=INK, ha="center", va="center")
    axL.text(TXT_W / 2, nt / 2, "Subtype-associated", rotation=90, rotation_mode="anchor",
             fontsize=6.0, color=INK, ha="center", va="center")

    # two rank scales stacked in one column (reference-free above RECAST), each named along its bar; the column is
    # headed "Rank".  Reference-free: rank 100 and beyond -> ">=100"; RECAST: outside its top 50 -> white (">=50").
    LAB = fy(3.4)
    CB_H = (H - 2 * LAB - fy(4.0)) / 2
    caxF = fig.add_axes([xcb, Y0, fx(CBB), CB_H])
    caxN = fig.add_axes([xcb, Y0 + CB_H + LAB + fy(4.0), fx(CBB), CB_H])
    for cax, cmap, lab, lc, lo, fw in [(caxN, neutral_cmap(), "Reference-free", REF_FREE, "≥100", "normal"),
                                       (caxF, recast_cmap(), "RECAST", RECAST_D, "≥50", "bold")]:
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation="vertical")
        cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels([lo, "10", "1"], fontsize=5.8)
        cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=0, pad=1.2)
        cax.set_ylabel(lab, fontsize=5.8, color=lc, fontweight=fw, rotation=90, labelpad=1.5)
        cax.yaxis.set_label_position("right")
    fig.text(xcb + fx((CBB + CBT) / 2), fy(ROW_HDR_Y), "Rank", fontsize=5.8, color=INK, ha="center", va="bottom")
    return save_fig(fig, "panel_b", OUT)


def panel_c():
    """Marker recovery of reference-free attribution and RECAST as two box plots over the 31 groups: box =
    interquartile range, line = median, whiskers to the last group within 1.5 x IQR, diamond = mean (printed beside
    it), the groups as points joined across the two boxes; bracket = group-level sign-flip test."""
    G = pd.read_csv(os.path.join(HERE, "out", "tables", "marker_recovery_by_group.csv"), index_col=0)
    st = pd.read_csv(os.path.join(HERE, "out", "tables", "marker_recovery_tests.csv")).set_index("comparison")
    p = st.loc["RECAST - reference_free", "p_group"]
    top = float(max(G[["RECAST", "reference_free", "no_model", "random_mean"]].max()))   # shared with panel d
    ticks = tick_values(top)
    vals = G[["reference_free", "RECAST"]]
    mN, mF = vals["reference_free"].mean(), vals["RECAST"].mean()
    W = ROWC_W
    P0, P1 = -0.11, 1.16                        # box centres
    fig, fx, fy = _row_fig(W)
    AX0, AXW = 11.0, 30.0
    XB = 10.4
    ax = fig.add_axes([fx(AX0), fy(XB), fx(AXW), fy(ROW_PLOT[1] - XB)])
    BW, CAP = 0.30, 0.17                        # box and whisker-cap half-widths (x data units)
    offs = list(np.linspace(-0.22, 0.22, len(vals)))   # one evenly spread x offset per group

    def tukey(v):
        v = np.sort(np.asarray(v, float))
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        return q1, med, q3, inside.min(), inside.max()

    for i, g in enumerate(vals.index):          # each group's two values, joined
        ax.plot([P0 + offs[i], P1 + offs[i]], [vals.loc[g, "reference_free"], vals.loc[g, "RECAST"]], "-",
                color=G400, lw=0.3, zorder=2.5)
    for x0, col, fill, alpha, edge, medc in [(P0, "reference_free", G100, 1.0, G550, G700),
                                              (P1, "RECAST", RECAST, 0.20, RECAST_D, RECAST_D)]:
        q1, med, q3, lo, hi = tukey(vals[col])
        ax.fill_between([x0 - BW, x0 + BW], q1, q3, facecolor=fill, alpha=alpha, edgecolor="none", zorder=2)
        ax.plot([x0 - BW, x0 + BW, x0 + BW, x0 - BW, x0 - BW], [q1, q1, q3, q3, q1], color=edge, lw=0.7, zorder=3)
        ax.plot([x0, x0], [lo, q1], color=edge, lw=0.7, zorder=3)
        ax.plot([x0, x0], [q3, hi], color=edge, lw=0.7, zorder=3)
        for yy in (lo, hi):
            ax.plot([x0 - CAP, x0 + CAP], [yy, yy], color=edge, lw=0.7, zorder=3)
        ax.plot([x0 - BW, x0 + BW], [med, med], color=medc, lw=1.1, zorder=4)
    for i, g in enumerate(vals.index):
        ax.scatter([P0 + offs[i]], [vals.loc[g, "reference_free"]], s=5, color=REF_FREE, edgecolors="none", zorder=4.5)
        ax.scatter([P1 + offs[i]], [vals.loc[g, "RECAST"]], s=5, color=RECAST, edgecolors="none", zorder=4.5)
    ax.scatter([P0], [mN], s=30, marker="D", color=REF_FREE, edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter([P1], [mF], s=30, marker="D", color=RECAST_D, edgecolors="white", linewidths=0.6, zorder=5)
    ax.text(P0 - 0.42, mN, f"{mN:.3f}", fontsize=6.0, color=INK, ha="right", va="center")
    ax.text(P1 + 0.42, mF, f"{mF:.3f}", fontsize=6.0, color=RECAST_D, ha="left", va="center")
    ax.set_xlim(-1.20, 2.20); ax.set_ylim(0, round(max(ticks[-1] + 0.02, top * 1.20), 4))
    RA.bracket(ax, P0, P1, round(top * 1.06, 4), round(top * 0.025, 4), RA.ptxt(p))
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v:.2f}" for v in ticks], fontsize=5.8)
    ax.set_xticks([P0, P1]); ax.set_xticklabels(["Reference-\nfree", "RECAST"], fontsize=6.2)
    for tl, tc in zip(ax.get_xticklabels(), (REF_FREE, RECAST_D)):
        tl.set_color(tc)
    ax.set_ylabel(RA.LABEL, fontsize=6.0, labelpad=2.0)
    ax.set_xlabel("attribution method", fontsize=6.0, labelpad=3.5)
    thin(ax, spines=("left", "bottom"))
    ax.tick_params(axis="x", length=2, pad=1.8); ax.tick_params(axis="y", length=2, pad=1.5)
    fig.text(fx(AX0 + AXW / 2), fy(ROW_TITLE_Y), "Marker recovery", fontsize=6.2, color=INK, ha="center", va="top")
    RA.ylabel(fig, ax)
    return save_fig(fig, "panel_c", OUT)


def tick_values(top):
    """y ticks of panels c and d: 0 to the first step at or above the highest value, by 0.1 (0.2 beyond 0.45)."""
    step = 0.2 if top > 0.45 else 0.1
    ticks = [round(v, 1) for v in np.arange(0, top + 1e-9, step)]
    return ticks if ticks[-1] >= top - 1e-9 else ticks + [round(ticks[-1] + step, 1)]
