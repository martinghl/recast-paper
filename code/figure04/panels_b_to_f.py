"""Fig. 4b-f: how the choice of reference changes the genes RECAST selects (16 Zhao subtypes, unless stated).

  b  panel_b  TCL1A+ naive B cells: ranks of the top genes with a sibling and with a global reference
  c  panel_c  marker recovery of the curated subtype panel and of a broad-lineage CellMarker panel, per reference
  d  panel_d  marker recovery and one-vs-rest AUROC over the seven references
  e  panel_e  marker recovery per subtype and reference, with the mean row
  f  panel_f  recovery with an embedding-based reference against an annotated sibling reference (Zhao subtypes and
              the subtypes of four further lineages)
Each panel is one cell of the Fig. 4 grid (58.5 x 62 mm); text is 6.2 pt at the 183-mm build width.
Data: data/reference_metrics.csv, data/reference_top_genes.json, data/reference_independent_panels.csv,
data/embedding_reference_recovery.csv.  Panels are saved to OUT (set by 01_draw.py).
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from panel_style import keep   # noqa: E402
from method_colours import (apply_style, mm, thin, save_fig, INK, SUB, HAIR, G700, G550, G400, G300, G200, G100,
                      RECAST, RECAST_D, RECAST_L, RECAST_XL, DE, DE_L, REF_FREE, NULLC, NULL_L, WARN, BIO, recast_cmap)

OUT = os.path.join(HERE, "out", "panels")
apply_style()

# ---- ONE type scale shared by every Fig. 4 panel (rendered 1:1 by the composer) --------------------
T_AXIS  = 6.6      # axis titles
T_ROW   = 6.0      # reference / category row labels
T_TICK  = 5.8      # tick labels, in-cell values, keys, notes
T_DENSE = 5.6      # the 16 subtype row labels of panel e (the only text below 5.8)

# ---- the 7 references: order (best->worst by Recall@20), display label, colour --------------------
# The mismatched control is shown as "unrelated" (the unrelated-lineage reference); its key in the tables is "wrong".
REF_ORDER = ["sibling", "geometry", "nearest", "broad", "global", "random", "wrong"]
REF_ROW   = {"sibling": "sibling", "geometry": "geometry", "nearest": "nearest", "broad": "broad lineage",
             "global": "global atlas", "random": "random", "wrong": "unrelated"}
REF_SHORT = {"sibling": "sibling", "geometry": "geometry", "nearest": "nearest", "broad": "broad",
             "global": "global", "random": "random", "wrong": "unrelated"}
# ONE reference -> colour map, used identically by EVERY Fig. 4 panel that colours by reference
# (b bars, c column headers, e bars, all row labels). Four tones, each with a stated job:
#   INK   = the matched annotated reference (hero, bold label)
#   RECAST = the label-free geometric reference (hero, bold label)
#   DE_L  = every other reference -- ONE uniform neutral, no undefined grey ramp
#   WARN  = the mismatched control (the §6-sanctioned degradation colour)
REF_COL   = {"sibling": RECAST_D, "geometry": RECAST, "nearest": DE_L, "broad": DE_L,
             "global": DE_L, "random": DE_L, "wrong": WARN}
# TEXT colours: the bar fills stay on the §6 ladder, but DE_L / NULLC are far too faint for
# 6 pt type on white (~2.5:1), so labels use ink / the dark RECAST end / the sanctioned WARN.
REF_TXT   = {"sibling": RECAST_D, "geometry": RECAST_D, "nearest": INK, "broad": INK,
             "global": INK, "random": INK, "wrong": WARN}
HERO = ("sibling", "geometry")


def _rect(fw, fh, x, y, w, h):
    """mm box (x,y = lower-left, measured from the figure's lower-left) -> figure-fraction rect."""
    return [x / fw, y / fh, w / fw, h / fh]

def _tw(fig, txt, fs, **kw):
    """Rendered width of `txt` at `fs` points, in millimetres."""
    t = fig.text(0, 0, txt, fontsize=fs, **kw)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def _bare(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2)

def _floor_cmap():
    """recast_cmap() with a VISIBLE light-neutral floor instead of pure white, so a genuine
    Recall = 0 cell can never be read as an empty / missing cell. Anchors are §6 tokens."""
    return LinearSegmentedColormap.from_list("recast_floor", [G100, RECAST_XL, RECAST, RECAST_D])


def _swarm(vals, mm_per_x, mm_per_y, d_mm, half=0.36, pack=0.88):
    """Deterministic beeswarm offsets along the (meaningless) y axis. No value is moved in x and
    no point is dropped -- points that would pile into one uncountable clump are displaced
    vertically instead, so 'dots: the 16 fine states' stays literally true."""
    v = np.asarray(vals, dtype=float)
    dx = d_mm / mm_per_x                      # marker diameter in x data units
    dy = pack * d_mm / mm_per_y               # ... and in y data units
    off = np.zeros(v.size); placed = []
    for i in np.argsort(v, kind="stable"):
        xi = float(v[i])
        near = [yj for xj, yj in placed if abs(xj - xi) < dx]
        cands = [0.0]
        for yj in near: cands += [yj + dy, yj - dy]
        best = half * (1.0 if len(placed) % 2 else -1.0)          # fallback: hug the band edge
        for c in sorted(set(np.round(cands, 6)), key=lambda t: (abs(t), t)):
            if abs(c) > half: continue
            if all(abs(c - yj) >= dy - 1e-6 for yj in near):
                best = c; break
        off[i] = best; placed.append((xi, best))
    return off


# ================================================================= d — recovery and AUROC per reference
def panel_d():
    # benchmarking_sample grammar: aligned metric columns; per-reference MEAN bar + the 16 per-state
    # points jittered ON TOP of the bar (raw structure visible), one row per reference, colour-tagged.
    m = pd.read_csv(f"{HERE}/data/reference_metrics.csv")
    refs = REF_ORDER
    yy = {r: len(refs) - 1 - i for i, r in enumerate(refs)}          # sibling at top
    rng = np.random.default_rng(0)

    FW, FH = 58.5, 62.0
    AX_Y, AX_H = 11.7, 46.3
    AXL_X, AXR_X, AX_W = 15.2, 38.15, 19.15
    YLO, YHI = -0.6, len(refs) - 0.4
    DOT_S, DOT_LW = 2.8, 0.22
    D_MM = (np.sqrt(DOT_S) + DOT_LW) * 25.4 / 72.0                   # marker footprint, mm
    fig = plt.figure(figsize=(mm(FW), mm(FH)))
    axL = fig.add_axes(_rect(FW, FH, AXL_X, AX_Y, AX_W, AX_H))
    axR = fig.add_axes(_rect(FW, FH, AXR_X, AX_Y, AX_W, AX_H))

    # axis ranges cover the full data range
    for ax, col, base, lo, hi, ticks, tlab, chance in [
            (axL, "recall20", 0.0, -0.03, 1.05, [0, 0.5, 1.0], ["0", "0.5", "1.0"], None),
            (axR, "auroc",    0.5,  0.20, 1.03, [0.25, 0.5, 0.75, 1.0],
             ["0.25", "0.5", "0.75", "1.0"], 0.5)]:
        mmx, mmy = AX_W / (hi - lo), AX_H / (YHI - YLO)
        for r in refs:
            sub = m[m.reference == r][col].values                    # 16 per-state values
            mu = float(sub.mean()); c = REF_COL[r]
            # light fill + full-strength edge: the 16 dots must stay visible INSIDE the bar
            ax.barh(yy[r], mu - base, left=base, height=0.62, facecolor=to_rgba(c, 0.55),
                    edgecolor=c, lw=0.8, zorder=2)
            ax.plot([mu, mu], [yy[r] - 0.31, yy[r] + 0.31], color=c, lw=1.5,
                    solid_capstyle="butt", zorder=5)
            # beeswarm, NOT random jitter: at the top of the AUROC axis 14 of the 16 states sit
            # inside 0.10 of each other and a jittered clump is uncountable.
            ax.scatter(sub, yy[r] + _swarm(sub, mmx, mmy, D_MM), s=DOT_S, facecolor=INK,
                       edgecolors="white", linewidths=DOT_LW, alpha=0.95, zorder=6)
        if chance is not None:
            # above the bars: every bar is anchored at 0.5, so a line drawn underneath would survive only in the
            # gaps between the bars
            ax.axvline(chance, color=SUB, lw=0.7, ls=(0, (2.6, 1.8)), zorder=8)
            ax.text(chance, len(refs) - 0.42, "chance", ha="center", va="bottom",
                    fontsize=T_TICK, color=SUB)
        ax.set_xlim(lo, hi); ax.set_ylim(YLO, YHI)
        ax.set_xticks(ticks); ax.set_xticklabels(tlab, fontsize=T_TICK)
        ax.set_yticks([]); _bare(ax); ax.spines["left"].set_visible(False)

    # reference labels on the far left, anchored to the axL rows
    for r in refs:
        axL.text(-0.055, yy[r], REF_ROW[r], transform=axL.get_yaxis_transform(),
                 ha="right", va="center", fontsize=T_ROW, color=REF_TXT[r],
                 fontweight="bold" if r in HERO else "normal")
    axL.text(-0.055, len(refs) - 0.42, "reference", transform=axL.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=T_ROW, color=SUB, fontweight="bold")

    # axis titles, placed in mm so the ink reaches the panel's bottom edge.  What the glyphs mean
    # ("dots: the 16 fine states", "bar base = chance (0.5)") is stated in the Fig. 3 legend, not
    # printed under the axes: it is prose about the drawing, and the caption is where prose belongs.
    for x0, ttl in [(AXL_X, "Marker recovery"), (AXR_X, "One-vs-rest AUROC")]:
        fig.text((x0 + AX_W / 2) / FW, 5.0 / FH, ttl, ha="center", va="baseline",
                 fontsize=T_AXIS, color=INK)
    return save_fig(fig, "panel_d", OUT)


# ================================================================= e — subtypes × references heat map
STATE_ORDER = [("zhao_Mono", "CD14+ Mo"), ("zhao_Mono", "CD16+ Mo"), ("zhao_Mono", "MerTK+ Mo"),
               ("zhao_CD8", "CCR7+ CD8"), ("zhao_CD8", "CRCR3+ CD8"), ("zhao_CD8", "CX3CR1+ CD8"), ("zhao_CD8", "CXCR6+ CD8"),
               ("zhao_CD4", "CCR7+ CD4"), ("zhao_CD4", "CXCR3+ CD4"), ("zhao_CD4", "Treg"),
               ("zhao_B", "Naive B"), ("zhao_B", "TCL1A+ naive B"), ("zhao_B", "Classical mem B"),
               ("zhao_B", "CD11c+ mem B"), ("zhao_B", "ZBTB32+ mem B"), ("zhao_B", "MZB")]
LIN_LABEL = {"zhao_Mono": "Mono", "zhao_CD8": "CD8 T", "zhao_CD4": "CD4 T", "zhao_B": "B"}

def panel_e():
    m = pd.read_csv(f"{HERE}/data/reference_metrics.csv")
    piv = m.pivot_table(index=["lineage", "target"], columns="reference", values="recall20")[REF_ORDER]
    M = np.array([piv.loc[k].values for k in STATE_ORDER])            # 16 x 7
    mean_row = M.mean(0)                                              # = summary per-ref means
    nS, nR = len(STATE_ORDER), len(REF_ORDER)

    FW, FH = 58.5, 62.0
    MX, MW = 22.2, 35.5                 # matrix x / width  -> 5.07 mm column pitch
    MY, MH = 11.7, 39.9                 # 16 rows -> 2.49 mm row pitch; top capped by the 90-deg headers
    RY = 7.6                            # mean-row strip (own axes: no NaN spacer to misread)
    RH = MH / nS
    cmap = _floor_cmap()

    fig = plt.figure(figsize=(mm(FW), mm(FH)))
    ax  = fig.add_axes(_rect(FW, FH, MX, MY, MW, MH))
    axm = fig.add_axes(_rect(FW, FH, MX, RY, MW, RH))
    im  = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axm.imshow(mean_row[None, :], cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for a, nrow in ((ax, nS), (axm, 1)):
        for x in np.arange(-0.5, nR, 1): a.axvline(x, color=G300, lw=0.5, zorder=4)
        for y in np.arange(-0.5, nrow, 1): a.axhline(y, color=G300, lw=0.5, zorder=4)
        a.add_patch(Rectangle((-0.5, -0.5), nR, nrow, fill=False, ec=G400, lw=0.7, zorder=5))
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_xticks([]); a.tick_params(length=0)
        a.set_xlim(-0.5, nR - 0.5); a.set_ylim(nrow - 0.5, -0.5)   # pin: brackets must not autoscale

    # column headers: VERTICAL, centred on their own column. A 45-degree lean puts a 9 mm label
    # over the next 2 columns at this 5 mm pitch -> mis-assignment risk; 90 degrees cannot.
    for j, r in enumerate(REF_ORDER):
        ax.text(j, -0.72, REF_SHORT[r], rotation=90, ha="center", va="bottom",
                fontsize=T_TICK, color=REF_TXT[r], clip_on=False,
                fontweight="bold" if r in HERO else "normal")

    _STATE_FIX = {"CRCR3+ CD8": "CXCR3+ CD8"}                        # upstream typo CRCR3->CXCR3 (display only)
    ax.set_yticks(range(nS))
    ax.set_yticklabels([_STATE_FIX.get(t, t) for (_, t) in STATE_ORDER], fontsize=T_DENSE)
    ax.tick_params(axis="y", length=0, pad=2.2)
    axm.set_yticks([0]); axm.set_yticklabels(["mean (16)"], fontsize=T_TICK, fontweight="bold")
    axm.tick_params(axis="y", length=0, pad=2.2)

    # lineage brackets: placed in AXES-fraction x (get_yaxis_transform) so they can never
    # stretch the data limits into a phantom empty column, and cleared from the row names.
    BR_X, LB_X = -0.513, -0.585         # ~6.3 mm and ~3.7 mm from the panel's left edge
    lin_rows = {}
    for i, (lin, _) in enumerate(STATE_ORDER): lin_rows.setdefault(lin, []).append(i)
    for lin, rows in lin_rows.items():
        y0, y1 = min(rows) - 0.4, max(rows) + 0.4
        ax.plot([BR_X, BR_X], [y0, y1], transform=ax.get_yaxis_transform(),
                color=SUB, lw=0.8, clip_on=False, zorder=6)
        ax.text(LB_X, (y0 + y1) / 2, LIN_LABEL[lin], transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=T_TICK, color=SUB, clip_on=False)
    ax.set_xlim(-0.5, nR - 0.5); ax.set_ylim(nS - 0.5, -0.5)

    # annotate the mean row
    for j in range(nR):
        axm.text(j, 0, f"{mean_row[j]:.2f}", ha="center", va="center",
                 fontsize=T_TICK, color="white" if mean_row[j] > 0.42 else INK,
                 fontweight="bold", zorder=6)

    cax = fig.add_axes(_rect(FW, FH, MX, 4.0, 22.0, 2.3))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=T_TICK, length=2, pad=1.2)
    cb.outline.set_edgecolor(G400); cb.outline.set_linewidth(0.5)   # the ramp's floor is light: box it
    fig.text((MX - 0.8) / FW, 5.15 / FH, "Marker recovery", ha="right", va="center",
             fontsize=T_TICK, color=INK)
    return save_fig(fig, "panel_e", OUT)


# ================================================================= b — top genes, sibling vs global reference
def panel_b():
    g = json.load(open(f"{HERE}/data/reference_top_genes.json"))["zhao_B|TCL1A+ naive B"]
    sib, glob = g["sibling"], g["global"]                            # each a rank-ordered top-20 list
    def strength(gene, lst): return (20 - lst.index(gene)) if gene in lst else np.nan
    union = list(dict.fromkeys(sib[:8] + glob[:8]))                  # top-8 of each, de-duped
    rows = []
    for gene in union:
        ss, gs = strength(gene, sib), strength(gene, glob)
        in_s, in_g = gene in sib[:8], gene in glob[:8]
        cat = "state" if (in_s and not in_g) else "identity" if (in_g and not in_s) else "shared"
        rows.append((gene, ss, gs, cat, (0 if np.isnan(ss) else ss) - (0 if np.isnan(gs) else gs)))
    order = {"state": 0, "shared": 1, "identity": 2}
    rows.sort(key=lambda r: (order[r[3]], -r[4]))
    genes = [r[0] for r in rows]
    Mg = np.array([[r[1], r[2]] for r in rows])                      # genes x [sibling, global]
    # the row label carries the category, so the three colours must survive a 5.8 pt italic glyph.
    # DE (#4D4D4D) against G550 (#7C7C7C) is a 2.0:1 lightness step and reads as one grey; the two
    # exclusive categories therefore take the colours of the two column headers they came from
    # (sibling = RECAST_D blue, global = INK), and "shared" keeps the neutral mid grey.
    gcol = {"state": RECAST_D, "identity": INK, "shared": G550}
    catc = [r[3] for r in rows]
    nG = len(genes)

    FW, FH = 58.5, 62.0
    MX, MW = 12.2, 34.06                 # 2 columns x 11 mm
    # the gene-label key is one line (below), so the matrix reaches down to the plotting boxes of e and f
    MY, MH = 9.6, 46.8
    fig = plt.figure(figsize=(mm(FW), mm(FH)))
    ax = fig.add_axes(_rect(FW, FH, MX, MY, MW, MH))
    cmap = _floor_cmap(); cmap.set_bad("white")
    im = ax.imshow(Mg, cmap=cmap, vmin=0, vmax=20, aspect="auto")
    for x in (-0.5, 0.5, 1.5): ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, nG, 1): ax.axhline(y, color="white", lw=0.8)
    ax.add_patch(Rectangle((-0.5, -0.5), 2, nG, fill=False, ec=G300, lw=0.6, zorder=5))
    # a gene absent from that reference's top-20 is marked, so it cannot be read as "weak rank"
    for i in range(nG):
        for j in range(2):
            if np.isnan(Mg[i, j]):
                ax.plot([j - 0.13, j + 0.13], [i, i], color=G400, lw=0.7, zorder=6, clip_on=False)
    ax.set_xticks([]); ax.set_yticks(range(nG))
    ax.set_yticklabels(genes, fontsize=T_TICK, fontstyle="italic")
    for t, c in zip(ax.get_yticklabels(), catc):                      # colour AND weight: two channels
        t.set_color(gcol[c]); t.set_fontweight("bold" if c != "shared" else "normal")
        if c == "shared":
            keep(t)                    # a category colour (keyed below), not a neutral label grey
    ax.tick_params(axis="y", length=0, pad=2.2)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(nG - 0.5, -0.5)

    for j, (nm_, c, w) in enumerate([("sibling", RECAST_D, "bold"), ("global", INK, "normal")]):
        xc = (MX + MW * (j + 0.5) / 2) / FW
        fig.text(xc, 59.5 / FH, nm_, ha="center", va="baseline", fontsize=T_TICK, color=c, fontweight=w)
        fig.text(xc, 57.0 / FH, "reference", ha="center", va="baseline", fontsize=T_TICK, color=SUB)

    # colour key: kept INSIDE panel d, hugging the matrix, so it cannot read as panel e's legend
    cax = fig.add_axes(_rect(FW, FH, 47.66, 23.0, 2.2, 20.0))
    # the ramp is a rank, so it is ticked with ranks, not with "high"/"low": the cell value is
    # strength = 21 - (rank in that reference's top-20), i.e. strength 20 is rank 1 and 1 is rank 20.
    cb = fig.colorbar(im, cax=cax, ticks=[1, 11, 20])
    cb.ax.set_yticklabels(["20", "10", "1"], fontsize=T_TICK, color=INK)
    cb.ax.tick_params(axis="y", length=1.5, width=0.5, color=G400, pad=1.0)
    cb.outline.set_edgecolor(G400); cb.outline.set_linewidth(0.5)   # the ramp's floor is light: box it
    fig.text(47.66 / FW, 47.6 / FH, "rank within", ha="left", va="baseline", fontsize=T_TICK, color=SUB)
    fig.text(47.66 / FW, 45.1 / FH, "that top-20", ha="left", va="baseline", fontsize=T_TICK, color=SUB)
    # gene-label colour key (the row labels carry the category; no extra column needed).
    # one horizontal line: the three categories are alternatives to each other and read as a series.
    # The words are laid out from their MEASURED widths, so the run cannot overflow the 50 mm page.
    # the "gene labels" lead-in is dropped: four phrases run 4.3 mm past the 50 mm page, and the
    # legend already says the labels are coloured by category, so the words alone carry the key.
    x = 0.8
    for txt, col, wt in [("sibling-prioritized", RECAST_D, "bold"),
                         ("global-prioritized", INK, "bold"), ("shared", G550, "normal")]:
        _kt = fig.text(x / FW, 4.4 / FH, txt, ha="left", va="baseline", fontsize=T_TICK, color=col,
                       fontweight=wt)
        if txt == "shared":
            keep(_kt)                  # the key word carries the same category grey as its genes
        x += _tw(fig, txt, T_TICK, fontweight=wt) + 1.7
    assert x < FW, f"the gene-label key runs {x - FW:.1f} mm past the panel edge"
    # the key sample is the in-cell mark itself (same grey, width and length)
    _L = 0.26 * MW / 2                                             # mm: the mark spans 0.26 of a cell
    fig.add_artist(plt.Line2D([0.8 / FW, (0.8 + _L) / FW], [1.62 / FH] * 2, color=G400, lw=0.7,
                              solid_capstyle="projecting", transform=fig.transFigure))
    fig.text((0.8 + _L + 1.0) / FW, 1.0 / FH, "not in that top-20", ha="left", va="baseline",
             fontsize=T_TICK, color=SUB)
    return save_fig(fig, "panel_b", OUT)


# ================================================================= c — curated subtype panel vs broad-lineage panel
def panel_c():
    """Recall@20 of two programs defined without any RECAST ranking:
      left  = the curated subtype panel of panel b (5-10 genes per subtype);
      right = a CellMarker broad-identity program for the lineage (20 genes, disjoint from the
              lineage's subtype panels).
    Bars are means over the 16 subtypes with s.e.m. whiskers. Solid = subtype panel, hollow = identity program; one REF_COL per reference."""
    s = pd.read_csv(f"{HERE}/data/reference_independent_panels.csv").set_index("reference")
    refs = REF_ORDER
    yy = {r: len(refs) - 1 - i for i, r in enumerate(refs)}

    FW, FH = 58.5, 62.0
    AX_X, AX_W = 11.0, 45.4
    AX_Y, AX_H = 9.6, 46.8
    XLIM = 1.0
    fig = plt.figure(figsize=(mm(FW), mm(FH)))
    ax = fig.add_axes(_rect(FW, FH, AX_X, AX_Y, AX_W, AX_H))
    for r in refs:
        st, idn = float(s.loc[r, "recall_subtype"]), float(s.loc[r, "recall_identity"])
        se_s, se_i = float(s.loc[r, "sem_subtype"]), float(s.loc[r, "sem_identity"])
        ax.barh(yy[r], -st, height=0.62, color=REF_COL[r], edgecolor="white", lw=0.5, zorder=2)
        ax.barh(yy[r], idn, height=0.62, facecolor="white", edgecolor=REF_TXT[r], lw=0.8, zorder=2)
        ax.plot([-st - se_s, -st + se_s], [yy[r]] * 2, color=INK, lw=0.6, solid_capstyle="butt", zorder=3)
        ax.plot([idn - se_i, idn + se_i], [yy[r]] * 2, color=INK, lw=0.6, solid_capstyle="butt", zorder=3)
        ax.text(-st - se_s - 0.03, yy[r], f"{st:.2f}", ha="right", va="center", fontsize=T_TICK, color=INK)
        ax.text(idn + se_i + 0.03, yy[r], f"{idn:.2f}", ha="left", va="center", fontsize=T_TICK, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-XLIM, XLIM); ax.set_ylim(-0.6, len(refs) - 0.4)
    ax.set_yticks([yy[r] for r in refs])
    ax.set_yticklabels([REF_SHORT[r] for r in refs], fontsize=T_ROW)
    for t, r in zip(ax.get_yticklabels(), refs):
        t.set_color(REF_TXT[r])
        if r in HERO: t.set_fontweight("bold")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["1.0", "0.5", "0", "0.5", "1.0"], fontsize=T_TICK)
    _bare(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, pad=2.4)
    # two-line half headers: keeps each line ~16 mm so the two never collide at the centre
    for xc_ax, l1, l2 in [(0.25, "subtype panel", "curated, as in b"),
                          (0.75, "broad-lineage panel", "CellMarker, 20 genes")]:
        xc = (AX_X + AX_W * xc_ax) / FW
        fig.text(xc, 57.0 / FH, l1, ha="center", va="baseline", fontsize=T_TICK, color=INK, fontweight="bold")
    fig.text((AX_X + AX_W / 2) / FW, 2.8 / FH, "Marker recovery",
             ha="center", va="baseline", fontsize=T_AXIS, color=INK)
    return save_fig(fig, "panel_c", OUT)


# ================================================================= f — embedding reference vs annotated reference
def panel_f():
    m = pd.read_csv(f"{HERE}/data/reference_metrics.csv")
    pm = m.pivot_table(index=["lineage", "target"], columns="reference", values="recall20")
    x16, y16 = pm["sibling"].values, pm["geometry"].values           # 16 zhao states
    lf = pd.read_csv(f"{HERE}/data/embedding_reference_recovery.csv")
    pl = lf.pivot_table(index="target", columns="rule", values="recall20")
    x14, y14 = pl["anno_sib"].values, pl["near_auto"].values          # 14 deployment states

    FW, FH = 58.5, 62.0
    AX_X, AX_Y, AX_S = 11.0, 11.7, 46.3      # square box -> aspect "equal" needs no extra slack
    fig = plt.figure(figsize=(mm(FW), mm(FH)))
    ax = fig.add_axes(_rect(FW, FH, AX_X, AX_Y, AX_S, AX_S))
    ax.plot([0, 1], [0, 1], color=G400, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.scatter(x16, y16, s=24, facecolor=RECAST, edgecolor="white", linewidths=0.4, alpha=0.9,
               zorder=3, label="Zhao subtypes (16)")
    ax.scatter(x14, y14, s=28, facecolor="none", edgecolor=RECAST_D, linewidths=1.0,
               zorder=4, label="four-lineage subtypes (14)")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=T_TICK)
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=T_TICK)
    _bare(ax)
    # SPLIT KEY, direct-labelled into the two off-diagonal regions (house style prefers direct
    # labels to a boxed legend). Both slots were checked against every plotted point: for
    # y in [0.68,0.76] no point has x < 0.62, and for y in [0.18,0.26] no point has x > 0.30.
    # the upper key at y = 0.865 (no point has y in [0.80, 0.93] with x < 0.85)
    Y_DEP, X_ZH = 0.865, 0.430
    ax.scatter([0.055], [Y_DEP], s=28, facecolor="none", edgecolor=RECAST_D, linewidths=1.0, zorder=5)
    t_dep = ax.text(0.105, Y_DEP, "four-lineage subtypes (14)", ha="left", va="center",
                    fontsize=T_TICK, color=SUB)          # SUB = the figure's "this is a key, not data" tone
    ax.scatter([X_ZH], [0.22], s=24, facecolor=RECAST, edgecolor="white", linewidths=0.4, zorder=5)
    t_zh = ax.text(X_ZH + 0.045, 0.22, "Zhao subtypes (16)", ha="left", va="center",
                   fontsize=T_TICK, color=SUB)
    # measured: each key text keeps >= 0.4 mm from every data marker and from the y = x line, and ends in the box
    fig.canvas.draw(); _r = fig.canvas.get_renderer()
    _px_mm = fig.dpi / 25.4
    _pts = ax.transData.transform(np.column_stack([np.r_[x16, x14], np.r_[y16, y14]]))
    _rad = np.r_[np.full(len(x16), np.sqrt(24)), np.full(len(x14), np.sqrt(28) + 1.0)] / 2 * fig.dpi / 72
    for _t in (t_dep, t_zh):
        _bb = _t.get_window_extent(_r)
        _dx = np.maximum(np.maximum(_bb.x0 - _pts[:, 0], _pts[:, 0] - _bb.x1), 0)
        _dy = np.maximum(np.maximum(_bb.y0 - _pts[:, 1], _pts[:, 1] - _bb.y1), 0)
        _gap = (np.hypot(_dx, _dy) - _rad).min() / _px_mm
        _d0 = ax.transData.inverted().transform([[_bb.x1, _bb.y1], [_bb.x1, _bb.y0]])
        _diag = min(abs(xx - yy) for xx, yy in _d0) * AX_S / 1.08      # distance to y = x along x, in mm
        _end = (ax.get_window_extent(_r).x1 - _bb.x1) / _px_mm
        assert _gap >= 0.4 and _diag >= 0.4 and _end >= 0.0
    fig.text((AX_X + AX_S / 2) / FW, 5.0 / FH, "Annotation reference — marker recovery",
             ha="center", va="baseline", fontsize=T_AXIS, color=INK)
    fig.text(2.2 / FW, (AX_Y + AX_S / 2) / FH, "Embedding reference — marker recovery",
             ha="center", va="center", rotation=90, fontsize=T_AXIS, color=INK)
    return save_fig(fig, "panel_f", OUT)
