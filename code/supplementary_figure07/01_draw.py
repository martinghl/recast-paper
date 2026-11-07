"""Supplementary Fig. 7: gene programs along the Jerber iPSC-to-midbrain differentiation.

82,166 cells of 12 annotated types, each compared with all remaining cells.
  a  UMAP of the 20,000-cell subsample provided with the data, coloured by annotated type, collection day and
     diffusion pseudotime; rho = 0.599 is over all 82,166 cells (the 20,000 drawn give 0.593)
  b  eta^2 per cell-type panel (points) and mean over the 12 panels (bars): 25 equal-count pseudotime bins (left) and
     collection days 11 / 30 / 52 (right)
  c  the five dopaminergic-neuron genes along the 25 pseudotime bins, with the profile of 200 random genes as the
     bottom row (same y scale; shaded above zero), and their RECAST and Wilcoxon ranks
  d  RECAST against Wilcoxon rank over all 28,231 genes: the five genes of c, the equal-rank diagonal and the first
     20 RECAST genes
The Wilcoxon collection-day values are not drawn; they are in Supplementary Data, sheet 5.

Run     python 01_draw.py
Output  out/S7_developmental_patterns.{pdf,svg,png}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S                    # noqa: E402
import supplementary_common as K                   # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.path import Path          # noqa: E402
from matplotlib.text import Text          # noqa: E402
from scipy.stats import spearmanr         # noqa: E402

from method_colours import G550, G200, INK, BIO, RECAST, RECAST_D, RECAST_L, RECAST_XL, NULLC, NULL_L   # noqa: E402
S.setup()

STEM = "S7_developmental_patterns"
W = 180.0


def _mix(c1, c2, f):
    """The colour a fraction f of the way from c1 to c2."""
    a = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3))


# cell types: the muted biological colours without the slate, which reads as RECAST blue; the seven hues go to the
# seven largest types, darker variants to the five smallest
_BIO_NB = [c for c in BIO if c != "#6E8CA0"]
CT12 = _BIO_NB + [_mix(c, INK, 0.36) for c in _BIO_NB[:5]]
# collection day and pseudotime: one mauve ladder, away from the method colours
_TIME = "#B58BA0"
DAY3 = [_mix(_TIME, "#FFFFFF", 0.35), _TIME, _mix(_TIME, INK, 0.45)]   # D11 -> D30 -> D52
PT_CMAP = LinearSegmentedColormap.from_list("bio_time", [_mix(_TIME, "#FFFFFF", 0.80), _mix(_TIME, "#FFFFFF", 0.38),
                                                         _TIME, _mix(_TIME, INK, 0.52)])

PTS = pd.read_csv(f"{HERE}/data/trajectory_pseudotime.csv")
ETA = pd.read_csv(f"{HERE}/data/trajectory_eta_squared.csv")
DAY = pd.read_csv(f"{HERE}/data/trajectory_collection_day_null.csv")
PRO = pd.read_csv(f"{HERE}/data/development_gene_profiles.csv")
RNK = pd.read_csv(f"{HERE}/data/development_top_gene_ranks.csv").set_index("gene")
FUL = pd.read_csv(f"{HERE}/data/development_gene_ranks.csv")

TYPE_NAME = {"Astro": "Astrocyte-like cells", "DA": "Dopaminergic neurons", "Epen1": "Ependymal-like 1",
             "Epen2": "Ependymal-like 2", "FPP": "Floor-plate progenitors", "NB": "Neuroblasts",
             "P_FPP": "Proliferating floor-plate progenitors", "P_Sert": "Proliferating serotonergic-like neurons",
             "Sert": "Serotonergic-like neurons", "U_Neur1": "Unknown neurons 1", "U_Neur2": "Unknown neurons 2",
             "U_Neur3": "Unknown neurons 3"}
assert len(PTS) == 20000 and set(PTS.celltype) == set(TYPE_NAME) and abs(PTS.dpt.max() - 1.0) < 1e-9
assert set(PTS.time) == {"D11", "D30", "D52"}
assert ETA.groupby("panel").size().eq(12).all() and set(ETA.state) == set(TYPE_NAME) and len(DAY) == 12
NU = ETA[ETA.panel == "RECAST_unique"].n_genes
assert NU.min() == 2 and NU.max() == 20
assert (ETA[ETA.panel == "RECAST"].n_genes == 20).all()
assert np.allclose(ETA[ETA.panel == "random"].mean_eta2.values, ETA[ETA.panel == "RECAST"].null_eta2.values)
GENES = [c for c in PRO.columns if c not in ("bin", "random")]
assert len(PRO) == 25 and len(GENES) == 5 and set(GENES) <= set(RNK.index) and len(FUL) == 28231
RHO_ALL = 0.599                                       # Spearman(pseudotime, collection day), all 82,166 cells
# the drawn collection-day values are the RECAST rows of mean_eta2_time (Supplementary Data, sheet 5)
piv = ETA.pivot(index="state", columns="panel", values="mean_eta2_time").loc[DAY.state]
assert np.allclose(piv.RECAST.values, DAY.RECAST_day.values, atol=1e-12)
assert np.allclose(ETA[ETA.panel == "RECAST_unique"].set_index("state").loc[DAY.state].mean_eta2_time.values,
                   DAY.RECASTuniq_day.values, atol=1e-12)

MINUS = "−"


def main(H):
    P = S.Page(W, H)
    fig = P.fig
    k = K.px_per_mm(fig)

    def bare(ax):
        S.frame(ax)

    # ================================================================ a: three colourings of one source UMAP
    AY, AW = 6.2, 37.0
    X_CT, X_LEG, X_DAY, X_PT, CB_X = 1.5, 40.0, 92.0, 131.0, 169.6
    by_size = PTS.celltype.value_counts().index.tolist()
    ccol = {c: CT12[i % len(CT12)] for i, c in enumerate(by_size)}
    days = ["D11", "D30", "D52"]
    dcol = {d: DAY3[i] for i, d in enumerate(days)}
    cx, cy = 0.5 * (PTS.ux.min() + PTS.ux.max()), 0.5 * (PTS.uy.min() + PTS.uy.max())
    hx = hy = 0.5 * 1.05 * max(PTS.ux.max() - PTS.ux.min(), PTS.uy.max() - PTS.uy.min())

    def umap(x0, colors, order=None, alpha=0.65, title=""):
        ax = P.ax(x0, AY, AW, AW)
        idx = np.arange(len(PTS)) if order is None else order
        ax.scatter(PTS.ux.values[idx], PTS.uy.values[idx], s=1.0, c=[colors[i] for i in idx], linewidths=0,
                   alpha=alpha, rasterized=True)
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.axis("off")
        P.text(x0 + AW / 2, AY - 0.9, title, ha="center", va="bottom")
        return ax

    axc = umap(X_CT, PTS.celltype.map(ccol).to_numpy(), title="Annotated type")
    # the two orienting call-outs: at 8 pt only the upper-left corner of the map is free of cells (checked below), so
    # the floor-plate label stays inside the map and the dopaminergic label moves just outside its
    # upper-right corner, above the type key, with its leader running into the map
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cell_px = axc.transData.transform(np.c_[PTS.ux.values, PTS.uy.values])
    callouts = {}
    spots = {"FPP": (0.015, 0.985, "left", "top"), "DA": (1.035, 1.0, "left", "top")}
    # >= 2.5 mm of leader left after matplotlib's 2 pt pad around the label, shrinkA (2 pt) and shrinkB (1.5 pt)
    min_lead = 2.5 * k + (2 + 2 + 1.5) * fig.dpi / 72
    for st, txt in [("FPP", "Floor-plate\nprogenitors"), ("DA", "Dopaminergic\nneurons")]:
        c = PTS[PTS.celltype == st]
        hh, xe, ye = np.histogram2d(c.ux.values, c.uy.values, bins=24)
        i, j = np.unravel_index(np.argmax(hh), hh.shape)
        peak = np.array([0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])])
        fx, fy, ha, va = spots[st]
        lab = np.array([cx - hx + fx * 2 * hx, cy - hy + fy * 2 * hy])
        d = (lab - peak) / np.linalg.norm(lab - peak)
        edge = peak + d * float(np.quantile((np.c_[c.ux.values, c.uy.values] - peak) @ d, 0.97))
        t = axc.annotate(txt, edge, xytext=(fx, fy), textcoords="axes fraction", fontsize=S.FS, ha=ha, va=va,
                         color=S.BLACK, fontweight="bold", linespacing=1.05, zorder=6, annotation_clip=False,
                         arrowprops=dict(arrowstyle="-", lw=S.LW, color=S.BLACK, shrinkA=2, shrinkB=1.5))
        fig.canvas.draw()
        tb = Text.get_window_extent(t, rend)
        # the leader ends at the cluster's 97th-percentile edge; a label that sits closer to its cluster
        # than a visible leader needs (the floor-plate label at 8 pt, about 1 mm) is a direct label without one, since
        # the leader would shrink to a 1 mm stub
        ex, ey = axc.transData.transform(edge)
        if np.hypot(max(tb.x0 - ex, 0, ex - tb.x1), max(tb.y0 - ey, 0, ey - tb.y1)) < min_lead:
            t.arrow_patch.set_visible(False)
        bb = tb.padded(0.5 * k)                                       # the label alone, not its leader
        n_under = int(((cell_px[:, 0] > bb.x0) & (cell_px[:, 0] < bb.x1) & (cell_px[:, 1] > bb.y0)
                       & (cell_px[:, 1] < bb.y1)).sum())
        assert n_under == 0, f"{st} call-out covers {n_under} cells"
        callouts[st] = bb
    assert not callouts["FPP"].overlaps(callouts["DA"])
    # type key: full names, alphabetical, beside the map
    KEY_TOP = P.H - callouts["DA"].y0 / k + 1.2                 # first row's top, below the dopaminergic label
    pitch = 3.35
    KH = 12 * pitch
    kax = P.ax(X_LEG, KEY_TOP, 50.0, KH)
    kax.axis("off")
    kax.set_xlim(0, 50.0)
    kax.set_ylim(0, KH)
    for i, c in enumerate(sorted(TYPE_NAME, key=lambda s: TYPE_NAME[s])):
        yy = KH - (0.5 + i) * pitch
        kax.scatter(1.0, yy, s=10, color=ccol[c], linewidths=0)
        kax.text(2.6, yy, TYPE_NAME[c], ha="left", va="center", fontsize=S.FS)
    fig.canvas.draw()
    kx1 = max(t.get_window_extent(rend).x1 for t in kax.texts) / k
    assert kx1 < X_DAY - 1.0, f"type key runs into the day map ({kx1:.1f} mm)"

    umap(X_DAY, PTS.time.map(dcol).to_numpy(), alpha=0.7, title="Collection day")
    leg_w = 2 * 12.0 + 11.6
    for i, dd in enumerate(days):
        x = X_DAY + (AW - leg_w) / 2 + i * 12.0
        dax = P.ax(x, AY + AW + 1.6, 2.0, 2.0)
        dax.axis("off")
        dax.set_xlim(0, 1)
        dax.set_ylim(0, 1)
        dax.scatter([0.5], [0.5], s=10, color=dcol[dd], linewidths=0)
        P.text(x + 2.7, AY + AW + 2.6, f"Day {dd[1:]}", ha="left", va="center")
    norm = Normalize(0.0, 1.0)
    umap(X_PT, [PT_CMAP(norm(v)) for v in PTS.dpt.values], order=np.argsort(PTS.dpt.values), alpha=0.8,
         title="Diffusion pseudotime")
    cax = P.ax(CB_X, AY + 4.0, 2.2, AW - 8.0)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=PT_CMAP), cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0", "0.5", "1"])
    cb.ax.tick_params(labelsize=S.FS, length=S.TICK, width=S.LW, pad=S.TICK_PAD)
    cb.outline.set_linewidth(S.LW)
    P.text(X_PT + AW / 2, AY + AW + 1.4, f"Spearman ρ with day = {RHO_ALL:.3f}\n(all cells)", ha="center",
           va="top", linespacing=1.15)
    P.letter(0.0, 1.0, "a")

    # ================================================================ b: eta^2, two tracks side by side
    fig.canvas.draw()
    a_bottom = P.H - min([t.get_window_extent(rend).y0 for t in fig.texts] +
                         [t.get_window_extent(rend).y0 for t in kax.texts]) / k
    BY, BH = a_bottom + 10.5, 31.0

    def strip(ax, groups, gap_after=()):
        rng = np.random.default_rng(0)
        xs, x = [], 0.0
        for kk in range(len(groups)):
            xs.append(x)
            x += 1.0 + (0.35 if kk in gap_after else 0.0)
        for x, (lab, v, col, isf) in zip(xs, groups):
            jx = x + (rng.random(len(v)) - 0.5) * 0.34
            ax.scatter(jx, v, s=8, color=col, alpha=0.7, edgecolors="white", linewidths=0.3, zorder=3)
            mu = float(np.mean(v))
            ax.plot([x - 0.30, x + 0.30], [mu, mu], color=col if col != NULLC else G550, lw=2.0,
                    solid_capstyle="round", zorder=4)
            ax.annotate(f"{mu:.3f}", xy=(x, max(mu, float(np.max(v)))), xytext=(0, 3.6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=S.FS, color=RECAST_D if isf else S.BLACK,
                        fontweight="bold" if isf else "normal", zorder=6)
        ax.set_xticks(xs)
        ax.set_xticklabels([g[0] for g in groups], fontsize=S.FS, linespacing=1.05)
        for t, g in zip(ax.get_xticklabels(), groups):
            if g[3]:
                t.set_color(RECAST_D)
                t.set_fontweight("bold")
        ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
        ax.tick_params(axis="x", length=0, pad=S.TICK_PAD)

    EXP = "Expression-\nmatched\nexpectation"
    col = lambda p: ETA[ETA.panel == p].mean_eta2.values      # noqa: E731
    nul = lambda p: ETA[ETA.panel == p].null_eta2.values      # noqa: E731
    axT = P.ax(16.0, BY, 68.0, BH)
    strip(axT, [("RECAST", col("RECAST"), RECAST, True), (EXP, nul("RECAST"), NULLC, False),
                ("RECAST-\nunique", col("RECAST_unique"), RECAST_L, False),
                (EXP, nul("RECAST_unique"), NULLC, False)], gap_after=(1,))
    axT.set_ylim(0, 0.70)
    axT.set_yticks([0, 0.2, 0.4, 0.6])
    axT.set_yticklabels(["0", "0.2", "0.4", "0.6"])
    axT.set_ylabel("η² (25 pseudotime bins)", fontsize=S.FS, labelpad=S.LABEL_PAD)
    axT.set_title("Pseudotime bins", fontsize=S.FS, pad=3)
    bare(axT)
    axT.tick_params(axis="x", length=0)
    axB = P.ax(108.0, BY, 68.0, BH)
    strip(axB, [("RECAST", DAY.RECAST_day.values, RECAST, True), (EXP, DAY.RECAST_day_null.values, NULLC, False),
                ("RECAST-\nunique", DAY.RECASTuniq_day.values, RECAST_L, False),
                (EXP, DAY.RECASTuniq_day_null.values, NULLC, False)], gap_after=(1,))
    axB.set_ylim(0, 0.40)
    axB.set_yticks([0, 0.1, 0.2, 0.3])
    axB.set_yticklabels(["0", "0.1", "0.2", "0.3"])
    axB.set_ylabel("η² (collection day)", fontsize=S.FS, labelpad=S.LABEL_PAD)
    axB.set_title("Collection days 11, 30 and 52", fontsize=S.FS, pad=3)
    bare(axB)
    axB.tick_params(axis="x", length=0)
    assert max(col("RECAST").max(), col("RECAST_unique").max(), col("random").max()) < 0.62
    assert max(DAY.drop(columns="state").max()) < 0.34
    P.letter(0.0, BY - 7.0, "b")

    # ================================================================ c: five genes, one random-gene row
    fig.canvas.draw()
    b_bottom = min(t.get_window_extent(rend).y0 for a_ in (axT, axB) for t in a_.get_xticklabels())
    b_bottom = P.H - b_bottom / k
    CY = b_bottom + 12.0
    CX0, SW_, RP, RH = 12.5, 58.0, 7.6, 6.6
    TX_G, TX_R, TX_W = 74.0, 93.5, 106.0                       # gene name (left), RECAST / Wilcoxon (centres)
    nbin = len(PRO)
    xb = np.arange(1, nbin + 1)
    rnd = PRO["random"].to_numpy()
    M = {g: PRO[g].to_numpy() for g in GENES}
    onset = {g: (np.argmax(M[g] > 0) if (M[g] > 0).any() else nbin) for g in GENES}
    order = sorted(GENES, key=lambda g: onset[g])
    ymax = max(np.nanmax([M[g] for g in GENES]), np.nanmax(rnd)) * 1.12
    ymin = min(np.nanmin([M[g] for g in GENES]), np.nanmin(rnd)) - 0.15
    rows = order + ["random"]
    c_axes = []
    for r, g in enumerate(rows):
        ax = P.ax(CX0, CY + r * RP, SW_, RH)
        y = rnd if g == "random" else M[g]
        ax.axhline(0, color=G200, lw=0.6, zorder=1)
        if g == "random":
            ax.plot(xb, y, "-", color=NULLC, lw=1.0, zorder=2)
            ax.fill_between(xb, np.minimum(y, 0), y, color=NULL_L, alpha=0.5, lw=0, zorder=1)
        else:
            ax.plot(xb, y, "-", color=RECAST, lw=1.3, zorder=4)
            ax.fill_between(xb, 0, y, where=(y > 0), color=RECAST_XL, alpha=0.7, lw=0, zorder=2)
        ax.set_xlim(1, nbin)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([0])
        ax.set_yticklabels(["0"])
        last = r == len(rows) - 1
        S.frame(ax, spines=("left", "bottom") if last else ("left",))
        if last:
            ax.set_xticks([1, nbin])
            ax.set_xticklabels(["1", str(nbin)])
        else:
            ax.set_xticks([])
        yc = CY + r * RP + RH / 2
        if g == "random":
            P.text(TX_G, yc, "Random expressed\ngenes (200)", ha="left", va="center", linespacing=1.1)
        else:
            P.text(TX_G, yc, g, ha="left", va="center", fontstyle="italic", fontweight="bold")
            P.text(TX_R, yc, f"#{int(RNK.loc[g, 'recast_rank'])}", ha="center", va="center", color=RECAST_D)
            P.text(TX_W, yc, f"#{int(RNK.loc[g, 'de_rank'])}", ha="center", va="center")
        c_axes.append(ax)
    P.text(TX_R, CY - 1.0, "RECAST\nrank", ha="center", va="bottom", linespacing=1.1, color=RECAST_D)
    P.text(TX_W, CY - 1.0, "Wilcoxon\nrank", ha="center", va="bottom", linespacing=1.1)
    c_h = (len(rows) - 1) * RP + RH
    P.text(CX0 + SW_ / 2, CY + c_h + 5.4, "Pseudotime bin (early → late)", ha="center", va="top")
    P.text(3.2, CY + c_h / 2, "Standardized expression (z)", rotation=90, ha="center", va="center")
    P.letter(0.0, CY - 8.6, "c")

    # ================================================================ d: rank vs rank
    N = len(FUL)
    TOPK = 20
    DX, DW = 128.5, 47.5
    DH = DW * 0.95
    ax = P.ax(DX, CY, DW, DH)
    HI = N * 2.2
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, HI)
    ax.set_ylim(0.7, HI)
    ax.scatter(FUL.recast_rank, FUL.de_rank, s=0.9, color=G200, alpha=0.30, linewidths=0, rasterized=True, zorder=1)
    ax.plot([1, N], [1, N], color=S.BLACK, lw=S.LW, ls=S.DASH, zorder=2)
    ax.axvline(TOPK, color=RECAST, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
    SS = FUL[FUL.gene.isin(GENES)]
    assert len(SS) == 5 and (SS.recast_rank <= TOPK).all()
    ax.scatter(SS.recast_rank, SS.de_rank, s=16, color=RECAST, edgecolors="white", linewidths=0.5, zorder=5)
    ticks, tlab = [1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tlab)
    ax.minorticks_off()
    ax.set_xlabel("RECAST rank", fontsize=S.FS, labelpad=S.LABEL_PAD)
    ax.set_ylabel("Wilcoxon rank", fontsize=S.FS, labelpad=S.LABEL_PAD)
    bare(ax)
    ax.minorticks_off()
    # the line's label goes to the first candidate spot with no gene under it
    fig.canvas.draw()
    pts = ax.transData.transform(np.c_[FUL.recast_rank, FUL.de_rank])
    lab = ax.text(0, 0, "First 20\nRECAST\ngenes", fontsize=S.FS, color=RECAST_D, linespacing=1.1, zorder=4)
    abb = ax.get_window_extent(rend)
    for (lx, ly, ha, va) in ((TOPK * 1.18, 0.95, "left", "bottom"), (TOPK * 1.18, HI / 1.4, "left", "top"),
                             (TOPK / 1.18, HI / 1.4, "right", "top")):
        lab.set_position((lx, ly))
        lab.set_ha(ha)
        lab.set_va(va)
        bb = lab.get_window_extent(rend).padded(1.0)
        n_under = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
        if n_under == 0:
            break
    assert n_under == 0, f"no free spot for the top-20 label ({n_under} genes under it)"
    assert abb.x0 < bb.x0 and bb.x1 < abb.x1 and abb.y0 < bb.y0 and bb.y1 < abb.y1, "top-20 label crosses an axis line"
    ax.set_title("Dopaminergic-neuron comparison", fontsize=S.FS, pad=3)
    fig.canvas.draw()
    d_xl = P.H - ax.xaxis.label.get_window_extent(rend).y0 / k
    yk = d_xl + 1.8
    P.letter(DX - 14.5, CY - 8.6, "d")

    # ================================================================ geometry checks + page height
    fig.canvas.draw()
    bottoms = [t.get_window_extent(rend).y0 for t in fig.texts] + \
              [t.get_window_extent(rend).y0 for a_ in fig.axes for t in a_.get_xticklabels() if t.get_text()]
    needed = P.H - min(bottoms) / k + 1.0
    c_right = max(t.get_window_extent(rend).x1 for t in fig.texts
                  if CY - 3 < P.H - t.get_window_extent(rend).y1 / k < CY + c_h) / k
    d_left = min(ax.yaxis.label.get_window_extent(rend).x0, *[t.get_window_extent(rend).x0
                                                               for t in ax.get_yticklabels()]) / k
    assert c_right < d_left - 3.0, (c_right, d_left)
    cb_right = max(t.get_window_extent(rend).x1 for t in cb.ax.get_yticklabels()) / k
    assert cb_right < W - 0.8, cb_right
    checks = dict(
        a=dict(cells_drawn=len(PTS), types=PTS.celltype.nunique(), rho_all_cells_logged=RHO_ALL,
               rho_drawn_20000=round(float(spearmanr(PTS.dpt, PTS.time.str[1:].astype(int)).correlation), 4)),
        b_pseudotime_means={p: round(float(col(p).mean()), 3) for p in ("RECAST", "RECAST_unique")},
        b_pseudotime_expectations={p: round(float(nul(p).mean()), 3) for p in ("RECAST", "RECAST_unique")},
        b_day_means=DAY.drop(columns="state").mean().round(3).to_dict(),
        b_day_exceeds_expectation=dict(recast=int((DAY.RECAST_day > DAY.RECAST_day_null).sum()),
                                       recast_unique=int((DAY.RECASTuniq_day > DAY.RECASTuniq_day_null).sum())),
        c_onset_order=order, c_ranks={g: [int(RNK.loc[g, "recast_rank"]), int(RNK.loc[g, "de_rank"])] for g in order},
        d_genes=N, d_label_spots=dict(top20=[lx, ly]),
        geometry=dict(c_right=round(c_right, 1), d_left=round(d_left, 1), colorbar_right=round(cb_right, 1)))
    return P, needed, checks, dict(order=order, ymin=ymin, ymax=ymax)


if __name__ == "__main__":
    P, needed, _, _ = main(200.0)                         # first pass: measure the height the content needs
    plt.close(P.fig)
    S_H = round(needed + 0.4, 1)
    P, needed2, checks, cv = main(S_H)
    assert abs(needed2 - needed) < 0.05
    checks.update(page_mm=(W, S_H))
    P.save(f"{HERE}/out", STEM)
    print(STEM, checks)
