"""Fig. 5a: enrichment of 18 curated biological programs in the top 30 genes of RECAST and of COSG, per subtype.

Rows are the programs (data/programs.csv), columns the subtypes of the 74-population cohort (common/subtype_columns.py)
grouped by compartment under a two-tier header (tissue over lineage).  Each mark is coloured by the method whose top 30
is enriched (FDR < 0.05; data/program_enrichment.csv): a RECAST-blue disc, an orange COSG disc with a hairline, or one
disc split down the middle when both are; the colour intensity is the fraction of the subtype's datasets in which the
top 30 is enriched.  Dashed outlines mark the expected program-subtype pairs (data/expected_pairs.csv, "E"); touching
expected cells share one outline.  Subtypes with neither an enrichment nor an expected program are not drawn.
The panel is drawn at 180 x 80 mm with text at 6.5 pt (tissue 8 pt bold, lineage 7 pt).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from panel_style import keep, BLACK                                    # noqa: E402
from method_colours import apply_style, mm, save_fig, INK, RECAST, RECAST_L, RECAST_XL    # noqa: E402
from subtype_columns import (POPULATIONS, COMPARTMENTS, COLS, TISSUES, LINEAGE_LABEL,      # noqa: E402
                             compartment_of, column_label)

apply_style()

PAGE_W, PAGE_H = 180.0, 80.0
FS, FS_TISSUE, FS_LINEAGE = 6.5, 8.0, 7.0          # labels / tissue spanners (bold) / lineage labels, pt
FRAME_LW = 0.4                                     # pt: compartment frames and spanner rules
BLOCK_GAP = 1.2                                    # mm between compartment blocks
ROW_PITCH = 2.75                                   # mm per program row
INK_X0, INK_X1 = 0.4, 179.6                        # the panel's ink edges on the 180 mm page
INK_L, INK_R = 0.32, 0.15                          # how far the widest label's glyphs sit inside its text box, mm
LABEL_GAP = 1.0                                    # program label -> matrix, mm
KEY_GAP = 2.0                                      # matrix -> key column, mm
CAP, DESC = 0.716, 0.212                           # Liberation Sans cap height / descender depth, in em
PT = 25.4 / 72.0                                   # mm per pt

MARK_AREA = 30.0                                   # pt^2: every mark is 2.18 mm across
EDGE, EDGE_LW = INK, 0.4                           # hairline on the COSG side of a mark, pt
RECAST_RAMP = LinearSegmentedColormap.from_list("a5_recast", [RECAST_XL, RECAST_L, RECAST])
COSG_RAMP = LinearSegmentedColormap.from_list("a5_cosg", ["#FDD0A2", "#FD8D3C", "#D94801"])
BOX_LW, BOX_DASH, BOX_DX = 0.45, (0, (3.6, 2.4)), 0.35   # outline: pt; dash in units of the line width; inset, mm


def _axmm(fig, W, H, x, y, w, h):
    """Axes rect from millimetres, y measured down from the top of a (W x H) mm figure."""
    return fig.add_axes([x / W, (H - y - h) / H, w / W, h / H])


def _text(fig, x, y, s, fs, **kw):
    """Figure text at (x, y) mm from the top-left corner, kept at the size asked for."""
    t = fig.text(x / PAGE_W, (PAGE_H - y) / PAGE_H, s, fontsize=fs, **kw)
    keep(t, color=False, size=True)
    return t


def _extent(fig, s, fs, **kw):
    """(width, height) in mm of `s` at `fs` pt."""
    t = fig.text(0, 0, s, fontsize=fs, **kw)
    keep(t, color=False, size=True)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 25.4, bb.height / fig.dpi * 25.4


def _colour(ramp, f):
    """The colour of a fraction f of datasets enriched: f itself on the method's ramp."""
    return ramp(float(f))


def _mark(ax, x, y, fr, fc):
    """One cell: a RECAST disc, a COSG disc with a hairline, or a split disc (left RECAST, right COSG); empty if
    neither method's top 30 is enriched.  fr, fc: fraction of the subtype's datasets enriched for each method."""
    if fr > 0 and fc > 0:
        ax.scatter([x], [y], s=MARK_AREA, marker=MarkerStyle("o", fillstyle="left"),
                   facecolor=_colour(RECAST_RAMP, fr), edgecolor="none", linewidths=0, zorder=3)
        ax.scatter([x], [y], s=MARK_AREA, marker=MarkerStyle("o", fillstyle="right"),
                   facecolor=_colour(COSG_RAMP, fc), edgecolor=EDGE, linewidths=EDGE_LW, zorder=4)
    elif fr > 0:
        ax.scatter([x], [y], s=MARK_AREA, facecolor=_colour(RECAST_RAMP, fr), edgecolor="none",
                   linewidths=0, zorder=3)
    elif fc > 0:
        ax.scatter([x], [y], s=MARK_AREA, facecolor=_colour(COSG_RAMP, fc), edgecolor=EDGE,
                   linewidths=EDGE_LW, zorder=3)


def load():
    """Programs, the fraction of each subtype's datasets enriched for RECAST (Fr) and COSG (Fc), and the expected
    pairs (a boolean programs x COLS matrix)."""
    prog = pd.read_csv(f"{HERE}/data/programs.csv")
    EM = pd.read_csv(f"{HERE}/data/expected_pairs.csv", index_col=0)
    assert len(prog) == 18 and list(EM.index) == list(prog.display)
    assert list(EM.columns) == [f"{c}:{lab}" for c, _, lab in COLS]
    R = pd.read_csv(f"{HERE}/data/program_enrichment.csv").merge(POPULATIONS, on=["lineage", "state"], how="inner")
    R["comp"] = R.lineage.map(compartment_of)
    nd = POPULATIONS.assign(comp=POPULATIONS.lineage.map(compartment_of)).groupby(["comp", "state"]).size()
    ndat = [int(nd[(c, st)]) for c, st, _ in COLS]
    assert sum(ndat) == len(POPULATIONS) == 74
    Fr = np.zeros((len(prog), len(COLS))); Fc = np.zeros_like(Fr)
    for gs, M in [("RECAST", Fr), ("DE_cosg", Fc)]:
        for ri, r in enumerate(prog.itertuples()):
            d = R[(R.geneset == gs) & (R.collection == r.collection) & (R.term == r.term) & (R.fdr < 0.05)]
            hit = d.groupby(["comp", "state"]).size()
            for ci, (c, st, lab) in enumerate(COLS):
                M[ri, ci] = hit.get((c, st), 0) / ndat[ci]
    E = EM.values == "E"
    assert int(E.sum()) == 46
    return prog, Fr, Fc, E


def _union_loops(cells):
    """Outlines of the union of grid cells {(row, col)}: closed rectilinear loops, counter-clockwise in a y-up frame
    (x = col, y = -row), corners only.  Two cells that meet at a single corner stay two loops: at such a vertex the
    left turn (around the cell already being traced) wins."""
    S = set(cells)
    edges = {}
    for r, c in S:
        x0, x1, y0, y1 = c - 0.5, c + 0.5, -r - 0.5, -r + 0.5
        if (r + 1, c) not in S: edges.setdefault((x0, y0), []).append((x1, y0))   # bottom, left to right
        if (r, c + 1) not in S: edges.setdefault((x1, y0), []).append((x1, y1))   # right, upwards
        if (r - 1, c) not in S: edges.setdefault((x1, y1), []).append((x0, y1))   # top, right to left
        if (r, c - 1) not in S: edges.setdefault((x0, y1), []).append((x0, y0))   # left, downwards
    loops = []
    while edges:
        start = next(iter(edges))
        path, cur, pdir = [start], start, None
        while True:
            outs = edges[cur]
            if pdir is None or len(outs) == 1:
                nxt = outs[0]
            else:
                nxt = max(outs, key=lambda o: pdir[0] * (o[1] - cur[1]) - pdir[1] * (o[0] - cur[0]))
            outs.remove(nxt)
            if not outs:
                del edges[cur]
            pdir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            path.append(cur)
        n = len(path)
        loops.append([path[i] for i in range(n)
                      if (path[i][0] - path[i - 1][0]) * (path[(i + 1) % n][1] - path[i][1])
                      - (path[i][1] - path[i - 1][1]) * (path[(i + 1) % n][0] - path[i][0]) != 0])
    return loops


def _inset_x(loop, dx):
    """Move every vertical edge of a counter-clockwise (y-up) rectilinear loop dx towards the inside; the horizontal
    edges follow, because every corner ends exactly one vertical edge."""
    n, out = len(loop), [list(v) for v in loop]
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        if a[0] == b[0]:
            sh = -dx if b[1] > a[1] else dx                      # going up the inside is at -x, going down at +x
            out[i][0] = a[0] + sh
            out[(i + 1) % n][0] = b[0] + sh
    return out


def draw_matrix(fig, prog, Fr, Fc, expected):
    """Draw the matrix, its header, the subtype labels and the key on `fig`; return the labels' bottom in mm."""
    RP = ROW_PITCH
    A = (Fr > 0) | (Fc > 0)
    assert A.any(axis=1).all(), "a program row is empty"
    shown = [j for j in range(len(COLS)) if A[:, j].any() or expected[:, j].any()]
    kc = [COLS[j] for j in shown]
    ngap = sum(1 for j in range(1, len(kc)) if kc[j][0] != kc[j - 1][0])
    nrow = len(prog)
    width = lambda s, fs=FS, **kw: _extent(fig, s, fs, **kw)[0]          # noqa: E731
    TH = _extent(fig, "Ag", FS)[1]                                        # one label line's box

    # ---- the key column, measured first: the matrix gets what is left
    DMK = 2.0 * math.sqrt(MARK_AREA / math.pi) / 72.0 * 25.4             # 2.18 mm, every mark
    KGP = 1.0                                                            # mark -> its text
    kmk = [(0.8, 0.0, "RECAST only"), (0.8, 0.8, "Both"), (0.0, 0.8, "COSG only")]
    KCAP = ["Datasets", "enriched (%)"]
    KTICKS = [(0.0, "0"), (0.5, "50"), (1.0, "100")]
    KW = max([DMK + KGP + width(t) for _, _, t in kmk] + [width(t) for t in KCAP] + [4.4 * width("100")]
             + [DMK + KGP + width("Expected")])
    KX = INK_X1 - INK_R - KW
    XR = KX - KEY_GAP
    ital = ["italic" if c == "CollecTRI" else "normal" for c in prog.collection]
    LABW = max(width(lab, fontstyle=st) for lab, st in zip(prog.display, ital))
    LR = INK_X0 - INK_L + LABW                                           # program labels end here (right-aligned)
    AX = LR + LABEL_GAP                                                  # matrix left edge
    pitch = (XR - AX - ngap * BLOCK_GAP) / len(kc)
    assert pitch * 0.7071 - TH > 0.3, \
        f"the 45 deg subtype labels would touch: {pitch * 0.7071 - TH:.2f} mm apart (pitch {pitch:.2f})"
    assert pitch - DMK > 0.6, "the marks in adjacent columns would touch"
    hl = BOX_LW * PT / 2.0                                               # every outline edge clears every mark by 0.2 mm
    assert RP / 2.0 - DMK / 2.0 - hl >= 0.2, "a horizontal outline edge runs into a mark"
    assert pitch / 2.0 - BOX_DX - DMK / 2.0 - hl >= 0.2, "a vertical outline edge runs into a mark"
    gu, xc, ng = BLOCK_GAP / pitch, [], 0
    for j, (c, st, lab) in enumerate(kc):
        if j and c != kc[j - 1][0]:
            ng += 1
        xc.append(j + ng * gu)
    matw = (len(kc) + ngap * gu) * pitch

    # ---- vertical: tissue spanner (8 pt bold) / rule / lineage (7 pt) / matrix, from y = 0 at the spanners' cap top
    E1, E2 = FS_TISSUE * PT, FS_LINEAGE * PT                             # em in mm
    y1 = 0.3 + CAP * E1                                                  # tissue baseline (0.3 mm of white above)
    yr = y1 + DESC * E1 + 0.30                                           # the rule, 0.3 mm under the "g" of Lung
    y2 = yr + 0.45 + CAP * E2                                            # lineage baseline
    Y0 = y2 + DESC * E2 + 0.60                                           # matrix top: 0.6 mm under the "p" of Epithelium
    ybot = Y0 + RP * nrow

    ax = _axmm(fig, PAGE_W, PAGE_H, AX, Y0, matw, RP * nrow)
    ax.set_xlim(-0.5, xc[-1] + 0.5); ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    b0 = 0
    for j in range(1, len(kc) + 1):                                      # one frame per compartment block
        if j == len(kc) or kc[j][0] != kc[b0][0]:
            ax.add_patch(Rectangle((xc[b0] - 0.5, -0.5), xc[j - 1] - xc[b0] + 1.0, nrow, facecolor="none",
                                   edgecolor=BLACK, linewidth=FRAME_LW, zorder=1.5, clip_on=False))
            b0 = j
    for ri in range(nrow):
        for j, ci in enumerate(shown):
            _mark(ax, xc[j], ri, Fr[ri, ci], Fc[ri, ci])
    n_cells = 0
    j0 = 0
    for j in range(1, len(kc) + 1):                                      # the expected cells, outlined block by block
        if j == len(kc) or kc[j][0] != kc[j0][0]:
            cells = [(ri, jj - j0) for ri in range(nrow) for jj in range(j0, j) if expected[ri, shown[jj]]]
            for loop in _union_loops(cells):
                vx = [(xc[j0] + x, -y) for x, y in _inset_x(loop, BOX_DX / pitch)]
                ax.add_patch(Polygon(vx, closed=True, fill=False, edgecolor=BLACK, linewidth=BOX_LW,
                                     linestyle=BOX_DASH, joinstyle="miter", capstyle="butt", zorder=2.5,
                                     clip_on=False))
            n_cells += len(cells)
            j0 = j
    assert n_cells == int(expected[:, shown].sum())
    for ri, (lab, st) in enumerate(zip(prog.display, ital)):             # program names
        _text(fig, LR, Y0 + (ri + 0.5) * RP, lab, FS, ha="right", va="center_baseline", fontstyle=st, color=INK)

    # ---- two-tier header: lineage centred on each block, tissue over the blocks it groups
    span, i0 = {}, 0
    for j in range(1, len(kc) + 1):
        if j == len(kc) or kc[j][0] != kc[i0][0]:
            span[kc[i0][0]] = (AX + xc[i0] * pitch, AX + (xc[j - 1] + 1.0) * pitch)
            i0 = j
    assert set(span) == set(COMPARTMENTS), "every compartment must keep at least one drawn column"
    t2 = []
    for comp in COMPARTMENTS:
        x0, x1 = span[comp]
        w = width(LINEAGE_LABEL[comp], FS_LINEAGE)
        t2.append(((x0 + x1) / 2.0 - w / 2.0, (x0 + x1) / 2.0 + w / 2.0))
        _text(fig, (x0 + x1) / 2.0, y2, LINEAGE_LABEL[comp], FS_LINEAGE, ha="center", va="baseline", color=INK)
    gaps2 = [b[0] - a[1] for a, b in zip(t2, t2[1:])]
    assert min(gaps2) >= 0.8, f"two lineage labels come within {min(gaps2):.2f} mm"
    t1 = []
    for name, comps in TISSUES:
        gx0, gx1 = span[comps[0]][0], span[comps[-1]][1]
        tw1 = width(name, FS_TISSUE, fontweight="bold")
        # "Pancreas" is wider than the islet block: a spanner may overhang its rule by up to the block gap on each side
        assert (tw1 - (gx1 - gx0)) / 2.0 <= BLOCK_GAP - 0.1, \
            f"spanner '{name}' ({tw1:.1f} mm) overhangs its {gx1 - gx0:.1f} mm span by more than the block gap"
        t1.append(((gx0 + gx1) / 2.0 - tw1 / 2.0, (gx0 + gx1) / 2.0 + tw1 / 2.0))
        _text(fig, (gx0 + gx1) / 2.0, y1, name, FS_TISSUE, ha="center", va="baseline", fontweight="bold", color=INK)
        fig.add_artist(Line2D([gx0 / PAGE_W, gx1 / PAGE_W], [(PAGE_H - yr) / PAGE_H] * 2, color=BLACK, lw=FRAME_LW,
                              solid_capstyle="butt", transform=fig.transFigure))
    gaps1 = [b[0] - a[1] for a, b in zip(t1, t1[1:])]
    assert min(gaps1) >= 1.5 and t1[-1][1] <= INK_X1, f"two tissue spanners come within {min(gaps1):.2f} mm"

    # ---- subtype labels: 45 degrees, right end under the column centre
    ylab = ybot + 1.6
    depth = 0.0
    for j, ci in enumerate(shown):
        s = column_label(ci)
        depth = max(depth, (width(s) + TH / 2.0) * 0.7071)
        _text(fig, AX + (xc[j] + 0.5) * pitch, ylab, s, FS, ha="right", va="center", rotation=45,
              rotation_mode="anchor", color=INK)
    abot = ylab + depth

    # ---- the key: three marks, the outline, the caption and the two intensity bars, on one left edge
    kh = RP * nrow
    kax = _axmm(fig, PAGE_W, PAGE_H, KX, Y0, KW, kh)
    kax.set_xlim(0, KW); kax.set_ylim(kh, 0.0); kax.axis("off")
    for (fr, fc, txt), i in zip(kmk, (0, 1, 2)):
        y = (i + 0.5) * RP
        _mark(kax, DMK / 2.0, y, fr, fc)                                 # the mark's left edge on x = 0
        keep(kax.text(DMK + KGP, y, txt, fontsize=FS, va="center_baseline", ha="left", color=INK),
             color=False, size=True)
    bh = RP - 0.5                                                        # the outline sample, a little less than a row
    y0b, y1b = 3.5 * RP - bh / 2.0, 3.5 * RP + bh / 2.0
    kax.add_patch(Polygon([(0.0, y0b), (DMK, y0b), (DMK, y1b), (0.0, y1b)], closed=True, fill=False,
                          edgecolor=BLACK, linewidth=BOX_LW, linestyle=BOX_DASH, joinstyle="miter",
                          capstyle="butt", clip_on=False))
    keep(kax.text(DMK + KGP, 3.5 * RP, "Expected", fontsize=FS, va="center_baseline", ha="left", color=INK),
         color=False, size=True)
    ycap = 4.0 * RP + 1.15                                               # caption: below the outline sample
    for li, t in enumerate(KCAP):                                        # two lines, 1.2 em apart
        keep(kax.text(0.0, ycap + li * 1.2 * FS * PT, t, fontsize=FS, va="center_baseline", ha="left", color=INK),
             color=False, size=True)
    KBH, KBG, KGRAD = 1.5, 0.45, 96                                      # bar height, gap between the bars, slices
    KB_Y = ycap + (len(KCAP) - 1) * 1.2 * FS * PT + 1.45
    for bi, kramp in enumerate((RECAST_RAMP, COSG_RAMP)):
        yb = KB_Y + bi * (KBH + KBG)
        for sl in range(KGRAD):                                          # the same colour function as the marks
            f0 = sl / KGRAD
            kax.add_patch(Rectangle((f0 * KW, yb), KW / KGRAD * 1.03, KBH, facecolor=_colour(kramp, f0 + 0.5 / KGRAD),
                                    edgecolor="none", linewidth=0, zorder=2, clip_on=False))
        kax.add_patch(Rectangle((0.0, yb), KW, KBH, facecolor="none", edgecolor=EDGE, linewidth=EDGE_LW, zorder=3,
                                clip_on=False))
    ybb = KB_Y + 2.0 * KBH + KBG
    for f, t in KTICKS:
        kax.plot([f * KW, f * KW], [ybb, ybb + 0.5], color=EDGE, linewidth=EDGE_LW, zorder=3, clip_on=False,
                 solid_capstyle="butt")
        keep(kax.text(f * KW, ybb + 0.65, t, fontsize=FS, va="top", color=INK,
                      ha="left" if f == 0.0 else ("right" if f == 1.0 else "center")), color=False, size=True)
    kbot = Y0 + ybb + 0.65 + TH
    assert kbot < ybot, "the key runs below the matrix"
    return abot


def draw(stem, outdir):
    """Draw Fig. 5a and save it as outdir/stem.{pdf,svg,png}."""
    prog, Fr, Fc, E = load()
    fig = plt.figure(figsize=(mm(PAGE_W), mm(PAGE_H)))
    abot = draw_matrix(fig, prog, Fr, Fc, E)
    assert abot < PAGE_H - 1.0
    return save_fig(fig, stem, outdir)
