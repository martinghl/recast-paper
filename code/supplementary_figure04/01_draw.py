"""Supplementary Fig. 4: broad populations and cell-cycle states.

  a  expert-panel marker recovery of broad populations (62 populations in 7 atlases; CITE-seq has no expert panel)
  b  within-atlas AUROC of the selected genes scored as one signature, 52 methods per atlas
  c  cell-cycle fraction of up to 20 selected genes, 52 method configurations: mean (bar) and interquartile range
     (line) over the 35 phase-defined comparisons
  d  mean fraction at 20, 50, 100 and 200 requested genes for RECAST, the highest configuration mean at each number
     (an envelope, not one method) and the alphabetical control, with RECAST's rank among the 52 configurations
Data: broad_population_data.py (a, b) and cell_cycle_data.py (c, d).  Jitter in b uses default_rng(0), one draw per
grey point, keeping clear of the two named symbols.

Run     python 01_draw.py
Output  out/S4_broad_and_cell_cycle.{pdf,svg,png}
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
import matplotlib.text as mtext           # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402

sys.path.insert(0, HERE)
import broad_population_data as M         # noqa: E402
import cell_cycle_data as C9              # noqa: E402

S.setup()
from method_colours import G300, SUB, NULLC, NULL_L, DE_XL, INK   # noqa: E402

STEM = "S4_broad_and_cell_cycle"
# ------------------------------------------------------------------ data
DRAWN_OTHER = list(M.OTHER)
assert len(DRAWN_OTHER) == 50
B52 = M.auc[["RECAST", M.WIL] + DRAWN_OTHER]
assert B52.shape == (8, 52)
ROWS = M.ROWS
METH, DISP, CMAP, EDGE = M.METH, M.DISP, M.CMAP, M.EDGE
recall, npop, pooled = M.recall, M.npop, M.pooled
TC = "scanpy_wilcoxontiecorrect_rankby_raw"
_b = pd.read_csv(f"{HERE}/data/broad_marker_recall_by_population.csv")   # per population, all methods
_c = _b[(_b.method == "cosg") & _b.dataset.isin(recall.index)]
assert (_c.groupby("dataset").recall.mean().reindex(recall.index) - recall["COSG"]).abs().max() < 1e-12
_b = _b[(_b.method == TC) & _b.dataset.isin(recall.index)]
assert _b.groupby("dataset").cluster.nunique().to_dict() == npop.to_dict() and len(_b) == 62
recall = recall.copy()
recall["Wilcoxon"] = _b.groupby("dataset").recall.mean().reindex(recall.index)
pooled = dict(pooled, Wilcoxon=float(_b.recall.mean()))
assert round(pooled["Wilcoxon"], 3) == 0.287
DISP = dict(DISP, Wilcoxon="Tie-corrected Wilcoxon")
# b: COSG is ringed; Scanpy Wilcoxon without tie correction takes its place among the grey points
RING = "cosg"
OTHER_B = [m for m in M.OTHER if m != RING] + [M.WIL]
assert len(OTHER_B) == 50
f20, pe, rank_recast = C9.f20, C9.pe, C9.rank_recast
q25 = f20.groupby("method").precision.quantile(0.25)
q75 = f20.groupby("method").precision.quantile(0.75)
KS, recast_k, best_k, best_id, rand_k, rk = C9.KS, C9.recast_k, C9.best_k, C9.best_id, C9.rand_k, C9.rk

# ------------------------------------------------------------------ page geometry (mm)
W = 180.0
LEFT_R = 70.0                              # right edge of the left column (texts included)
C_LAB_R, C_AX_W = 127.0, 35.5              # c: label column right edge, bar axis width (0-1)
C_TOP, C_PITCH = 6.0, 3.25
A_TOP, A_PITCH = 11.0, 5.2
B_PITCH = 4.6
P = None


def main():
    global P
    nC = len(pe)
    c_h = (nC - 1 + 1.0) * C_PITCH                       # rows -0.5 .. n-0.5
    H = C_TOP + c_h + 13.0
    P = S.Page(W, H)
    fig = P.fig
    checks = {}

    # ============================================================== a
    nA = len(ROWS) + 1
    ax_a = P.ax(31.0, A_TOP, LEFT_R - 31.0 - 1.0, nA * A_PITCH)
    S.frame(ax_a, spines=("bottom",))
    off = dict(zip(METH, (-0.27, -0.09, 0.09, 0.27)))
    yl = []
    for r, (sid, rid, lab) in enumerate(ROWS):
        if rid is None:
            ax_a.text(0.005, r, "no expert panel", ha="left", va="center", color=SUB)
            yl.append(lab)
            continue
        for m in METH:
            ax_a.barh(r + off[m], recall.loc[rid, m], height=0.17, color=CMAP[m], edgecolor=EDGE[m], lw=0.3, zorder=3)
        yl.append(f"{lab} ({int(npop[rid])})")
    rp = len(ROWS)
    for m in METH:
        ax_a.barh(rp + off[m], pooled[m], height=0.17, color=CMAP[m], edgecolor=EDGE[m], lw=0.3, zorder=3)
    yl.append("Pooled (62)")
    ax_a.axhline(rp - 0.5, color=S.HAIR, lw=S.HAIR_LW, zorder=1)
    ax_a.set_yticks(range(nA))
    ax_a.set_yticklabels(yl)
    ax_a.get_yticklabels()[-1].set_fontweight("bold")
    ax_a.tick_params(axis="y", length=0, pad=2)
    ax_a.set_ylim(nA - 0.5, -0.5)
    ax_a.set_xlim(0, 0.5)
    ax_a.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_a.set_xticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    ax_a.set_xlabel("Expert-marker recovery")
    assert recall.max().max() < 0.5 and max(pooled.values()) < 0.5
    key_a = []
    for i, m in enumerate(METH):                          # two key rows above the plot
        row, col = divmod(i, 2)
        kx = 5.0 + col * 30.0
        ky = 2.6 + row * 3.6
        fig.patches.append(plt.Rectangle((kx / W, 1 - (ky + 0.8) / H), 2.4 / W, 1.6 / H, transform=fig.transFigure,
                                         fc=CMAP[m], ec=EDGE[m], lw=0.3))
        key_a.append(P.text(kx + 3.2, ky, DISP[m], va="center", ha="left",
                            fontweight="bold" if m == "RECAST" else "normal"))
    P.letter(0.0, 0.6, "a")

    # ============================================================== b
    fig.canvas.draw()
    a_bottom = (fig.bbox.y1 - ax_a.xaxis.label.get_window_extent(fig.canvas.get_renderer()).y0) / K.px_per_mm(fig)
    B_LET = a_bottom + 4.0
    B_TOP = B_LET + 9.0
    nB = len(ROWS)
    XLIM = (0.15, 1.0)
    ax_b = P.ax(26.0, B_TOP, LEFT_R - 26.0 - 1.0, nB * B_PITCH)
    S.frame(ax_b, spines=("bottom",))
    assert B52.min().min() > XLIM[0] and B52.max().max() <= XLIM[1]
    mm_x = (LEFT_R - 27.0) / (XLIM[1] - XLIM[0])
    mm_y = B_PITCH
    rng = np.random.default_rng(0)
    R_REC, R_WIL = 1.15, 1.05
    for r, (sid, _, _) in enumerate(ROWS):
        fv, wv = float(M.auc.loc[sid, "RECAST"]), float(M.auc.loc[sid, RING])
        f = M.auc.loc[sid, OTHER_B].to_numpy()
        ys = []
        for v in f:                                       # one draw per grey point, away from the two named symbols
            for _ in range(2000):
                y = r + rng.uniform(-0.30, 0.30)
                if np.hypot((v - fv) * mm_x, (y - r) * mm_y) > R_REC and np.hypot((v - wv) * mm_x, (y - r) * mm_y) > R_WIL:
                    break
            else:
                raise RuntimeError(f"no free jitter position in {sid}")
            ys.append(y)
        keep = np.ones(len(OTHER_B), dtype=bool)
        assert keep.sum() == 50
        ax_b.scatter(f[keep], np.asarray(ys)[keep], s=4.0, color=G300, lw=0, zorder=2, clip_on=False)
        ax_b.scatter([wv], [r], s=14, facecolors="none", edgecolors=INK, lw=0.7, zorder=4, clip_on=False)
        ax_b.scatter([fv], [r], s=16, color=S.RECAST, edgecolors="white", lw=0.4, zorder=5, clip_on=False)
    ax_b.set_yticks(range(nB))
    ax_b.set_yticklabels([lab for _, _, lab in ROWS])
    ax_b.tick_params(axis="y", length=0, pad=2)
    ax_b.set_ylim(nB - 0.5, -0.5)
    ax_b.set_xlim(*XLIM)
    ax_b.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
    for r in range(nB):
        ax_b.axhline(r, color=S.HAIR, lw=S.HAIR_LW, zorder=0)
    ax_b.set_xlabel("Within-atlas AUROC\n(up to 20 selected genes)", linespacing=1.12)
    key_b = []
    for i, (kind, lab) in enumerate((("rec", "RECAST"), ("wil", "COSG"), ("oth", "Other methods"))):
        row, col = (0, i) if i < 2 else (1, 0)
        kx = 6.2 + col * 30.0
        ky = B_LET + 2.0 + row * 3.6
        kax = P.ax(kx - 1.2, ky - 1.2, 2.4, 2.4)
        kax.set_xlim(0, 1)
        kax.set_ylim(0, 1)
        kax.set_axis_off()
        if kind == "rec":
            kax.scatter([0.5], [0.5], s=16, color=S.RECAST, edgecolors="white", lw=0.4)
        elif kind == "wil":
            kax.scatter([0.5], [0.5], s=14, facecolors="none", edgecolors=INK, lw=0.7)
        else:
            kax.scatter([0.5], [0.5], s=4.0, color=G300, lw=0)
        key_b.append(P.text(kx + 2.0, ky, lab, va="center", ha="left", fontweight="bold" if kind == "rec" else "normal"))
    P.letter(0.0, B_LET + 0.2, "b")

    # ============================================================== c
    methods = pe.index.tolist()
    ax_c = P.ax(C_LAB_R + 1.5, C_TOP, C_AX_W, c_h)
    S.frame(ax_c)
    ypos = np.arange(len(methods))
    for m, y in zip(methods, ypos):
        c_, z = (S.RECAST, 4) if m == "RECAST_M0" else ((NULL_L, 3) if m == "random" else (DE_XL, 3))
        ax_c.barh(y, pe[m], height=0.72, color=c_, zorder=z, hatch="////" if m == "random" else None,
                  edgecolor=NULLC if m == "random" else "none", linewidth=0.3)
        ax_c.plot([q25[m], q75[m]], [y, y], color=INK, lw=0.5, zorder=5, solid_capstyle="butt")
    ax_c.set_ylim(len(methods) - 0.5, -0.5)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([C9.name(m) for m in methods])
    ax_c.tick_params(axis="y", length=0, pad=1.5)
    for lab, m in zip(ax_c.get_yticklabels(), methods):
        if m == "RECAST_M0":
            lab.set_color(S.RECAST_D)
            lab.set_fontweight("bold")
    callouts = []
    for m, s_, col, bold in (("cepo", f"{pe['cepo']:.3f}", SUB, False),
                             ("RECAST_M0", f"{pe['RECAST_M0']:.3f} (#{rank_recast}/52)", S.RECAST_D, True),
                             ("random", f"{pe['random']:.3f}", SUB, False)):
        callouts.append(ax_c.annotate(s_, (q75[m] + 0.015, ypos[methods.index(m)]), va="center", color=col, zorder=7,
                                      fontweight="bold" if bold else "normal", annotation_clip=False))
    ax_c.set_xlim(0, 1.0)
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_c.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax_c.set_xlabel("Cell-cycle fraction of selected genes\n(up to 20 genes; mean of 35 comparisons)", linespacing=1.12)
    P.letter(LEFT_R + 2.0, 0.6, "c")

    # ============================================================== d
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    b_bottom = (fig.bbox.y1 - ax_b.xaxis.label.get_window_extent(rend).y0) / K.px_per_mm(fig)
    D_LET = b_bottom + 4.0
    D_TOP = D_LET + 9.0                                       # two key rows above the plot
    D_BOT = C_TOP + c_h                                       # bottom axis line level with c's
    ax_d = P.ax(17.0, D_TOP, LEFT_R - 17.0 - 2.0, D_BOT - D_TOP)
    S.frame(ax_d)
    xi = np.arange(len(KS))
    ax_d.plot(xi, best_k, "-o", color=S.DE, ms=3.0, lw=1.0, label="Best-performing comparator", zorder=4)
    ax_d.plot(xi, recast_k, "-o", color=S.RECAST, ms=3.0, lw=1.0, label="RECAST", zorder=4)
    ax_d.plot(xi, rand_k, "--", color=NULLC, lw=0.8, label="Alphabetical control", zorder=3)
    ranks = []
    for x, y, r in zip(xi, recast_k, rk):
        ranks.append(ax_d.annotate(f"#{r}", (x, y - 0.045), ha="center", va="top", color=S.RECAST_D))
    ax_d.set_xticks(xi)
    ax_d.set_xticklabels([str(k_) for k_ in KS])
    ax_d.set_xlabel("Requested number of genes")
    ax_d.set_ylabel("Cell-cycle fraction\nof selected genes\n(mean of 35 comparisons)", linespacing=1.12)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax_d.set_xlim(-0.25, 3.25)
    hd = {h.get_label(): h for h in ax_d.get_lines()}
    leg_kw = dict(frameon=False, handlelength=1.6, borderaxespad=0, borderpad=0, handletextpad=0.5)
    y_row1, y_row2 = D_LET + 1.6, D_LET + 5.1             # mm from the top
    leg_d1 = fig.legend(handles=[hd["Best-performing comparator"]], loc="center left",
                        bbox_to_anchor=(16.0 / W, 1 - y_row1 / H), **leg_kw)
    leg_d2 = fig.legend(handles=[hd["RECAST"], hd["Alphabetical control"]], loc="center left", ncol=2,
                        columnspacing=1.4, bbox_to_anchor=(16.0 / W, 1 - y_row2 / H), **leg_kw)
    leg_d = leg_d2
    P.letter(0.0, D_LET - 2.0, "d")

    # ============================================================== checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = K.px_per_mm(fig)
    right_col_x0 = min(t.get_window_extent(rend).x0 for t in ax_c.get_yticklabels()) / k
    left_texts = (key_a + key_b + [ax_a.xaxis.label, ax_b.xaxis.label, ax_d.xaxis.label]
                  + ax_a.get_xticklabels() + ax_b.get_xticklabels() + ax_d.get_xticklabels() + ranks
                  + leg_d1.get_texts() + leg_d2.get_texts())
    left_x1 = max(t.get_window_extent(rend).x1 for t in left_texts if t.get_text()) / k
    checks["left_column_right_mm"] = round(left_x1, 2)
    checks["c_labels_left_mm"] = round(right_col_x0, 2)
    assert right_col_x0 - left_x1 > 2.5, (left_x1, right_col_x0)
    for t in callouts:                                    # c's callouts stay on the page
        assert t.get_window_extent(rend).x1 / k < W - 0.8, t.get_text()
    ax_top = ax_d.get_window_extent(rend).y1             # d's key sits above the plot
    for lg in (leg_d1, leg_d2):
        assert lg.get_window_extent(rend).y0 > ax_top + 0.5 * k
    for t in ranks:                                       # rank labels clear of every line vertex
        bb = mtext.Text.get_window_extent(t, rend)
        for ln in ax_d.get_lines():
            xy = ln.get_transform().transform(ln.get_xydata())
            assert not ((xy[:, 0] > bb.x0) & (xy[:, 0] < bb.x1) & (xy[:, 1] > bb.y0) & (xy[:, 1] < bb.y1)).any()
    checks.update(pooled_a={DISP[m]: round(pooled[m], 3) for m in METH},
                  b_signature_mean={"RECAST": round(float(M.auc["RECAST"].mean()), 3),
                                    "COSG": round(float(M.auc[RING].mean()), 3)},
                  c_recast=(round(float(pe["RECAST_M0"]), 3), rank_recast), d_ranks=rk,
                  d_best=[round(float(v), 3) for v in best_k], page_mm=(W, round(H, 1)))
    P.save(f"{HERE}/out", STEM)
    print(STEM, checks)

if __name__ == "__main__":
    main()
