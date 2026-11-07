"""Supplementary Fig. 6: gene programs outside the expected ones.

  a  difference in program-gene counts, RECAST minus COSG, for each of the 1,164 unexpected program x population pairs:
     all pairs, then by compartment (box with IQR and 1.5 x IQR whiskers, dashed mean, jittered points; the jitter
     positions come from panel_a_points.R)
  b  % of unexpected pairs enriched in the equally sized method-specific gene sets (BH FDR < 0.05), RECAST-specific
     against COSG-specific, counts above the bars, 95% intervals for the overall percentages
  c  mean expression (log(1 + CP10K)) of four program genes in the subtype RECAST ranks the gene for and in every
     sibling subtype; GZMM: upper bar Ren, lower bar Yoshida (two datasets of the same program in the same block)
Populations, compartments and pair classes are those of Fig. 5 (74 populations, 39 subtypes, 8 compartments,
99 expected / 69 plausible / 1,164 unexpected pairs).  Panel c: example_genes.py; short subtype names from
common/subtype_columns.py.

Run     python 01_draw.py        (runs Rscript once; RSCRIPT names it, default "Rscript")
Output  out/S6_program_validation.{pdf,svg,png}
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S                    # noqa: E402
import supplementary_common as K                   # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import Patch, Rectangle   # noqa: E402

import subtype_columns as SC              # noqa: E402  (short subtype names)
sys.path.insert(0, HERE)
import example_genes as X                 # noqa: E402
S.setup()

STEM = "S6_program_validation"
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")
D = f"{K.CODE}/shared_data"
W = 180.0
RECAST_BLUE, COSG_ORANGE, REF_COL, ZERO_GREY, BOX_FILL = "#4E86B0", "#D94801", "#8C8C8C", "#9E9E9E", "#EFEFEF"
G400, SUB = "#A0A0A0", "#6E6E6E"          # sibling bars / sibling labels (grey, explained in the key)
COMP = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]
GRP = ["All pairs"] + COMP
MINUS = "−"


def minus(s):
    return s.replace("-", MINUS)


def fmt_p(p):                             # P values as printed in Fig. 5
    return "P < 0.001" if p < 0.001 else (f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}")


# ------------------------------------------------------------------ a, b data
P_ = pd.read_csv(f"{D}/enrichment_pairs.csv")
SU = pd.read_csv(f"{D}/enrichment_summary.csv")
SU = SU[SU.kind == "capture"]
SP = pd.read_csv(f"{D}/program_genes_method_specific.csv")
SP = SP[SP.set == "specific"]
CC = pd.read_csv(f"{D}/program_genes_by_comparison.csv")
assert (P_.cls == "E").sum() == 99 and (P_.cls == "U").sum() == 1164 and set(P_.comp) == set(COMP)
assert len(SU) == 2 and len(SP) == 2 and list(CC.comp) == COMP
su = SU[SU.pairs == "unexpected"].iloc[0]
sp = SP[SP.metric == "unexpected"].iloc[0]

pts_csv = os.path.join(HERE, "out", "panel_a_points.csv")
os.makedirs(os.path.dirname(pts_csv), exist_ok=True)
subprocess.run([RSCRIPT, os.path.join(HERE, "panel_a_points.R"), f"{D}/enrichment_pairs.csv",
                f"{D}/enrichment_summary.csv", pts_csv], check=True)
BX = pd.read_csv(pts_csv)
assert len(BX) == 2 * 1164 and not BX.off.any() and BX.dk.max() < 3.6
assert abs(BX[BX.grp == "All pairs"].dk.mean() - su["diff"]) < 1e-9
NB = BX.groupby(["gi", "grp"]).size().reset_index(name="n").sort_values("gi")
NB["lab"] = [f"{g} ({n:,})" for g, n in zip(NB.grp, NB.n)]
assert len(NB) == 9 and NB.n.iloc[0] == su.n_pairs and NB.n.iloc[1:].sum() == su.n_pairs

U = P_[P_.cls == "U"]
BYC = (U.groupby("comp").agg(n=("comp", "size"), k_r=("hit_recast_specific", "sum"),
                             k_c=("hit_cosg_specific", "sum")).reindex(COMP).reset_index())
assert BYC.n.sum() == sp.n and BYC.k_r.sum() == sp.recast_n and BYC.k_c.sum() == sp.cosg_n
assert (BYC.n.to_numpy() == NB.n.iloc[1:].to_numpy()).all()
G = pd.concat([pd.DataFrame(dict(grp=["All pairs"], n=[sp.n], k_r=[sp.recast_n], k_c=[sp.cosg_n], lo_r=[sp.r_lo],
                                 hi_r=[sp.r_hi], lo_c=[sp.c_lo], hi_c=[sp.c_hi])),
               BYC.rename(columns={"comp": "grp"}).assign(lo_r=np.nan, hi_r=np.nan, lo_c=np.nan, hi_c=np.nan)])
G["gi"] = [GRP.index(g) + 1 for g in G.grp]
BARS = pd.concat([
    G.assign(method="RECAST-specific", k=G.k_r, lo=G.lo_r, hi=G.hi_r, dx=-0.15),
    G.assign(method="COSG-specific", k=G.k_c, lo=G.lo_c, hi=G.hi_c, dx=+0.15)])[
    ["gi", "grp", "n", "method", "k", "lo", "hi", "dx"]].reset_index(drop=True)
BARS["pct"] = 100 * BARS.k / BARS.n
assert abs(BARS[(BARS.gi == 1) & (BARS.method == "RECAST-specific")].pct.iloc[0] - sp.recast) < 1e-9
assert abs(BARS[(BARS.gi == 1) & (BARS.method == "COSG-specific")].pct.iloc[0] - sp.cosg) < 1e-9
_ci = BARS.dropna(subset=["lo"])
assert ((_ci.lo <= _ci.pct) & (_ci.pct <= _ci.hi)).all()
SUB_BOX = minus(f"Mean difference {su['diff']:+.2f} genes (95% CI {su.lo:.2f} to {su.hi:.2f}); {fmt_p(su.p)}")
SUB_BAR = minus(f"Difference {sp['diff']:+.1f} points (95% CI {sp.d_lo:.1f} to {sp.d_hi:.1f}); "
                f"{fmt_p(sp.p_perm)}")
assert SUB_BOX == "Mean difference −0.02 genes (95% CI −0.07 to 0.02); P = 0.40", SUB_BOX
assert SUB_BAR == "Difference −0.4 points (95% CI −1.1 to 0.2); P = 0.44", SUB_BAR


def box_stats(v):
    """ggplot2 stat_boxplot: type-7 quartiles, whiskers = most extreme data within 1.5 x IQR of the hinges"""
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_w = v[v >= q1 - 1.5 * iqr].min()
    hi_w = v[v <= q3 + 1.5 * iqr].max()
    return dict(n=len(v), q1=q1, median=med, q3=q3, whisker_low=lo_w, whisker_high=hi_w, mean=float(np.mean(v)))


def main(H):
    """Draw the page at height H (mm); every position is measured from the top, so H only adds or trims the bottom
    margin.  Returns the page, the height the content needs, the checks and the tables."""
    R1_TOP, PH, PW = 10.6, 30.0, 70.0
    AX_A, AX_B = 16.0, 108.0
    XL = (0.4, len(GRP) + 0.6)
    P = S.Page(W, H)
    fig = P.fig
    checks = {}

    # ============================================================== a
    ax = P.ax(AX_A, R1_TOP, PW, PH)
    ax.axhline(0, color=REF_COL, lw=0.4, ls=(0, (2 / 0.4, 2 / 0.4)), zorder=0.5)    # R linetype "22": 2 pt on / 2 pt off
    stats = []
    for gi, grp in enumerate(GRP, start=1):
        v = BX[BX.grp == grp].dk.to_numpy()
        st = box_stats(v)
        stats.append(dict(group=grp, gi=gi, **st))
        ax.add_patch(Rectangle((gi - 0.3, st["q1"]), 0.6, st["q3"] - st["q1"], facecolor=BOX_FILL, edgecolor="none",
                               zorder=1))
    col = {-1: COSG_ORANGE, 0: ZERO_GREY, 1: RECAST_BLUE}
    for sgn, c in col.items():
        q = BX[BX.sgn == sgn]
        ax.scatter(q.xj, q.yp, s=4.2, facecolor=c, edgecolor=c, linewidths=0.2, alpha=0.7, zorder=2)
    for st in stats:                                                   # outlines, median, whiskers, mean over points
        gi = st["gi"]
        ax.add_patch(Rectangle((gi - 0.3, st["q1"]), 0.6, st["q3"] - st["q1"], facecolor="none",
                               edgecolor=S.BLACK, lw=0.4, zorder=3))
        ax.plot([gi - 0.3, gi + 0.3], [st["median"]] * 2, color=S.BLACK, lw=0.8, solid_capstyle="butt", zorder=3)
        ax.plot([gi, gi], [st["q3"], st["whisker_high"]], color=S.BLACK, lw=0.4, zorder=3)
        ax.plot([gi, gi], [st["whisker_low"], st["q1"]], color=S.BLACK, lw=0.4, zorder=3)
        ax.plot([gi - 0.3, gi + 0.3], [st["mean"]] * 2, color=S.BLACK, lw=0.4, ls=(0, (4 / 0.4, 4 / 0.4)), zorder=4)  # R "44"
    ax.set_xlim(*XL)
    ax.set_ylim(-3.6, 3.6)
    ax.set_yticks(range(-3, 4))
    ax.set_yticklabels([minus(str(t)) for t in range(-3, 4)])
    S.frame(ax)
    ax.set_xticks(NB.gi)
    ax.set_xticklabels(NB.lab, rotation=45, ha="right", va="top", rotation_mode="anchor")
    ax.set_ylabel("Program genes,\nRECAST − COSG", linespacing=1.12)
    ax_a = ax

    # ============================================================== b
    ax = P.ax(AX_B, R1_TOP, PW, PH)
    YMAX = 5.0
    for meth, c in (("RECAST-specific", RECAST_BLUE), ("COSG-specific", COSG_ORANGE)):
        q = BARS[BARS.method == meth]
        ax.bar(q.gi + q.dx, q.pct, width=0.3, color=c, edgecolor="none", zorder=2)
    for _, r in BARS.dropna(subset=["lo"]).iterrows():
        x = r.gi + r.dx
        ax.plot([x, x], [r.lo, r.hi], color=S.BLACK, lw=0.4, zorder=3)
        for yy in (r.lo, r.hi):
            ax.plot([x - 0.05, x + 0.05], [yy, yy], color=S.BLACK, lw=0.4, zorder=3)
    mm2y = YMAX / PH
    for _, r in BARS.iterrows():
        top = r.hi if not np.isnan(r.hi) else r.pct
        ax.text(r.gi + r.dx, top + 0.8 * mm2y, f"{int(r.k)}", ha="center", va="bottom")
    ax.set_xlim(*XL)
    ax.set_ylim(0, YMAX)
    ax.set_yticks(range(0, 6))
    S.frame(ax)
    ax.set_xticks(NB.gi)
    ax.set_xticklabels(NB.lab, rotation=45, ha="right", va="top", rotation_mode="anchor")
    ax.set_ylabel("Pairs enriched by\nspecific genes (%)", linespacing=1.12)
    leg = ax.legend(handles=[Patch(facecolor=RECAST_BLUE, label="RECAST-specific"),
                             Patch(facecolor=COSG_ORANGE, label="COSG-specific")],
                    loc="upper right", frameon=False, handlelength=1.0, handleheight=1.0, borderaxespad=0.1,
                    labelspacing=0.3, handletextpad=0.5)
    ax_b = ax

    # titles (bold pair class + one statistic line), on the panel letters' line
    for x_let, ax_, sub in ((0.0, ax_a, SUB_BOX), (92.0, ax_b, SUB_BAR)):
        P.letter(x_let, 0.6, "a" if ax_ is ax_a else "b")
        P.text(x_let + 4.2, 0.6, "Unexpected pairs", fontweight="bold", ha="left", va="top")
        P.text(x_let + 4.2, 4.6, sub, ha="left", va="top")

    # ============================================================== c
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = K.px_per_mm(fig)
    row1_bottom = max((fig.bbox.y1 - t.get_window_extent(rend).y0) / k
                      for a_ in (ax_a, ax_b) for t in a_.get_xticklabels())
    C_LET = row1_bottom + 4.0
    PITCH, LABW, GAPL, AXW = 3.5, 21.0, 1.5, 18.5
    BX0 = [0.0, 45.5, 91.0, 136.5]
    C_TOP = C_LET + 9.4
    C_H = 6 * PITCH
    SHORT = {st: lab for c_, st, lab in SC.COLS}
    blocks = {b[0]: b for b in X.example_blocks()}
    order = [g for row in X.D_GRID for g in row]
    assert order == ["HMGB2", "CD19", "GZMM", "DNAH5"]
    assert max(len(blocks[g][6]) for g in order) == 6
    c_rows, label_texts, axes_c = [], [], []
    for j, gene in enumerate(order):
        gene, state, prg, lins, hit, sub, rows = blocks[gene]
        x0 = BX0[j]
        ax = P.ax(x0 + LABW + GAPL, C_TOP, AXW, C_H)
        ax.set_xlim(0, X.D_XMAX)
        ax.set_ylim(C_H, 0)
        ax.set_yticks([])
        S.frame(ax, spines=("bottom",))
        ax.set_xticks(X.D_TICKS)
        ax.set_xticklabels([f"{t:g}" for t in X.D_TICKS])
        axes_c.append(ax)
        g_t = P.text(x0, C_TOP - 2.4, gene, ha="left", va="center", fontstyle="italic")
        fig.canvas.draw()
        gw = g_t.get_window_extent(fig.canvas.get_renderer()).width / k
        ital = prg.replace(" (PU.1)", "").replace("-", "").isupper()
        p_t = P.text(x0 + gw + 1.8, C_TOP - 2.4, prg, ha="left", va="center",
                     fontstyle="italic" if ital else "normal")
        label_texts += [g_t, p_t]
        n = len(lins)
        bh = (1.85 / 2.37 * PITCH) / n                       # bar height 1.85 mm per 2.37 mm row, scaled to this pitch
        for ri, st in enumerate(rows):
            tgt = st == state
            yc = (ri + 0.5) * PITCH
            for li, lin in enumerate(lins):
                q = sub[(sub.lineage == lin) & (sub.state == st)]
                if q.empty:
                    continue
                v = float(q.iloc[0].mean_lognorm)
                assert v < X.D_XMAX
                ax.barh(yc + (li - (n - 1) / 2.0) * bh * 1.10, v, height=bh,
                        color=S.RECAST if tgt else G400, edgecolor="none", zorder=3)
                c_rows.append(dict(gene=gene, program=prg, ranked_subtype=state, row_subtype=st,
                                   row_label=SHORT.get(st, st), role="ranked subtype" if tgt else "sibling subtype",
                                   lineage=lin, bar_position="upper" if (n == 2 and li == 0) else
                                   ("lower" if n == 2 else "single"), mean_lognorm=v))
            label_texts.append(P.text(x0 + LABW, C_TOP + yc, SHORT.get(st, st), ha="right", va="center",
                                      color=S.BLACK if tgt else SUB))
    P.letter(0.0, C_LET - 0.4, "c")
    kx = 5.0
    for colr, lab in ((S.RECAST, "the subtype ranked for"), (G400, "its sibling subtypes")):
        fig.patches.append(plt.Rectangle((kx / W, 1 - (C_LET + 2.2 + 0.75) / P.H), 2.4 / W, 1.5 / P.H,
                                         transform=fig.transFigure, facecolor=colr, edgecolor="none"))
        t = P.text(kx + 3.2, C_LET + 2.2, lab, ha="left", va="center")
        fig.canvas.draw()
        kx += 3.2 + t.get_window_extent(fig.canvas.get_renderer()).width / k + 4.0
    xl_c = (BX0[0] + LABW + GAPL + BX0[3] + LABW + GAPL + AXW) / 2
    P.text(xl_c, C_TOP + C_H + 6.6, "mean expression, log(1 + CP10K)", ha="center", va="top")

    # ============================================================== page height and checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bottom = max((fig.bbox.y1 - t.get_window_extent(rend).y0) / k for t in fig.texts)
    H_new = bottom + 1.0
    checks["page_height_needed_mm"] = round(H_new, 2)
    for t in label_texts:                                      # every text inside its block
        x0t, x1t = t.get_window_extent(rend).x0 / k, t.get_window_extent(rend).x1 / k
        blk = max(i for i, b in enumerate(BX0) if b <= x0t + 0.01)
        assert x1t < (BX0[blk + 1] - 1.0 if blk < 3 else W - 0.5), (t.get_text(), x1t)
    bbl = leg.get_window_extent(rend)                          # b's key above every bar and count
    for p_ in ax_b.patches:
        assert not p_.get_window_extent(rend).overlaps(bbl)
    for t in ax_b.texts:
        assert not t.get_window_extent(rend).overlaps(bbl), t.get_text()
    sub_right = max(t.get_window_extent(rend).x1 / k for t in fig.texts if t.get_text() == SUB_BOX)
    assert sub_right < 92.0 - 2.0, sub_right
    checks.update(a_summary=dict(diff=round(float(su["diff"]), 4), lo=round(float(su.lo), 4), hi=round(float(su.hi), 4),
                                 p=round(float(su.p), 4), n_pos=int(su.n_pos), n_zero=int(su.n_zero),
                                 n_neg=int(su.n_neg)),
                  b_all_pairs=dict(recast=f"{int(sp.recast_n)}/{int(sp.n)}", cosg=f"{int(sp.cosg_n)}/{int(sp.n)}"),
                  b_by_comp={r.comp: f"{int(r.k_r)}/{int(r.k_c)} of {int(r.n)}" for r in BYC.itertuples()},
                  c_bars=len(c_rows))
    return P, H_new, checks


if __name__ == "__main__":
    P, H_new, _ = main(120.0)                             # first pass: measure the height the content needs
    plt.close(P.fig)
    S_H = round(H_new + 0.5, 1)
    P, H_chk, checks = main(S_H)
    assert abs(H_chk - H_new) < 0.05
    checks["page_mm"] = (W, S_H)
    P.save(f"{HERE}/out", STEM)
    print(STEM, checks)
