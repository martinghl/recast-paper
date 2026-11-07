"""Supplementary Fig. 9: transfer of marker panels between atlases.

  a  summary table: the six methods with the highest overall score, then reference-free attribution below the line
     (it is one of the 52 methods of the broad-type comparison but not of the 51 of the fine-subtype comparisons; its
     overall score uses the same scaling, clipped to 0-1).  Tile colour = the value's position in its own comparison
     set (mean -/+ 2 s.d.; grey below, purple above); overall score = mean of the five endpoints after min-max scaling
     over each endpoint's full set (51 methods for the four fine-subtype endpoints, 52 for broad cell types)
  b  all 51 methods on the four fine-subtype endpoints, 42 directed comparisons (atlas pair x compartment) per box: IQR
     box, whiskers to 1.5 x IQR, median line, mean point, jittered comparisons, RECAST mean line, AUROC 0.5 line, open
     circle = paired bootstrap 95% interval of RECAST minus the method includes zero, "#k" = RECAST's rank by mean
RECAST ranks 5, 6, 8 and 14 of 51 on the fine-subtype endpoints, 4 of 52 on broad AUROC and 3 of 51 on the overall
score (asserted).  b uses the box-drawing code of common/marker_benchmark_panels.py with the transfer tables of
transfer_data.py, at the geometry of Supplementary Fig. 3.

Run     python 01_draw.py
Output  out/S9_cross_atlas_transfer.{pdf,svg,png}
"""
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S          # noqa: E402  (sets PANEL_STYLE_OFF=1 before the benchmark module is imported)
import supplementary_common as K         # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory        # noqa: E402

sys.path.insert(0, HERE)
import transfer_data as T                # noqa: E402
B = T.F2                                  # common/marker_benchmark_panels.py
S.setup()                                 # the benchmark module set its own rcParams on import; restore ours

STEM = "S9_cross_atlas_transfer"
IN_PW = f"{HERE}/data/transfer_pairs.csv"                    # per method, endpoint and comparison
IN_BASE = f"{HERE}/data/transfer_comparators.csv"            # broad cell types, comparator methods
IN_CUR = f"{HERE}/data/transfer_summary.csv"                 # broad cell types, RECAST and the remaining methods
IN_SIG = f"{HERE}/data/transfer_program_scoring.csv"         # reference-free attribution, all five endpoints

# ------------------------------------------------------------------ b: the 51 methods
d_tr, N_DUP = T.load_transfer()
_u = d_tr[(d_tr.method == "recast_M0") & (d_tr.metric == "auroc")]
assert _u.groupby("compartment").size().to_dict() == {"b": 6, "cd4": 12, "cd8": 12, "mono": 12}
assert not d_tr.transfer.isna().any()
assert T.TR_PUBLISHED == {"tr_sum_max": (0.725, 5), "tr_auroc": (0.895, 6), "tr_ontarget": (0.766, 8), "tr_knn": (None, 14)}
master, dots = T.transfer_tables(d_tr)
assert len(master) == 51 and "recast_M0" in master.index
assert B.family_of("recast_M0") == "RECAST" and B.field_name("recast_M0") == "RECAST" and B.field_name("random") == "Random"
assert list(dots.groupby(["method", "block"]).size().unique()) == [42]
B.MASTER, B.DOTS = master, dots           # the box-drawing helpers now read the transfer tables
B.METRICS.update(T.TR_METRICS)
BLOCKS = [k for k, _ in T.TR_BLOCKS]      # tr_sum_max, tr_auroc, tr_ontarget, tr_knn
RANKS = {}
for _k in BLOCKS:
    _rk, _n = B.recast_rank(_k)
    _pm, _pr = T.TR_PUBLISHED[_k]
    assert (_rk, _n) == (_pr, 51), (_k, _rk, _n)
    if _pm is not None:
        assert round(float(master.loc["recast_M0", _k]), 3) == _pm, _k
    RANKS[_k] = _rk
groups = T.transfer_groups(master)        # family order, mean-AUROC-descending inside a family
mpos, hpos, _ = B.layout(groups)
assert len(mpos) == 51


def name(mid):
    return B.field_name(mid)


# ------------------------------------------------------------------ a: summary table
piv = pd.read_csv(IN_PW)
piv = piv[piv.method != "seurat_roc"]          # degenerate ~6-gene method, dropped benchmark-wide


def pfield(metric):
    agg = piv[piv.metric == metric].groupby("method")["transfer"].mean()
    foc = float(agg["RECAST_M0"])
    rank = int((agg.values > foc).sum() + 1)
    return foc, rank, int(agg.size)


def pseries(metric):
    agg = piv[piv.metric == metric].groupby("method")["transfer"].mean()
    return agg.rename(index={"RECAST_M0": "RECAST"}).astype(float)


_base = pd.read_csv(IN_BASE).groupby("method")["transfer_auc"].mean()
_cur = pd.read_csv(IN_CUR).groupby("method")["transfer_auc"].mean()
comb = pd.concat([_base, _cur])
comb = comb[~comb.index.duplicated()].drop("seurat_roc", errors="ignore")
_IS_RECAST_CFG = lambda m: m.startswith("m0_")   # noqa: E731  RECAST's alternative settings, not other methods
comb = comb[[not _IS_RECAST_CFG(m) for m in comb.index]]
foc_b = float(comb["M0"])
others_b = comb.drop("M0").values.astype(float)
RANK_B, N_B = int((others_b > foc_b).sum() + 1), int(comb.size)
ENDPOINTS = ["sum_max", "auroc", "ontarget", "knn"]
for met, prk in zip(ENDPOINTS, (5, 6, 8, 14)):
    assert pfield(met)[1:] == (prk, 51), met
assert (RANK_B, N_B) == (4, 52)
field = pd.DataFrame({met: pseries(met) for met in ENDPOINTS})
field["broad"] = comb.rename(index={"M0": "RECAST"}).reindex(field.index)
assert len(field) == 51 and field.isna().sum().sum() == 0, "fine and broad rosters disagree"
lo, hi = field.min(), field.max()
lo["broad"], hi["broad"] = float(comb.min()), float(comb.max())      # the 52-method broad set
overall = ((field - lo) / (hi - lo)).mean(axis=1)
RANK_OVERALL = int((overall > overall["RECAST"]).sum() + 1)
assert RANK_OVERALL == 3
_sig = pd.read_csv(IN_SIG).set_index("track")
SIG_KEYS = {"sum_max": "fine_subtype_sum_max", "auroc": "fine_subtype_auroc", "ontarget": "fine_subtype_ontarget",
            "knn": "fine_subtype_knn", "broad": "broad_cross_atlas_auroc"}
_bidx = list(comb.drop("M0").index).index("native_direct")
assert others_b[_bidx] == float(comb["native_direct"])
assert float(_sig.loc["broad_cross_atlas_auroc", "value"]) == float(comb["native_direct"])
sig_row = pd.Series({k: float(_sig.loc[v, "value"]) for k, v in SIG_KEYS.items()})
sig_overall = float(((sig_row - lo) / (hi - lo)).clip(0, 1).mean())
TOP6 = overall.sort_values(ascending=False).index[:6].tolist()
assert "RECAST" in TOP6 and "random" not in TOP6
# a and b show the same fine-subtype means (b drops the duplicated rows of the alphabetical control)
for met, blk in zip(ENDPOINTS, BLOCKS):
    _f = field[met].rename(index={"RECAST": "recast_M0"})
    assert np.allclose(_f.loc[master.index].to_numpy(), master[blk].to_numpy(), rtol=0, atol=1e-12), met

A_COLS = ENDPOINTS + ["broad"]
A_ROWS = [(m, field.loc[m, A_COLS].astype(float), float(overall[m])) for m in TOP6] + [("SIGnature", sig_row[A_COLS], sig_overall)]   # "SIGnature" = reference-free attribution


def a_name(m):
    return "RECAST" if m == "RECAST" else ("Reference-free" if m == "SIGnature" else B.field_name(m))


# ------------------------------------------------------------------ display strings
TITLES = {"tr_sum_max": "Sum-max specificity\n(median subtype F1)", "tr_auroc": "Subtype\nAUROC",
          "tr_ontarget": "On-target\nrecovery", "tr_knn": "3-NN classification\n(median subtype F1)"}
A_TITLES = {"sum_max": "Sum-max", "auroc": "AUROC", "ontarget": "On-target", "knn": "3-NN", "broad": "AUROC"}
ABBREV = ["abs, absolute-value ranking", "ov, overestimated variance", "det, detection-rate log FC",
          "tc, tie-corrected"]
PURPLE, MID = "#AC78B4", "#F1F1F1"         # tile colours: light grey -> near-white -> purple
CMAP = LinearSegmentedColormap.from_list("field", [S.DE_L, MID, PURPLE])
SIG_D = "#4F6B49"                          # reference-free attribution (dark sage, as in Fig. 6)
G300 = "#BEBEBE"


def field_norm(vals):
    s = pd.Series(vals)
    return Normalize(vmin=s.mean() - 2.0 * s.std(), vmax=s.mean() + 2.0 * s.std())


NORMS = {c: field_norm(field[c].values) for c in ENDPOINTS}
NORMS["broad"] = field_norm(comb.values)                       # the 52-method broad set
NORMS["overall"] = field_norm(overall.values)

# ------------------------------------------------------------------ geometry (mm, from the top-left corner)
W = 180.0
L_EDGE, LBL_W, GAP_LBL = 0.5, 36.6, 2.0                       # b: label column as in S3
GUT = 6.0
INTRA, R_EDGE = GUT + 1.6, GUT + 0.9
SLOT = 3.05                                                    # row pitch, mm
CAP = 0.72 * S.FS * 25.4 / 72                                  # cap height of 8 pt Liberation Sans, mm
# a: name column | four fine-subtype columns | broad column | overall column | abbreviations
XN, XF, CWF, XB, CWB, XO, WO = 0.5, 36.0, 12.5, 88.5, 20.5, 111.5, 24.0
Y_G1, Y_G2, Y_BR, Y_T, Y_R = 2.25, 5.45, 7.55, 10.55, 12.35   # group lines, bracket, column titles, header rule
RH, N_A = 3.4, len(A_ROWS)
Y_BOT = Y_R + 0.4 + N_A * RH + 0.4                             # bottom rule of the table
GAP_AB = 3.0                                                   # table bottom rule -> b column titles
dmin, dmax = min(mpos.values()), max(mpos.values())
ymin, ymax = dmin - 0.9, dmax + 0.9
PANEL_H = (ymax - ymin) * SLOT

x0 = L_EDGE + LBL_W + GAP_LBL
COL_W = (W - x0 - R_EDGE - 3 * INTRA) / 4
COLS = []
x = x0
for i, k in enumerate(BLOCKS):
    if i:
        x += INTRA
    COLS.append(dict(key=k, x=x, w=COL_W, annx=1.0 + 0.5 * GUT / COL_W))
    x += COL_W
assert abs(x + R_EDGE - W) < 1e-9


def main(H, TOP_B):
    """Draw the page at height H (mm) with b's box columns starting at TOP_B (mm from the top).  Returns the page, the
    height the content needs, the measurements used to place b, the checks and the open-circle table."""
    P = S.Page(W, H)
    fig = P.fig
    deco = fig.add_axes([0, 0, 1, 1])
    deco.axis("off")
    deco.set_xlim(0, W)
    deco.set_ylim(H, 0)                                        # deco data units = mm from the top-left corner
    checks = dict(page_mm=(W, round(float(H), 2)), b_col_w_mm=round(COL_W, 2), ranks=RANKS, broad_rank=(RANK_B, N_B),
                  overall_rank=(RANK_OVERALL, 51), duplicate_random_rows_dropped=N_DUP)
    P.letter(0.0, 0.6, "a")
    P.letter(0.0, Y_BOT + GAP_AB, "b")

    # ============================================================== a  (summary table)
    a_txt, tiles, vals_txt = [], [], []

    def t(x_, y_, s, **kw):
        kw.setdefault("fontsize", S.FS)
        kw.setdefault("color", S.BLACK)
        h = deco.text(x_, y_, s, **kw)
        a_txt.append(h)
        return h

    fx0, fx1 = XF + 0.75, XF + 0.75 + 4 * CWF
    for (xa, xb), l1, l2 in (((fx0, fx1), "Fine-subtype transfer", "(51 methods)"),
                             ((XB, XB + CWB), "Broad", "(52 methods)")):
        t((xa + xb) / 2, Y_G1, l1, ha="center", va="center")
        t((xa + xb) / 2, Y_G2, l2, ha="center", va="center")
        deco.plot([xa + 0.4, xa + 0.4, xb - 0.4, xb - 0.4], [Y_BR + 0.6, Y_BR, Y_BR, Y_BR + 0.6], color=S.BLACK,
                  lw=S.LW, solid_capstyle="butt")
    colx = {c: fx0 + (j + 0.5) * CWF for j, c in enumerate(ENDPOINTS)}
    colx["broad"] = XB + CWB / 2
    t(XN, Y_T, "Method", ha="left", va="center")
    for c in A_COLS:
        t(colx[c], Y_T, A_TITLES[c], ha="center", va="center")
    t(XO + WO / 2 + 0.4, Y_T, "Overall score", ha="center", va="center")
    for yy in (Y_R, Y_BOT):
        deco.plot([XN, XO + WO], [yy, yy], color=S.BLACK, lw=S.LW, solid_capstyle="butt")
    for xx in (XF, XO):
        deco.plot([xx, xx], [Y_R, Y_BOT], color=S.BLACK, lw=S.LW, solid_capstyle="butt")
    y_first = Y_R + 0.4
    deco.plot([XN, XO + WO], [y_first + 6 * RH] * 2, color=S.BLACK, lw=S.LW, solid_capstyle="butt")  # reference-free below
    tw, th = CWF - 1.3, RH - 0.5
    bx0, bspan = XO + 0.8, WO - 1.6                            # the overall bar's 0-1 axis
    for i, (m, r, ov) in enumerate(A_ROWS):
        yc = y_first + (i + 0.5) * RH
        yb = yc + CAP / 2                                       # baseline that centres a cap-height glyph on the row
        kw = dict(fontweight="bold", color=S.RECAST_D) if m == "RECAST" else (
            dict(color=SIG_D) if m == "SIGnature" else {})
        t(XN, yb, a_name(m), ha="left", va="baseline", **kw)
        for c in A_COLS:
            w_ = tw if c != "broad" else CWB - 1.3
            rect = Rectangle((colx[c] - w_ / 2, yc - th / 2), w_, th, facecolor=CMAP(NORMS[c](float(r[c]))),
                             edgecolor=G300, linewidth=0.3, zorder=2)
            deco.add_patch(rect)
            tiles.append(rect)
            vals_txt.append(t(colx[c], yb, f"{float(r[c]):.2f}", ha="center", va="baseline", zorder=3))
        rect = Rectangle((bx0, yc - th / 2), bspan * ov, th, facecolor=CMAP(NORMS["overall"](ov)), edgecolor=G300,
                         linewidth=0.3, zorder=2)
        deco.add_patch(rect)
        tiles.append(rect)
        vals_txt.append(t(bx0 + bspan * ov / 2, yb, f"{ov:.2f}", ha="center", va="baseline", zorder=3))
    abbr = []
    for i, s in enumerate(ABBREV):
        abbr.append(t(0.0, Y_G1 + i * 3.15, s, ha="left", va="center"))

    # ============================================================== b  (51 methods x 4 endpoints)
    ns_table = []
    for c in COLS:
        M = B.METRICS[c["key"]]
        ax = P.ax(c["x"], TOP_B, c["w"], PANEL_H)
        c["ax"] = ax
        ax.set_ylim(ymax, ymin)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(S.BLACK)
            sp.set_linewidth(S.LW)
        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=S.TICK, width=S.LW, pad=S.TICK_PAD, labelsize=S.FS, colors=S.BLACK)
        lo_a, hi_a = M["axis"]
        assert (lo_a, hi_a) == (0.0, 1.0) and M["scale"] == "lin" and M["higher"]
        pad = 0.035 * (hi_a - lo_a)
        ax.set_xlim(lo_a - pad, hi_a + pad)
        tk = M["box_ticks"]                                     # 0, 0.2, ..., 1
        ax.set_xticks(tk)
        ax.set_xticklabels([("%g" % v) for v in tk])
        if M["ref"] is not None:
            S.refline(ax, x=M["ref"], zorder=0)                 # chance: black 0.5 pt, 4/4 dashes
        fv = float(master.loc["recast_M0", c["key"]])
        ax.axvline(fv, color=S.RECAST_D, lw=S.LW, ls=S.DASH, zorder=8)   # RECAST mean
        ax.set_title(TITLES[c["key"]], fontsize=S.FS, linespacing=1.0, pad=3)
        for mid, d in mpos.items():
            is_f = mid == "recast_M0"
            vals = B.vals_of(mid, c["key"])
            st = B.box_stats(vals)
            ns = (not is_f) and B.is_ns(mid, c["key"])
            seed = zlib.crc32((mid + "|" + c["key"]).encode()) & 0xFFFFFFFF
            B._draw_box(ax, d, st, vals, is_f, ns, seed, higher=True, ann_x=c["annx"])
            ns_table.append((mid, c["key"], ns))
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        c["rank_txt"] = ax.text(c["annx"], mpos["recast_M0"], f"#{RANKS[c['key']]}", transform=trans, ha="center",
                                va="center", fontsize=S.FS, fontweight="bold", color=S.RECAST_D, zorder=9, clip_on=False)
    lab = P.ax(L_EDGE, TOP_B, LBL_W, PANEL_H)
    lab.set_ylim(ymax, ymin)
    lab.set_xlim(0, 1)
    lab.axis("off")
    labels = []
    for hname, d in hpos:
        if hname == "RECAST":
            continue
        labels.append(lab.text(0.0, d + 0.12, {"Marker Scoring": "Marker scoring", "Pseudobulk DE": "General-purpose DE"}.get(hname, hname), ha="left", va="center", fontsize=S.FS, fontweight="bold",
                               color=S.BLACK))
    for mid, d in mpos.items():
        is_f = mid == "recast_M0"
        labels.append(lab.text(1.0, d, name(mid), ha="right", va="center", fontsize=S.FS,
                               fontweight="bold" if is_f else "normal", color=S.RECAST_D if is_f else S.BLACK))
        if is_f:
            lab.plot([0.02, 0.02], [d - 0.34, d + 0.34], color=S.RECAST, lw=2.2, solid_capstyle="butt")
    y_pb = TOP_B + PANEL_H
    xa, xb = COLS[0]["x"], COLS[-1]["x"] + COLS[-1]["w"]

    # ============================================================== measurements and geometry checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = K.px_per_mm(fig)

    def bb(a_):
        return a_.get_window_extent(rend)

    def y_mm(px):
        return H - px / k

    # a: abbreviations go to the right edge; they clear the table's right end
    ab_w = max(bb(s).width for s in abbr) / k
    for s in abbr:
        s.set_x(W - 0.4 - ab_w)
    fig.canvas.draw()
    assert W - 0.4 - ab_w - (XO + WO) > 2.0, f"abbreviations {W - 0.4 - ab_w - (XO + WO):.2f} mm from the table"
    # a: every value sits inside its tile / bar with >= 0.3 mm on each side (digits: cap height centred on the row,
    # no descenders, so the vertical fit holds by construction); titles and group labels do not touch
    assert th / 2 - CAP / 2 > 0.25
    for v_, r_ in zip(vals_txt, tiles):
        vb, rb = bb(v_), bb(r_)
        assert vb.x0 - rb.x0 > 0.3 * k and rb.x1 - vb.x1 > 0.3 * k, v_.get_text()
    hdr = sorted([s for s in a_txt if abs(s.get_position()[1] - Y_T) < 1e-9], key=lambda s: bb(s).x0)
    for a_, b_ in zip(hdr, hdr[1:]):
        assert (bb(b_).x0 - bb(a_).x1) / k > 0.8, (a_.get_text(), b_.get_text())
    names_ = [s for s in a_txt if s.get_ha() == "left" and abs(s.get_position()[0] - XN) < 1e-9]
    assert max(bb(s).x1 for s in names_) / k < XF - 0.8, "a method name reaches the first value column"
    for s in a_txt:
        if s.get_text() in ("Fine-subtype transfer", "(51 methods)", "Broad", "(52 methods)"):
            assert y_mm(bb(s).y0) < Y_BR - 0.3, s.get_text()
    # b: labels left of the first column; tick labels apart
    first = COLS[0]["ax"]
    x_first = bb(first).x0
    tick_boxes = [bb(tl) for c in COLS for tl in c["ax"].get_xticklabels() if tl.get_text()]
    for tl in labels:
        assert (x_first - bb(tl).x1) / k > 1.5, tl.get_text()
    for a_, b_ in zip(tick_boxes, tick_boxes[1:]):
        if abs(a_.y0 - b_.y0) < 1 and b_.x0 > a_.x0:
            assert (b_.x0 - a_.x1) / k > 1.0, "tick labels closer than 1 mm"
    titles = [bb(c["ax"].title) for c in COLS]
    for a_, b_ in zip(titles, titles[1:]):
        assert (b_.x0 - a_.x1) / k > 1.5, (a_, b_)
    for c in COLS:
        rb = bb(c["rank_txt"])
        assert rb.x0 > bb(c["ax"]).x1 + 0.3 * k and rb.x1 < fig.bbox.x1 - 0.3 * k, c["key"]
    meas = dict(b_title_top=round(float(min(y_mm(t_.y1) for t_ in titles)), 3))
    every = list(fig.texts) + list(deco.texts) + list(lab.texts)
    for c in COLS:
        every += [c["ax"].title] + list(c["ax"].texts) + [tl for tl in c["ax"].get_xticklabels() if tl.get_text()]
    needed = max(y_mm(bb(s).y0) for s in every if s.get_text())
    assert min(y_mm(bb(s).y1) for s in every if s.get_text()) > 0.0, "text above the page"
    checks["n_open_circles"] = int(sum(ns for _, _, ns in ns_table))
    checks["abbrev_gap_from_table_mm"] = round(float(W - 0.4 - ab_w - (XO + WO)), 2)
    return P, needed, meas, checks, ns_table


TOP_B0 = Y_BOT + GAP_AB + 8.0
P0, need0, m0, _, _ = main(300.0, TOP_B0)
plt.close(P0.fig)
TOP_B = TOP_B0 + (Y_BOT + GAP_AB - m0["b_title_top"])          # b's titles start GAP_AB below a's bottom rule
H = round(need0 + (TOP_B - TOP_B0) + 0.4, 1)
P, need, m1, checks, ns_table = main(H, TOP_B)
assert abs(m1["b_title_top"] - (Y_BOT + GAP_AB)) < 0.02, m1
assert abs(need - (H - 0.4)) < 0.06, (need, H)
checks.update(b_title_top_mm=m1["b_title_top"], top_b_mm=round(TOP_B, 3), a_bottom_rule_mm=round(Y_BOT, 2))
P.save(f"{HERE}/out", STEM)
print(STEM, checks)
