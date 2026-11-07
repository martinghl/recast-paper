"""Supplementary Fig. 3: the complete marker benchmark, all 56 methods on the 97 subtypes.

One row per method, grouped by method family; five box columns (marker recovery, one-vs-rest AUROC, marker rank,
selection time, contamination).  Boxes, whiskers, means, jittered subtype dots, the open circles (95% paired bootstrap
interval of RECAST minus the method includes zero) and the "#k" ranks come from the benchmark module shared with Fig. 3
(common/marker_benchmark_panels.py); this script lays them out at 8 pt on a 180-mm page.

Run     python 01_draw.py
Output  out/S3_marker_benchmark.{pdf,svg,png}
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
from matplotlib.transforms import blended_transform_factory   # noqa: E402
from matplotlib.ticker import NullLocator                     # noqa: E402

import marker_benchmark_panels as B      # noqa: E402  (reads the benchmark tables; defines functions only)

S.setup()                                 # the benchmark module set its own rcParams on import; restore ours
STEM = "S3_marker_benchmark"
BLOCKS = ("recall20", "auroc", "marker_rank200", "ms_per_cell", "contamination")

# ------------------------------------------------------------------ the tables
assert len(B.MASTER) == 56
for blk in ("recall20", "auroc", "marker_rank200", "contamination"):
    _n = B.DOTS[B.DOTS.block == blk].groupby('method').size()          # 97 subtypes; scMAGS gave no output for 3
    assert _n.drop('scmags').eq(97).all() and _n['scmags'] == 94, (blk, _n[_n != 97].to_dict())
assert B.DOTS[B.DOTS.block == "ms_per_cell"].groupby("method").size().eq(20).all()
assert int(B.DOTS[(B.DOTS.block == "auroc") & (B.DOTS.method == "cepo")].value.notna().sum()) == 94
RANKS = {k: B.recast_rank(k)[0] for k in BLOCKS}
assert RANKS == {"recall20": 1, "auroc": 5, "marker_rank200": 1, "ms_per_cell": 2, "contamination": 11}, RANKS
FAMILY_TITLE = {"Pseudobulk DE": "General-purpose DE"}   # family titles as printed


def name(mid):                            # display names; 'random' returns genes in alphabetical order
    return "Alphabetical control" if mid == "random" else B.field_name(mid)


TITLES = {"recall20": "Subtype-marker\nrecovery\n(20 selected genes)",
          "auroc": "One-vs-rest\nAUROC",
          "marker_rank200": "Marker-prioritization\nscore\n(first 200 genes)",
          "ms_per_cell": "Selection time\n(ms per cell, log)",
          "contamination": "Ribosomal or mito.\ngenes (first 200)"}
COLS = [dict(key="recall20", region="A"), dict(key="auroc", region="A"), dict(key="marker_rank200", region="A"),
        dict(key="ms_per_cell", region="B"), dict(key="contamination", region="B")]
ABBREV = ["abs, absolute-value ranking", "ov, overestimated variance", "det, detection-rate log FC",
          "tc, tie-corrected"]                      # longest first: they sit above the narrow first title line
KEY = [("dot", "subtype (time: lineage)"), ("dashed", "RECAST mean"), ("chance", "chance (AUROC 0.5)"),
       ("circle", "95% CI includes zero")]

# ------------------------------------------------------------------ geometry (mm)
W = 180.0
L_EDGE, LBL_W, GAP_LBL = 0.5, 36.6, 2.0
GUT = 6.0                                           # annotation strip right of every column ("#11" bold 8 pt = 4.6 mm)
INTRA, REGION_GAP, R_EDGE = GUT + 1.6, GUT + 5.0, GUT + 0.9
SLOT = 3.25                                         # row pitch
TOP, BOT = 14.0, 14.0

groups = B.field_groups()
mpos, hpos, _ = B.layout(groups)
assert len(mpos) == 56
dmin, dmax = min(mpos.values()), max(mpos.values())
ymin, ymax = dmin - 0.9, dmax + 0.9
PANEL_H = (ymax - ymin) * SLOT
H = TOP + PANEL_H + BOT

n_intra = sum(1 for i in range(1, len(COLS)) if COLS[i]["region"] == COLS[i - 1]["region"])
n_gap = len(COLS) - 1 - n_intra
x0 = L_EDGE + LBL_W + GAP_LBL
COL_W = (W - x0 - R_EDGE - n_intra * INTRA - n_gap * REGION_GAP) / len(COLS)
x = x0
for i, c in enumerate(COLS):
    if i:
        x += INTRA if c["region"] == COLS[i - 1]["region"] else REGION_GAP
    c["x"], c["w"] = x, COL_W
    c["annx"] = 1.0 + 0.5 * GUT / COL_W              # strip centre, in axes fraction
    x += COL_W
assert abs(x + R_EDGE - W) < 1e-9

P = S.Page(W, H)
fig = P.fig
checks = dict(page_mm=(W, round(H, 2)), col_w_mm=round(COL_W, 2), ranks=RANKS)

# ------------------------------------------------------------------ x-limits (draw()'s box-column rules)
for c in COLS:
    M = B.METRICS[c["key"]]
    lo_a, hi_a = M["axis"]
    v = B.DOTS[B.DOTS.block == c["key"]].value
    v = v[np.isfinite(v)]
    if M["scale"] == "log":
        vmin = float(v[v > 0].min())
        vmax = max(float(v.max()), float(B.MASTER[c["key"]].max()))
        lo_a = vmin / 1.6
        hi_a = vmax * (vmax / lo_a) ** 0.12
    elif hi_a != 1.0:
        hi_a = max(float(v.max()), float(B.MASTER[c["key"]].max()))
        lo_a = min(0.0, float(v.min()))
    c["lim"] = (lo_a, hi_a)

# ------------------------------------------------------------------ columns
ns_table = []
for c in COLS:
    M = B.METRICS[c["key"]]
    ax = P.ax(c["x"], TOP, c["w"], PANEL_H)
    c["ax"] = ax
    ax.set_ylim(ymax, ymin)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.BLACK)
        sp.set_linewidth(S.LW)
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    lo, hi = c["lim"]
    if M["scale"] == "log":
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        dec = [t for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000) if lo <= t <= hi]
        major = [t for t in dec if int(round(np.log10(t))) % 2 == 1]
        minor = [t for t in dec if t not in major]
        ax.set_xticks(major)
        ax.set_xticklabels([("%g" % t) for t in major])
        ax.set_xticks(minor, minor=True)
        ax.tick_params(axis="x", which="minor", length=S.TICK * 0.8, width=S.LW * 0.8)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        pad = 0.035 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        bt = M.get("box_ticks_narrow") or M["box_ticks"]          # 20 mm column: first / middle / last for rates
        tk = [t for t in bt if lo <= t <= hi]
        ax.set_xticks(tk)
        ax.set_xticklabels([("%g" % t) for t in tk])
    if M["ref"] is not None and lo <= M["ref"] <= hi:
        S.refline(ax, x=M["ref"], zorder=0)        # chance: black 0.5 pt 4/4 dashes
    fv = B.MASTER.loc["recast_M0", c["key"]]
    if lo <= fv <= hi:
        ax.axvline(fv, color=B.RECAST_D, lw=S.LW, ls=S.DASH, zorder=8)   # RECAST mean: 0.5 pt 4/4 dashes
    ax.set_title(TITLES[c["key"]], fontsize=S.FS, linespacing=1.0, pad=3)
    for mid, d in mpos.items():
        is_f = mid == "recast_M0"
        vals = B.vals_of(mid, c["key"])
        st = B.box_stats(vals)
        if st is None:
            continue
        ns = (not is_f) and B.is_ns(mid, c["key"])
        seed = zlib.crc32((mid + "|" + c["key"]).encode()) & 0xFFFFFFFF
        B._draw_box(ax, d, st, vals, is_f, ns, seed, higher=M["higher"], ann_x=c["annx"])
        ns_table.append((mid, c["key"], ns))
    r, _ = B.recast_rank(c["key"])
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    c["rank_txt"] = ax.text(c["annx"], mpos["recast_M0"], f"#{r}", transform=trans, ha="center", va="center",
                            fontsize=S.FS, fontweight="bold", color=B.RECAST_D, zorder=9, clip_on=False)

# ------------------------------------------------------------------ label column
lab = P.ax(L_EDGE, TOP, LBL_W, PANEL_H)
lab.set_ylim(ymax, ymin)
lab.set_xlim(0, 1)
lab.axis("off")
labels = []
for hname, d in hpos:
    if hname == "RECAST":
        continue
    labels.append(lab.text(0.0, d + 0.12, FAMILY_TITLE.get(hname, hname), ha="left", va="center", fontsize=S.FS, fontweight="bold",
                           color=B.INK))
for mid, d in mpos.items():
    is_f = mid == "recast_M0"
    labels.append(lab.text(1.0, d, name(mid), ha="right", va="center", fontsize=S.FS,
                           fontweight="bold" if is_f else "normal", color=B.RECAST_D if is_f else S.BLACK))
    if is_f:
        lab.plot([0.02, 0.02], [d - 0.34, d + 0.34], color=B.RECAST, lw=2.2, solid_capstyle="butt")

# ------------------------------------------------------------------ direction arrows + captions, keys
y_pb = TOP + PANEL_H                                # bottom of the box columns (mm from the top)
deco = fig.add_axes([0, 0, 1, 1])
deco.axis("off")
deco.set_xlim(0, W)
deco.set_ylim(H, 0)                                 # deco data units = mm from the top-left corner
for reg, higher in (("A", True), ("B", False)):
    cr = [c for c in COLS if c["region"] == reg]
    xa, xb = cr[0]["x"], cr[-1]["x"] + cr[-1]["w"]
    tail, head = (xa, xb) if higher else (xb, xa)
    deco.annotate("", xy=(head, y_pb + 7.3), xytext=(tail, y_pb + 7.3),
                  arrowprops=dict(arrowstyle="-|>", color=S.BLACK, lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=9))
    deco.text((xa + xb) / 2, y_pb + 10.9, ("Higher" if higher else "Lower") + " values are better",
              ha="center", va="center", fontsize=S.FS)
k_glyph, k_text, k_half = L_EDGE + 2.6, L_EDGE + 5.4, 2.1
key_texts = []
for i, (kind, txt) in enumerate(KEY):
    ky = y_pb + 2.0 + i * 3.2
    if kind == "dot":
        deco.scatter([k_glyph], [ky], s=3.0, color=B.G400, linewidths=0)
    elif kind == "dashed":
        deco.plot([k_glyph - k_half, k_glyph + k_half], [ky, ky], color=B.RECAST_D, lw=S.LW, ls=S.DASH)
    elif kind == "chance":
        deco.plot([k_glyph - k_half, k_glyph + k_half], [ky, ky], color=S.BLACK, lw=S.LW, ls=S.DASH)
    else:
        deco.scatter([k_glyph], [ky], s=13, facecolors="none", edgecolors=B.G900, linewidths=0.7)
    key_texts.append(deco.text(k_text, ky, txt, ha="left", va="center", fontsize=S.FS, color=S.BLACK))
abbr_texts = [deco.text(L_EDGE, 1.9 + i * 3.1, s, ha="left", va="center", fontsize=S.FS, color=S.BLACK)
              for i, s in enumerate(ABBREV)]

# ------------------------------------------------------------------ geometry checks
fig.canvas.draw()
rend = fig.canvas.get_renderer()
k = K.px_per_mm(fig)


def bb(t):
    return t.get_window_extent(rend)


first = COLS[0]["ax"]
x_first = bb(first).x0
tick_x0 = min(bb(t).x0 for t in first.get_xticklabels() if t.get_text())
for t in labels:                                     # labels stay left of the first column
    assert (x_first - bb(t).x1) / k > 1.5, t.get_text()
for t in key_texts:
    assert (min(x_first, tick_x0) - bb(t).x1) / k > 2.0, t.get_text()
    assert bb(t).y1 < bb(labels[-1]).y0 - 1.0 * k, t.get_text()          # below the last row label
titles = [bb(c["ax"].title) for c in COLS]
for t in abbr_texts:
    tb = bb(t)
    for tt in titles:
        if tb.y0 < tt.y1 and tt.y0 < tb.y1:
            assert (tt.x0 - tb.x1) / k > 1.5, (t.get_text(), (tt.x0 - tb.x1) / k)
    assert tb.y0 > max(bb(c["ax"]).y1 for c in COLS) + 0.8 * k, t.get_text()   # above the first row
for a, b_ in zip(titles, titles[1:]):
    assert (b_.x0 - a.x1) / k > 2.0, (a, b_)
for c in COLS:                                       # the rank glyph stays inside the strip and the page
    rb = bb(c["rank_txt"])
    assert rb.x0 > bb(c["ax"]).x1 + 0.3 * k and rb.x1 < fig.bbox.x1 - 0.3 * k, c["key"]
checks["abbrev_bottom_mm"] = round(H - bb(abbr_texts[-1]).y0 / k, 2)
checks["first_title_left_mm"] = round(min(t.x0 for t in titles) / k, 2)
checks["n_open_circles"] = int(sum(ns for _, _, ns in ns_table))
P.save(f"{HERE}/out", STEM)
print(STEM, checks)
