"""Fig. 2e: normalized projected encoder output as the top-ranked genes of the target are replaced by reference values.

One line per gene ranking (RECAST, the mean-expression difference, a random gene order): the mean over the 14 subtypes
of four lineages, with its s.e.m. as a band.  x = the fraction of all 28,231 input genes replaced (log axis, k = 0
left out); y = 0 at the reference profile, 1 at the target profile.  The AUC in the key is the trapezoid over the linear
fraction scale including k = 0.
Data: ../shared_data/deletion_insertion_{curves,auc}.csv (also used by Supplementary Fig. 1).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "common"))
import panel_style                                                            # noqa: E402,F401  (house style)
from method_colours import apply_style, mm, save_fig, RECAST, RECAST_D, DE, NULLC   # noqa: E402

C = pd.read_csv(os.path.join(CODE, "shared_data", "deletion_insertion_curves.csv"))
AU = pd.read_csv(os.path.join(CODE, "shared_data", "deletion_insertion_auc.csv"))
assert C.groupby(["lineage", "state"]).ngroups == 14 and C.lineage.nunique() == 4
assert AU.groupby("ranking").size().eq(14).all() and set(AU.ranking) == {"RECAST", "DE", "random"}
assert (C.k == 0).any() and abs(C.frac.max() - 1.0) < 1e-12
assert np.allclose(C[C.k > 0].frac * 28231, C[C.k > 0].k)          # fraction of all 28,231 input genes

STYLE = {"RECAST": (RECAST, "RECAST", 1.9), "DE": (DE, "Mean-expression\ndifference", 1.3),
         "random": (NULLC, "Random", 1.3)}
W, H = 44.2, 48.0                                   # mm: one slot of the four-panel row
RECT = [0.300, 0.205, 0.670, 0.740]
KEY_SHIFT = 0.45                                    # mm: the key starts this far right of the y axis


def draw(stem, outdir):
    aucm = AU.groupby("ranking")["del_auc"].mean()
    apply_style()
    fig = plt.figure(figsize=(mm(W), mm(H)))
    ax = fig.add_axes(RECT)
    for rk in ("random", "DE", "RECAST"):                          # RECAST drawn last (on top)
        g = C[(C.ranking == rk) & (C.k > 0)].groupby("k").agg(
            frac=("frac", "mean"), mu=("del_f", "mean"), n=("del_f", "size"),
            se=("del_f", lambda s: s.std(ddof=1) / np.sqrt(s.size))).reset_index()
        assert (g.n == 14).all()
        col, lab, lw = STYLE[rk]
        ax.fill_between(g["frac"], g["mu"] - g["se"], g["mu"] + g["se"], color=col, alpha=0.14, lw=0, zorder=1)
        ax.plot(g["frac"], g["mu"], color=col, lw=lw, zorder=3, label=f"{lab} (AUC {aucm[rk]:.2f})")
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"], fontsize=5.6)
    ax.minorticks_off()
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Fraction of input\ngenes replaced", fontsize=6.0, labelpad=1.8, linespacing=1.15)
    ax.set_ylabel("Normalized projected\nencoder output", fontsize=6.0, labelpad=1.6, linespacing=1.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.2, labelsize=5.6)
    leg = ax.legend(loc="lower left", fontsize=5.3, frameon=False, handlelength=0.8, handletextpad=0.4,
                    labelspacing=0.22, borderaxespad=0.0, borderpad=0.15,
                    bbox_to_anchor=(KEY_SHIFT / (RECT[2] * W), 0.0))
    for t in leg.get_texts():
        t.set_linespacing(1.08)                     # inside the two-line entry
        if t.get_text().startswith("RECAST"):
            t.set_color(RECAST_D)
            t.set_fontweight("bold")
    # the key must not sit on a curve, and no key text or handle may come within 0.3 mm of a s.e.m. band
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = leg.get_window_extent(rend)
    n_under = 0
    for ln in ax.get_lines():
        xy = ax.transData.transform(ln.get_xydata())
        n_under += int(((xy[:, 0] > bb.x0) & (xy[:, 0] < bb.x1) & (xy[:, 1] > bb.y0) & (xy[:, 1] < bb.y1)).sum())
    assert n_under == 0, f"{stem}: key covers {n_under} curve vertices"
    px = fig.dpi / 25.4
    bands = [ax.transData.transform_path(pth) for pc in ax.collections for pth in pc.get_paths()]
    items = [t.get_window_extent(rend) for t in leg.get_texts()] + [h.get_window_extent(rend) for h in leg.legend_handles]
    clear = 0.0
    while clear < 2.0 and not any(b.intersects_bbox(it.expanded(1, 1).padded(clear * px), filled=True)
                                  for b in bands for it in items):
        clear += 0.05
    sx, sy = ax.spines["left"].get_window_extent(rend), ax.spines["bottom"].get_window_extent(rend)
    dx = (min(it.x0 for it in items) - sx.x1) / px
    dy = (min(it.y0 for it in items) - sy.y1) / px
    assert dx >= 0.3 and dy >= 0.3, (round(dx, 2), round(dy, 2))
    assert clear - 0.05 >= 0.3, f"{stem}: key within {clear - 0.05:.2f} mm of a s.e.m. band"
    assert fig.get_tightbbox(rend).width * 25.4 <= W + 0.05, "panel e is wider than its slot"
    return save_fig(fig, stem, outdir)
