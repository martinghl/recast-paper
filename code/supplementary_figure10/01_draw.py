"""Supplementary Fig. 10: selection time on whole atlases.

  input   data/whole_atlas_time_all_methods.csv: 6 atlases x 51 methods = 306 rows of selection time in seconds;
          data/whole_atlas_time_recast.csv: RECAST's per-atlas timing (asserted equal)
  point   a method's median selection time over the six atlases; 49 comparators = the 51 minus RECAST (blue point)
          and "random" (the alphabetical control, not a selection method)
  RECAST  median of the six per-atlas times, each the median of 3 calls after a warm-up (H200 GPU)
The timing here (whole atlases) differs from the per-lineage selection-time column of Supplementary Fig. 3.

Run     python 01_draw.py
Output  out/S10_whole_atlas_time.{pdf,svg,png}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S                    # noqa: E402
import supplementary_common as K                   # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
import matplotlib.text as mtext           # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402

STEM = "S10_whole_atlas_time"
from method_colours import G300                 # noqa: E402  comparator grey
T = pd.read_csv(f"{HERE}/data/whole_atlas_time_all_methods.csv")
A = pd.read_csv(f"{HERE}/data/whole_atlas_time_recast.csv")
assert len(T) == 306 and T.dataset.nunique() == 6 and T.groupby("method").size().eq(6).all() and T.method.nunique() == 51
assert (A.n_rep == 3).all() and set(A.dataset) == set(T.dataset) and (A.gpu == "NVIDIA H200 NVL").all()
_r = T[T.method == "recast"].set_index("dataset").sel_s
assert (np.abs(_r - A.set_index("dataset").select_s.reindex(_r.index)) < 1e-4).all()
MED = T[~T.method.isin(["recast", "random"])].groupby("method").sel_s.median()
assert len(MED) == 49
FOC = float(_r.median())
NAMES = {"cepo": "Cepo", "cosg": "COSG", "edger_ql": "edgeR (QL)", "edger_glm": "edgeR (GLM)",
         "limma_trend": "limma (trend)", "scanpy_t_rankby_raw": "Scanpy t-test",
         "scanpy_tover_rankby_raw": "Scanpy t-test (overestimated variance)", "seurat_negbinom": "Seurat negbinom",
         "seurat_LR": "Seurat LR"}

W, H = 180.0, 45.0
AX_X, AX_W, AX_TOP, AX_H = 6.0, 168.0, 10.0, 19.0
XL = (0.25, 3e5)
S_CMP, S_REC = 12, 50                      # marker areas (pt^2); comparator diameter sqrt(12) pt = 1.22 mm
D_MM = 1.45                                # swarm: minimum centre distance (diameter + clearance)


def swarm(xmm, step_mm):
    """deterministic beeswarm: each point, in x order, takes the free row nearest the centre"""
    ys = np.zeros(len(xmm))
    placed = []
    for i in np.argsort(xmm):
        for k in range(0, 40):
            for cand in ((0.0,) if k == 0 else (k * step_mm, -k * step_mm)):
                if all((xmm[i] - xmm[j]) ** 2 + (cand - ys[j]) ** 2 >= D_MM ** 2 for j in placed):
                    ys[i] = cand
                    break
            else:
                continue
            break
        placed.append(i)
    return ys


P = S.Page(W, H)
fig = P.fig
ax = P.ax(AX_X, AX_TOP, AX_W, AX_H)
S.frame(ax, spines=("bottom",))
ax.set_xscale("log")
ax.set_xlim(*XL)
lx = (np.log10(MED.values) - np.log10(XL[0])) / (np.log10(XL[1]) - np.log10(XL[0])) * AX_W
ymm = swarm(lx, 0.75)
half_mm = AX_H / 2
assert np.abs(ymm).max() < half_mm - 1.5, "swarm taller than the axes"
yv = ymm / half_mm
ax.scatter(MED.values, yv, s=S_CMP, color=G300, edgecolors="white",
           linewidths=0.25, zorder=2)
ax.scatter([FOC], [0], s=S_REC, color=S.RECAST, edgecolors="white", linewidths=0.6, zorder=6)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ticks = [1, 10, 100, 1000, 10000, 100000]
ax.set_xticks(ticks)
ax.set_xticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
ax.minorticks_off()
ax.set_xlabel("Median selection time per atlas (s, logarithmic scale)")
labs = [ax.annotate(f"RECAST\n{FOC:.1f} s", (FOC, 0), xytext=(-9, 0), textcoords="offset points", color=S.RECAST_D,
                    fontweight="bold", ha="right", va="center", linespacing=1.1,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=S.RECAST_D, shrinkA=1, shrinkB=3.5))]
fast, slow = MED.idxmin(), MED.idxmax()
for key, tag in ((fast, "fastest comparator"), (slow, "slowest comparator")):
    i = list(MED.index).index(key)
    labs.append(ax.annotate(f"{NAMES.get(key, key)}, {MED[key]:,.{1 if MED[key] < 100 else 0}f} s\n({tag})",
                            (MED[key], yv[i]), xytext=(MED[key], 1.12), textcoords="data", ha="center", va="bottom",
                            linespacing=1.1, annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color=S.DE_L, shrinkA=1, shrinkB=2.5)))
leg = fig.legend(handles=[Line2D([0], [0], marker="o", ls="none", mfc=S.RECAST, mec="white", ms=np.sqrt(S_REC),
                                 label="RECAST"),
                          Line2D([0], [0], marker="o", ls="none", mfc=G300,
                                 mec="white", ms=np.sqrt(S_CMP), label=f"Comparator method (n = {len(MED)})")],
                 loc="center", bbox_to_anchor=(0.5, 1 - (H - 3.4) / H), ncol=2, frameon=False,
                 handletextpad=0.3, columnspacing=2.0, borderaxespad=0, borderpad=0)

# ------------------------------------------------------------------ checks: labels clear of the points
fig.canvas.draw()
rend = fig.canvas.get_renderer()
k = K.px_per_mm(fig)
pts = ax.transData.transform(np.c_[np.r_[MED.values, FOC], np.r_[yv, 0]])
r_px = 1.1 * k
for t in labs:
    bb = mtext.Text.get_window_extent(t, rend).padded(r_px)
    n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
    assert n == 0, f"{t.get_text()!r} covers {n} points"
    assert fig.bbox.x0 + 0.5 * k < bb.x0 and bb.x1 < fig.bbox.x1 - 0.5 * k and bb.y1 < fig.bbox.y1 - 0.3 * k, \
        f"{t.get_text()!r} off the canvas"
lb = leg.get_window_extent(rend)
assert lb.y0 > fig.bbox.y0 and lb.y1 < ax.xaxis.label.get_window_extent(rend).y0 - 1.0 * k
checks = dict(recast_median_s=round(FOC, 3), per_atlas_recast={d: round(v, 3) for d, v in _r.items()},
              n_comparators=len(MED), fastest=(fast, round(float(MED[fast]), 2)), slowest=(slow, round(float(MED[slow]), 0)),
              swarm_max_mm=round(float(np.abs(ymm).max()), 2))
P.save(f"{HERE}/out", STEM)
print(STEM, checks)
