"""Supplementary Fig. 8: robustness of gene-program scoring to genes added to the curated panels.

  rows     the 19 scoring methods in four families; within a family, ordered by the median per-task change with 20
           added genes, least affected first; families follow their least affected member, so RECAST is the first row
  columns  5, 10, 20 and 40 added genes, on one shared x axis: change in balanced accuracy from the curated panels
           (dashed line: no change)
  points   the 28 tasks, each the mean over the two gene pools and five draws; vertical jitter only separates points;
           black bar = median over the 28 tasks
  marks    each method against RECAST in the per-task change, cluster bootstrap over the 11 parent atlases
           (RECAST minus the method; * P < 0.05, ** P < 0.01, *** P < 0.001, unadjusted)
Families (how the score is built, Supplementary Methods):
  attribution weighted   RECAST, Reference-free (weights from training cells; method id "SIGnature" in the tables)
  rank based             UCell, AUCell, singscore, JASMINE (LH, OR), ssGSEA, GSVA (within-cell gene ranks)
  control-gene based     ANS, Seurat, Seurat (AG), Seurat (LVG), Scanpy
  other label-free       mean expression, SCSE, combined z-score, VAM, decoupleR ULM

Run     python 01_draw.py
Output  out/S8_scoring_added_genes.{pdf,svg,png}
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

S.setup()
STEM = "S8_scoring_added_genes"
W = 180.0
METRIC = "balanced_acc"
DOSES = [5, 10, 20, 40]
# display names, and one colour per number of added genes (as in Fig. 6d)
NAMES = dict(RECAST="RECAST", SIGnature="Reference-free", ANS="ANS", Seurat="Seurat", Seurat_AG="Seurat (AG)",
             Seurat_LVG="Seurat (LVG)", Scanpy="Scanpy", UCell="UCell", Jasmine_LH="JASMINE (LH)",
             Jasmine_OR="JASMINE (OR)", expr_mean="Mean expr.", AUCell="AUCell", ssGSEA="ssGSEA", GSVA="GSVA",
             zscore="z-score", VAM="VAM", SCSE="SCSE", singscore="singscore", decoupleR_ULM="decoupleR ULM")
DOSE_COL = {5: "#7A6FAC", 10: "#612158", 20: "#A11035", 40: "#F6A800"}
FAMILY = {"Attribution weighted": ["RECAST", "SIGnature"],
          "Rank based": ["UCell", "AUCell", "singscore", "Jasmine_LH", "Jasmine_OR", "ssGSEA", "GSVA"],
          "Control-gene based": ["ANS", "Seurat", "Seurat_AG", "Seurat_LVG", "Scanpy"],
          "Other label-free": ["expr_mean", "SCSE", "zscore", "VAM", "decoupleR_ULM"]}
MINUS = "−"

RAW = pd.read_csv(f"{K.CODE}/shared_data/scoring_added_genes.csv")
STATS = pd.read_csv(f"{K.CODE}/shared_data/scoring_added_genes_stats.csv")
X = RAW[RAW.metric == METRIC]
assert set(X.tool) == set(NAMES) and sorted(X.dose.unique()) == [0] + DOSES and X.task.nunique() == 28
assert len(X) == 19 * 28 * 5 and (X[X.dose == 0].value == 0).all()
assert sorted(sum(FAMILY.values(), [])) == sorted(NAMES)
SIG = STATS[(STATS.metric == METRIC) & (STATS.comparison == "RECAST - comparator")]
assert len(SIG) == 18 * 4 and ((SIG["mean"] > 0) | (SIG.stars == "n.s.")).all()
MED20 = X[X.dose == 20].groupby("tool").value.median()


def order():
    """Families by their least affected member, methods within a family by median change at 20 added genes
    (ties broken by method name, as in Fig. 6d)."""
    fam = []
    for f, tools in FAMILY.items():
        ts = sorted(tools, key=lambda t: (-MED20[t], t))
        fam.append((f, ts))
    fam.sort(key=lambda ft: (-MED20[ft[1][0]], ft[0]))
    assert fam[0][1][0] == "RECAST"
    return fam


def main(H):
    P = S.Page(W, H)
    fig = P.fig
    k = K.px_per_mm(fig)
    fam = order()
    # ---------------------------------------------------------------- geometry (mm)
    LBL_R, X0, CW, STW, GAP = 27.0, 29.0, 31.0, 4.6, 2.2                     # label column | 4 x (cell + stars)
    RP, FGAP, TOP = 3.55, 1.1, 13.0                                          # row pitch, gap before a family, first row
    XL = (-0.62, 0.10)
    rows, y = [], TOP                                                        # (kind, key, y_centre_mm)
    for i, (f, ts) in enumerate(fam):
        if i:
            y += FGAP
        rows.append(("family", f, y + RP / 2))
        y += RP
        for t in ts:
            rows.append(("method", t, y + RP / 2))
            y += RP
    bottom = y
    cols = [X0 + j * (CW + STW + GAP) for j in range(4)]
    assert cols[-1] + CW + STW <= W - 0.3, cols[-1] + CW + STW
    rng = np.random.default_rng(1)                                           # display jitter only
    jit = {(t, d): rng.uniform(-0.30, 0.30, 28) for t in NAMES for d in DOSES}
    meds = {}
    for j, d in enumerate(DOSES):
        ax = P.ax(cols[j], TOP, CW, bottom - TOP)
        ax.set_xlim(*XL)
        ax.set_ylim(bottom, TOP)                                            # y in mm from the top, like the page
        for gx in (-0.6, -0.4, -0.2):
            ax.axvline(gx, color=S.HAIR, lw=S.HAIR_LW, zorder=0)
        S.refline(ax, x=0)
        for kind, key, yc in rows:
            if kind != "method":
                continue
            v = X[(X.tool == key) & (X.dose == d)].sort_values("task").value.to_numpy()
            assert len(v) == 28
            ax.scatter(v, yc + jit[(key, d)] * RP, s=3.2, color=DOSE_COL[d], alpha=0.6, linewidths=0, zorder=2)
            m = float(np.median(v))
            meds[(key, d)] = m
            ax.plot([m, m], [yc - 0.40 * RP, yc + 0.40 * RP], color=S.BLACK, lw=1.0, solid_capstyle="butt", zorder=3)
            if key != "RECAST":
                st = SIG[(SIG.comparator == key) & (SIG.dose == d)].stars.iloc[0]
                P.text(cols[j] + CW + 0.7, yc + 0.55, st, ha="left", va="center")
        S.frame(ax, spines=("bottom",))
        ax.set_yticks([])
        ax.set_xticks([-0.6, -0.4, -0.2, 0])
        ax.set_xticklabels([f"{MINUS}0.6", f"{MINUS}0.4", f"{MINUS}0.2", "0"])
        P.text(cols[j] + CW / 2, TOP - 1.4, str(d), ha="center", va="bottom")
    P.text((cols[0] + cols[-1] + CW) / 2, TOP - 6.2, "Genes added per curated panel", ha="center", va="bottom")
    P.text((cols[0] + cols[-1] + CW) / 2, bottom + 6.4, "Change in balanced accuracy",
           ha="center", va="top")
    for kind, key, yc in rows:
        if kind == "family":
            P.text(0.0, yc, key, ha="left", va="center", fontweight="bold")
        else:
            P.text(LBL_R, yc, NAMES[key], ha="right", va="center")
    # ---------------------------------------------------------------- checks
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lab_right = max(t.get_window_extent(rend).x1 for t in fig.texts if t.get_text() in NAMES.values()) / k
    assert lab_right < X0 - 0.8, lab_right
    fam_right = max(t.get_window_extent(rend).x1 for t in fig.texts if t.get_text() in FAMILY) / k
    assert fam_right < cols[0] + CW, fam_right                              # family headers stay in the empty row
    star_right = max(t.get_window_extent(rend).x1 for t in fig.texts if set(t.get_text()) == {"*"}) / k
    assert star_right < W - 0.3, star_right
    lo, hi = X[X.dose > 0].value.min(), X[X.dose > 0].value.max()
    assert XL[0] < lo and hi < XL[1], (lo, hi)
    needed = P.H - min(t.get_window_extent(rend).y0 for t in fig.texts) / k + 1.0
    order_flat = [t for _, ts in fam for t in ts]
    checks = dict(order=[NAMES[t] for t in order_flat],
                  families={f: [NAMES[t] for t in ts] for f, ts in fam},
                  median_k20={NAMES[t]: round(float(MED20[t]), 4) for t in order_flat},
                  stars_k20={NAMES[t]: SIG[(SIG.comparator == t) & (SIG.dose == 20)].stars.iloc[0]
                             for t in order_flat if t != "RECAST"},
                  value_range=[round(float(lo), 4), round(float(hi), 4)],
                  recast_vs_0=STATS[(STATS.metric == METRIC) & (STATS.comparison == "RECAST vs 0")]
                  [["dose", "mean", "ci_lo", "ci_hi", "p_boot"]].round(5).to_dict("records"))
    return P, needed, checks, meds, fam


if __name__ == "__main__":
    P, needed, _, _, _ = main(200.0)
    plt.close(P.fig)
    S_H = round(needed + 0.4, 1)
    P, needed2, checks, meds, fam = main(S_H)
    assert abs(needed2 - needed) < 0.05
    checks["page_mm"] = (W, S_H)
    P.save(f"{HERE}/out", STEM)
    print(STEM, checks)
