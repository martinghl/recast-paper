"""Supplementary Fig. 2: encoder consistency and model dependence.

  a  cosine similarity between the encoded-profile displacement and the embedding-centroid displacement
  b  projected profile displacement relative to the centroid separation
  c  relative completeness error at 50 integration steps
  d  error at 50, 200 and 1,000 steps for the eight SCimilarity sibling comparisons with the largest 50-step error
  e  mean subtype-classification accuracy against mean marker-recovery gain, for five encoder configurations
In a-c each group has its own n (printed); open symbols are the four Baron pancreas control states, black bars medians.

Run     python 01_draw.py
Output  out/S2_encoder_consistency.{pdf,svg,png}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S                    # noqa: E402
import supplementary_common as K                   # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from scipy import stats                   # noqa: E402
import matplotlib.text as mtext           # noqa: E402

STEM = "S2_encoder_consistency"
W, H = 180.0, 117.0

# ------------------------------------------------------------------------- a-d data
D = pd.read_csv(f"{HERE}/data/profile_centroid_alignment.csv")
ST = pd.read_csv(f"{HERE}/data/profile_centroid_alignment_steps.csv")
GROUPS = [("SCimilarity", (D.cohort == "fine65") & (D.encoder == "SCimilarity"), S.RECAST),
          ("SCimilarity reference\nvariants", (D.cohort == "zhaoref"), S.RECAST_L),
          ("scTab_SSL", (D.cohort == "fine65") & (D.encoder == "scTab_SSL"), S.DE),
          ("PBMC_SSL", (D.cohort == "fine65") & (D.encoder == "PBMC_SSL"), S.DE_L),
          ("scVI", (D.cohort == "fine65") & (D.encoder == "scVI"), S.REF_FREE),
          ("scFoundation", (D.cohort == "crossfm6") & (D.encoder == "scFoundation"), S.BIO[0])]
NS = []
for name, m, _ in GROUPS:
    sub = D[m]
    if name.startswith("SCimilarity reference"):
        assert len(sub) == 112 and sub.target.nunique() == 16 and sub.groupby("target").size().eq(7).all()
    elif name == "scFoundation":
        assert len(sub) == 24 and sub.lineage.nunique() == 6
    else:
        assert len(sub) == 74 and sub.lineage.nunique() == 21 and (sub.tier == "control").sum() == 4
        assert sub[sub.tier == "control"].lineage.str.contains("baron").all()
    NS.append(len(sub))
assert sum(m.sum() for _, m, _ in GROUPS) == len(D) and NS == [74, 112, 74, 74, 74, 24]
sib = D[GROUPS[0][1]]
top8 = set(map(tuple, sib.nlargest(8, "completeness_rel")[["lineage", "target"]].values))
assert set(map(tuple, ST[["lineage", "target"]].drop_duplicates().values)) == top8 and len(top8) == 8
assert (ST.encoder == "SCimilarity").all() and sorted(ST.n_steps.unique()) == [50, 200, 1000]

# ------------------------------------------------------------------------- e data
KNN = pd.read_csv(f"{HERE}/data/encoder_knn_macro_f1.csv")
REC = pd.read_csv(f"{HERE}/data/encoder_marker_recall.csv")
JT = pd.read_csv(f"{HERE}/data/encoder_summary.csv").set_index("FM")
WC = pd.read_csv(f"{HERE}/data/encoder_within_lineage_correlation.csv").set_index("lineage")
ENC = ["SCimilarity", "scTab_SSL", "PBMC_SSL", "scVI", "scFoundation"]
LIN = {"zhao_CD8": "Zhao CD8 T cells", "zhao_B": "Zhao B cells",
       "dom_helperT": "Domínguez-Condé helper T cells", "dom_cytoT": "Domínguez-Condé cytotoxic T cells",
       "pbmcbmn_Bmat": "pbmcbmn B cells (maturation)", "yoshida_Bsw": "Yoshida B cells (class switching)"}
assert (REC.k == 20).all() and set(REC.encoder) == set(ENC) and set(KNN.FM) == set(ENC)
assert KNN.groupby("FM").size().eq(6).all() and REC.groupby("encoder").size().eq(6).all()
assert set(KNN.lineage) == set(LIN) == set(REC.lineage)
assert np.allclose(REC.delta, REC.RECAST - REC.expression)
ACC = KNN.groupby("FM").state_knn_acc.mean()
GAIN = REC.groupby("encoder").delta.mean()
for e in ENC:
    assert abs(ACC[e] - JT.loc[e, "sep_fine_knn"]) <= 5e-4 + 1e-9
    assert abs(GAIN[e] - JT.loc[e, "RECAST_delta_mean"]) <= 5e-4 + 1e-9
RLIN = {}
for lin in LIN:
    m = KNN[KNN.lineage == lin][["FM", "state_knn_acc"]].merge(
        REC[REC.lineage == lin][["encoder", "delta"]].rename(columns={"encoder": "FM"}), on="FM")
    assert len(m) == 5
    RLIN[lin] = stats.pearsonr(m.state_knn_acc, m.delta)[0]
    assert abs(RLIN[lin] - WC.loc[lin, "pearson"]) <= 5e-4 + 1e-9 and WC.loc[lin, "n"] == 5

# ------------------------------------------------------------------------- figure
P = S.Page(W, H)
fig = P.fig
rng = np.random.default_rng(0)              # one jitter stream, consumed in the order a, b, c
checks = {}
R1_TOP, R1_H = 5.0, 40.0
LOGT = [1e-5, 1e-3, 1e-1, 1e1]
LOGL = ["0.00001", "0.001", "0.1", "10"]


def hstrip(ax, col, xlab, dashed=None, zero=False, log=False):
    S.frame(ax)
    for i, (name, mask, colr) in enumerate(GROUPS):
        sub = D[mask]
        v = sub[col].to_numpy(dtype=float)
        y = i + rng.uniform(-0.22, 0.22, size=len(v))
        hollow = (sub.tier == "control").to_numpy()
        ax.scatter(v[~hollow], y[~hollow], s=7.0, color=colr, alpha=0.75, linewidths=0, zorder=3)
        if hollow.any():
            ax.scatter(v[hollow], y[hollow], s=9.0, facecolors="none", edgecolors=colr, linewidths=0.6, zorder=3)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.34, i + 0.34], color=S.BLACK, lw=1.1, zorder=4, solid_capstyle="butt")
    if dashed is not None:
        S.refline(ax, x=dashed, zorder=1)
    if zero:
        S.refline(ax, x=0, ls="-", zorder=0.5)
    ax.set_ylim(len(GROUPS) - 0.5, -0.5)
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab, linespacing=1.12)
    if log:
        ax.set_xscale("log")


axa = P.ax(38.0, R1_TOP, 36.0, R1_H)
hstrip(axa, "cos_align", "Cosine similarity, profile vs\ncentroid displacement", dashed=0.9, zero=True)
axa.set_xlim(min(0.7, D.cos_align.min() - 0.02), 1.005)
axa.set_xticks([0, 0.5, 1.0])
axa.set_xticklabels(["0", "0.5", "1"])
axb = P.ax(84.0, R1_TOP, 36.0, R1_H)
hstrip(axb, "ratio", "Projected profile displacement /\ncentroid separation", dashed=1.0, zero=True)
axb.set_xlim(min(0.0, D.ratio.min() - 0.05), max(2.0, D.ratio.max() * 1.05))
axb.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axb.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
axc = P.ax(130.0, R1_TOP, 46.0, R1_H)
hstrip(axc, "completeness_rel", "Relative completeness error\n(50 steps)", log=True)
lo = D.completeness_rel[D.completeness_rel > 0].min()
assert lo > 0
axc.set_xlim(lo * 0.5, max(0.1, D.completeness_rel.max() * 2))
assert axc.get_xlim()[0] < LOGT[0] and axc.get_xlim()[1] > LOGT[-1]
axc.set_xticks(LOGT)
axc.set_xticklabels(LOGL)
axc.minorticks_off()
# shared group labels and the n column, at the rows of a (and of b, c: same y transform)
lab_texts = []
for i, (name, _, colr) in enumerate(GROUPS):
    yd = axa.transData.transform((0, i))[1]
    y_mm = H - yd / K.px_per_mm(fig)
    lab_texts.append(P.text(28.5, y_mm, name, ha="right", va="center", linespacing=1.05))
    P.text(33.0, y_mm, f"{NS[i]}", ha="center", va="center")
P.text(33.0, R1_TOP - 1.2, "n", ha="center", va="bottom", fontstyle="italic")
P.letter(0.5, 1.0, "a")
P.letter(79.0, 1.0, "b")
P.letter(125.0, 1.0, "c")

# ---- d: integration-step convergence
R2_TOP = 70.0
axd = P.ax(17.0, R2_TOP, 57.0, 36.0)          # right edge = right edge of a
S.frame(axd)
for _, grp in ST.groupby(["lineage", "target"]):
    grp = grp.sort_values("n_steps")
    axd.plot(grp.n_steps, grp.completeness_rel, color=S.RECAST, lw=0.8, marker="o", ms=2.6, alpha=0.85)
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xticks([50, 200, 1000])
axd.set_xticklabels(["50", "200", "1,000"])
axd.minorticks_off()
axd.set_xlim(38, 1400)
axd.set_xlabel("Integration steps")
axd.set_ylabel("Relative completeness error")
axd.set_ylim(axc.get_xlim())
axd.set_yticks(LOGT)
axd.set_yticklabels(LOGL)
axd.minorticks_off()
axd.set_title("Eight largest errors at 50 steps", pad=4)
P.letter(0.5, R2_TOP - 7.5, "d")

# ---- e: accuracy vs gain
axe = P.ax(96.0, R2_TOP, 80.0, 36.0)
S.frame(axe)
S.refline(axe, y=0, zorder=1)
for e in ENC:
    isc = e == "SCimilarity"
    axe.scatter(ACC[e], GAIN[e], s=60 if isc else 34, color=S.RECAST if isc else S.REF_FREE,
                edgecolors="white", linewidths=0.6, zorder=4)
lab = {"SCimilarity": ((0, 10), "center", False), "scFoundation": ((0, 10), "center", False),
       "scTab_SSL": ((-14, 0), "right", True), "PBMC_SSL": ((16, -3), "left", True),
       "scVI": ((16, -9), "left", True)}
texts = []
for e in ENC:
    xy, ha, arr = lab[e]
    ap = dict(arrowstyle="-", lw=0.4, color=S.DE_L, shrinkA=1, shrinkB=2.5) if arr else None
    texts.append(axe.annotate(e, (ACC[e], GAIN[e]), xytext=xy, textcoords="offset points",
                              color=S.RECAST_D if e == "SCimilarity" else S.BLACK,
                              fontweight="bold" if e == "SCimilarity" else "normal", ha=ha, va="center",
                              arrowprops=ap))
axe.set_xlim(0.755, 0.915)
axe.set_ylim(-0.46, 0.31)                       # top 0.27 -> 0.31: room for the 8 pt label above SCimilarity
axe.set_xticks([0.76, 0.80, 0.84, 0.88])
axe.set_xticklabels(["0.76", "0.80", "0.84", "0.88"])
axe.set_yticks([-0.4, -0.2, 0, 0.2])
axe.set_yticklabels(["−0.4", "−0.2", "0", "0.2"])
axe.set_xlabel("Mean subtype-classification accuracy")
axe.set_ylabel("Mean marker-recovery gain\n(20 selected genes)", linespacing=1.12)
P.letter(79.0, R2_TOP - 7.5, "e")

# ---- checks: labels clear of the data
fig.canvas.draw()
rend = fig.canvas.get_renderer()
pts = axe.transData.transform(np.c_[[ACC[e] for e in ENC], [GAIN[e] for e in ENC]])
r_px = np.sqrt(60) / 2 * fig.dpi / 72
ab = axe.get_window_extent(rend)
for t in texts:
    bb = mtext.Text.get_window_extent(t, rend).padded(r_px)
    n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
    assert n == 0, f"label {t.get_text()!r} covers {n} encoder points"
    assert ab.x0 < bb.x0 and bb.x1 < ab.x1 + 4 and ab.y0 < bb.y0 and bb.y1 < ab.y1 + 4, t.get_text()
for i, a in enumerate(texts):
    for b in texts[i + 1:]:
        assert not mtext.Text.get_window_extent(a, rend).overlaps(mtext.Text.get_window_extent(b, rend))
for left, right in ((axa, axb), (axb, axc)):          # x labels of neighbouring strips must not touch
    gap = (right.xaxis.label.get_window_extent(rend).x0 - left.xaxis.label.get_window_extent(rend).x1) \
        / K.px_per_mm(fig)
    assert gap > 1.5, gap
x_lab_right = max(t.get_window_extent(rend).x1 for t in lab_texts) / K.px_per_mm(fig)
checks["group_labels_right_mm"] = round(x_lab_right, 2)
checks["medians_cos_ratio"] = {g[0].replace("\n", " "): (round(float(D[g[1]].cos_align.median()), 3),
                                                        round(float(D[g[1]].ratio.median()), 3)) for g in GROUPS}
checks["encoder_means_acc_gain"] = {e: (round(ACC[e], 3), round(GAIN[e], 3)) for e in ENC}
P.save(f"{HERE}/out", STEM)
print(STEM, checks)
