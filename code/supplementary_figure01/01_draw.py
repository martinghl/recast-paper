"""Supplementary Fig. 1: attribution faithfulness.

  a  insertion curves: the model output as the top-ranked genes are inserted, for RECAST, the mean-expression
     difference and a random gene order (shared_data/deletion_insertion_*.csv)
  b  loss of the initial assignment when the 20 selected genes are replaced in single cells, per subtype, RECAST
     against COSG and Wilcoxon (data/cell_replacement*.csv)
  c  paired differences with 95% bootstrap intervals (10,000 resamples of the 61 subtypes)

Run     python 01_draw.py
Output  out/S1_attribution_faithfulness.{pdf,svg,png}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import supplementary_style as S          # noqa: E402  (sets PANEL_STYLE_OFF=1 before any panel module is imported)
import supplementary_common as K         # noqa: E402

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

STEM = "S1_attribution_faithfulness"
W, H = 180.0, 64.5


def bootstrap_mean(x, n=10000, seed=0):
    """Mean of x and its percentile 95% interval from n bootstrap resamples of the subtypes."""
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------------------------------------------ data
C = pd.read_csv(f"{K.CODE}/shared_data/deletion_insertion_curves.csv")
AU = pd.read_csv(f"{K.CODE}/shared_data/deletion_insertion_auc.csv")
assert C.groupby(["lineage", "state"]).ngroups == 14 and C.lineage.nunique() == 4
assert AU.groupby("ranking").size().eq(14).all() and set(AU.ranking) == {"RECAST", "DE", "random"}
assert (C.k == 0).any() and abs(C.frac.max() - 1.0) < 1e-12
assert np.allclose(C[C.k > 0].frac * 28231, C[C.k > 0].k)

main = pd.read_csv(f"{HERE}/data/cell_replacement.csv")
units = pd.read_csv(f"{HERE}/data/cell_replacement_subtypes.csv")
eqd = pd.read_csv(f"{HERE}/data/cell_replacement_equal_shift.csv")
rep = main[main["mode"] == "replace"]
assert units.unit.nunique() == 61

# ------------------------------------------------------------------------------------------------ figure
P = S.Page(W, H)
fig = P.fig
AX_TOP, AX_BOT = 6.5, 54.0
checks = {}

# ---- a: insertion curves (normalized model output "ins_f")
STYLE = {"RECAST": (S.RECAST, "RECAST", 1.9),
         "DE": (S.DE, "Mean-expression\ndifference", 1.3),
         "random": (S.NULLC, "Random gene\norder", 1.3)}
aucm = AU.groupby("ranking")["ins_auc"].mean()
ax = P.ax(14.0, AX_TOP, 55.0, AX_BOT - AX_TOP)
S.frame(ax)
curves = []
for rk in ("random", "DE", "RECAST"):                               # RECAST drawn last (on top), as before
    g = C[(C.ranking == rk) & (C.k > 0)].groupby("k").agg(
        frac=("frac", "mean"), mu=("ins_f", "mean"), n=("ins_f", "size"),
        se=("ins_f", lambda s: s.std(ddof=1) / np.sqrt(s.size))).reset_index()
    assert (g.n == 14).all()
    col, lab, lw = STYLE[rk]
    ax.fill_between(g["frac"], g["mu"] - g["se"], g["mu"] + g["se"], color=col, alpha=0.14, lw=0, zorder=1)
    ax.plot(g["frac"], g["mu"], color=col, lw=lw, zorder=3, label=f"{lab} (AUC {aucm[rk]:.2f})")
    curves.append(g.assign(ranking=rk))
ax.set_xscale("log")
ax.set_xlim(2e-4, 1.0)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0])
ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
ax.minorticks_off()
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.set_xlabel("Fraction of input genes replaced")
ax.set_ylabel("Normalized projected\nencoder output", linespacing=1.1)
leg = ax.legend(loc="upper left", frameon=False, handlelength=1.3, handletextpad=0.45, labelspacing=0.35,
                borderaxespad=0.25, borderpad=0.0)
for t in leg.get_texts():
    t.set_linespacing(1.12)
    if t.get_text().startswith("RECAST"):
        t.set_color(S.RECAST_D)
        t.set_fontweight("bold")
items = list(leg.get_texts()) + list(leg.legend_handles)
checks["a_key_clearance_mm"] = K.clearance_mm(fig, ax, items)
assert checks["a_key_clearance_mm"] >= 0.3, checks
P.letter(0.5, 1.0, "a")

# ---- b: per-subtype loss of the initial assignment ("identity_loss")
MK = {"COSG": ("o", S.DE, S.DE), "Wilcoxon": ("s", "white", S.DE_L)}      # marker, face, edge
SB = 9.0                                                                   # marker area, pt^2
bw = 44.0
BC_TOP = AX_BOT - bw                          # b, c axes: 44 mm tall, bottoms aligned with a
axb = P.ax(81.0, BC_TOP, bw, bw)
S.frame(axb)
p = rep[rep.K == 20].pivot(index="unit", columns="method", values="identity_loss")
assert len(p) == 61
lo_ = min(0.0, float(p.min().min()))
mx = float(p.max().max()) * 1.05
axb.plot([lo_, mx], [lo_, mx], lw=S.LW, color=S.HAIR, zorder=1)
lowp = units.set_index("unit").n_anchored.reindex(p.index) < 100
for mth, (mk, fc, ec) in MK.items():
    for sel, al in ((~lowp, 1.0), (lowp, 0.35)):
        axb.scatter(p[mth][sel], p.RECAST[sel], s=SB, marker=mk, facecolors=fc, edgecolors=ec, lw=0.5, zorder=3,
                    alpha=al)
axb.set_xlim(lo_, mx)
axb.set_ylim(lo_, mx)
axb.set_xticks([0, 0.2, 0.4, 0.6])
axb.set_yticks([0, 0.2, 0.4, 0.6])
axb.set_xticklabels(["0", "0.2", "0.4", "0.6"])
axb.set_yticklabels(["0", "0.2", "0.4", "0.6"])
axb.set_xlabel("Comparator")
axb.set_ylabel("RECAST")
axb.set_title("Loss of initial assignment", pad=4)
cnt = {}
key_texts = [axb.text(0.98, 0.215, "above / below", ha="right", va="center", transform=axb.transAxes)]
for k, (mth, (mk, fc, ec)) in enumerate(MK.items()):
    a_, b_ = int((p.RECAST > p[mth]).sum()), int((p.RECAST < p[mth]).sum())
    cnt[mth] = (a_, b_)
    yy = 0.135 - 0.085 * k
    axb.scatter([0.40], [yy], s=SB, marker=mk, facecolors=fc, edgecolors=ec, lw=0.5, transform=axb.transAxes,
                zorder=4)
    key_texts.append(axb.text(0.45, yy, mth, va="center", transform=axb.transAxes))
    key_texts.append(axb.text(0.98, yy, f"{a_} / {b_}", va="center", ha="right", transform=axb.transAxes))
assert cnt == {"COSG": (52, 7), "Wilcoxon": (56, 4)}, cnt                 # subtypes above / below the diagonal
xy_all = np.c_[np.r_[p.COSG, p.Wilcoxon], np.r_[p.RECAST, p.RECAST]]
K.text_clear_of_points(fig, axb, key_texts, xy_all, pad_mm=0.4, what="S1b key")
checks["b_counts_above_below"] = cnt
P.letter(72.0, 1.0, "b")

# ---- c: paired differences with bootstrap intervals
axc = P.ax(148.0, BC_TOP, 28.5, bw)
S.frame(axc, spines=("bottom",))


def piv(df, v):
    return df.pivot(index="unit", columns="method", values=v)


z20 = main[(main["mode"] == "zero") & (main.K == 20)]
eqp = eqd.pivot(index="unit", columns="vs", values="a2_diff").assign(RECAST=0.0)
eqp[["COSG", "Wilcoxon"]] = -eqp[["COSG", "Wilcoxon"]]
tests = [("20 genes", piv(rep[rep.K == 20], "identity_loss")), ("5 genes", piv(rep[rep.K == 5], "identity_loss")),
         ("50 genes", piv(rep[rep.K == 50], "identity_loss")),
         ("Set to zero,\n20 genes", piv(z20, "identity_loss")), ("Equal mean\ninput change", eqp)]
YP = [0, 1.5, 2.5, 3.5, 4.5]
for i, (name, pv) in enumerate(tests):
    for j, cmp_ in enumerate(["COSG", "Wilcoxon"]):
        dd = (pv["RECAST"] - pv[cmp_]).sort_index().to_numpy()
        assert len(dd) == 61
        mu, lo_c, hi_c = bootstrap_mean(dd)
        mk, fc, ec = MK[cmp_]
        y = YP[i] + 0.2 * (1 if j == 0 else -1)
        axc.plot([lo_c, hi_c], [y, y], color=ec, lw=0.8, solid_capstyle="butt", zorder=2)
        axc.scatter([mu], [y], s=14, marker=mk, facecolors=fc, edgecolors=ec, lw=0.5, zorder=3)
S.refline(axc, x=0, ls="-", zorder=0.5)
axc.set_yticks(YP)
axc.set_yticklabels([t_[0] for t_ in tests], linespacing=1.12)
axc.invert_yaxis()
axc.set_ylim(5.05, -0.55)
axc.tick_params(axis="y", length=0, pad=2)
axc.set_xlim(-0.01, 0.13)
axc.set_xticks([0, 0.05, 0.1])
axc.set_xticklabels(["0", "0.05", "0.1"])
axc.set_xlabel("RECAST − comparator")
axc.get_yticklabels()[0].set_fontweight("bold")
axc.set_title("Loss of initial assignment", pad=4)
fig.canvas.draw()
rr = fig.canvas.get_renderer()
lab_x0 = min(t_.get_window_extent(rr).x0 for t_ in axc.get_yticklabels()) / K.px_per_mm(fig)
axb_x1 = axb.get_window_extent(rr).x1 / K.px_per_mm(fig)
checks["c_rowlabels_left_mm"] = round(lab_x0, 2)
assert lab_x0 > axb_x1 + 3.0, (lab_x0, axb_x1)
ax_bb = axc.get_window_extent(rr)
axc.plot([(lab_x0 * K.px_per_mm(fig) - ax_bb.x0) / ax_bb.width, 1.0], [0.75, 0.75], color=S.HAIR, lw=S.HAIR_LW,
         transform=axc.get_yaxis_transform(), clip_on=False, zorder=0.4)
P.letter(lab_x0 - 1.0, 1.0, "c")

P.save(f"{HERE}/out", STEM)
print(STEM, checks)
