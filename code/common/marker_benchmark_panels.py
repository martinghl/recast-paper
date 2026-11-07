#!/usr/bin/env python
"""The marker-selection benchmark as columns of box plots, one row per method (Fig. 3, Supplementary Figs. 3 and 9).

Methods are grouped by algorithmic family (RECAST, dedicated panel selection, marker scoring, single-cell DE,
general-purpose DE, control), each metric in its own compact column.  A box summarises the matched subtypes
(box = interquartile range, centre line = median, whiskers = 1.5 x IQR, point = mean, dots = subtypes); RECAST's box is
blue, every other method grey.  Beside each column: a hollow circle where a method cannot be told apart from RECAST
(is_ns), and RECAST's rank over the whole field.

Data: ../shared_data/marker_benchmark_summary.csv (mean per method and metric) and marker_benchmark_by_subtype.csv
(one value per method, metric and subtype; `unit` pairs the methods).  recast_M0 is RECAST's id in these tables.
"""
import os, sys, zlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from method_colours import (apply_style, mm, save_fig, INK, SUB, RECAST, RECAST_D,
                      G900, G700, G550, G400, G300, G200, G100)

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the code/ folder
F2   = f"{CODE}/shared_data"
OUT  = os.path.join(os.getcwd(), "out")                              # default output folder of save_fig
apply_style()
RNG = np.random.default_rng(20260823)

# ----------------------------------------------------------------------------- data
# One value per subtype for recovery, AUROC, marker rank and contamination; selection time is per lineage (one run
# per lineage).  RECAST's marker rank comes from its full attribution ranking, as every other method's.
MASTER = pd.read_csv(f"{F2}/marker_benchmark_summary.csv").set_index("method")      # mean per method and endpoint
DOTS   = pd.read_csv(f"{F2}/marker_benchmark_by_subtype.csv")   # cols: method, block, unit, value  (unit=lineage|state, or lineage for time)

def vals_of(method, block):
    d = DOTS[(DOTS.method == method) & (DOTS.block == block)]
    return d.value.values

def lin_map(method, block):
    d = DOTS[(DOTS.method == method) & (DOTS.block == block)]
    return dict(zip(d.unit, d.value))

def box_stats(vals):
    """Tukey box-plot statistics from lineage-level values."""
    v = np.sort(np.asarray(vals, float)); v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lofence, hifence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lofence) & (v <= hifence)]
    whislo = inside.min() if len(inside) else v.min()
    whishi = inside.max() if len(inside) else v.max()
    fliers = v[(v < lofence) | (v > hifence)]
    return dict(q1=q1, med=med, q3=q3, whislo=whislo, whishi=whishi,
                mean=float(v.mean()), fliers=fliers, n=len(v))

_BOOT_IDX = {}
def _boot_idx(n_units, B=10000, seed=0):
    """One shared resample-index matrix per unit count, so every method on a block is resampled on
    the same draws (joint paired bootstrap)."""
    key = (n_units, B, seed)
    if key not in _BOOT_IDX:
        _BOOT_IDX[key] = np.random.default_rng(seed).integers(0, n_units, size=(B, n_units))
    return _BOOT_IDX[key]

def is_ns(method, block, B=10000, seed=0):
    """True if the method is NOT distinguishable from RECAST on this metric: the two-sided 95%
    percentile interval of the paired bootstrap of (RECAST - method) over the matched units includes
    zero. The settings are those of the method comparison in the Methods: B = 10,000 resamples,
    numpy default_rng(0), units = the subtypes both methods were scored on (97), and the lineages
    for selection time.  A log-scaled metric is resampled on log10 -- the scale its axis
    draws -- because a raw-ms difference over a handful of heavy-tailed lineages is dominated by the
    largest atlas. Fewer than three matched units: not tested (no circle)."""
    fm, cm = lin_map("recast_M0", block), lin_map(method, block)
    u = [x for x in fm if x in cm]
    if len(u) < 3:
        return False
    tx = np.log10 if METRICS[block]["scale"] == "log" else (lambda v: v)
    d = tx(np.array([fm[x] for x in u], float)) - tx(np.array([cm[x] for x in u], float))
    if np.allclose(d, 0):
        return True
    bm = d[_boot_idx(len(d), B, seed)].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return bool(lo <= 0.0 <= hi)


def paired_effect(method, block, higher, n=2000):
    """Paired effect vs RECAST over matched lineages. Delta>0 ALWAYS favours RECAST.
    Returns mean Delta + 95% paired bootstrap CI (resampling unit = lineage)."""
    fm, cm = lin_map("recast_M0", block), lin_map(method, block)
    units = [u for u in fm if u in cm]
    if len(units) < 3:
        return None
    f = np.array([fm[u] for u in units], float)
    c = np.array([cm[u] for u in units], float)
    delta = (f - c) if higher else (c - f)
    m = float(delta.mean())
    idx = RNG.integers(0, len(delta), size=(n, len(delta)))
    bm = delta[idx].mean(axis=1)
    return dict(mean=m, lo=float(np.percentile(bm, 2.5)),
                hi=float(np.percentile(bm, 97.5)), n=len(units))

# ----------------------------------------------------------------------------- metric metadata
#  higher = "higher value is better".  axis = the explicit x-range for the box column.
#
#  Bounded rates (recall, AUROC, precision, F1, accuracy, balanced accuracy) get their TRUE range,
#  exactly (0, 1) -- a proportion axis that stops at 1.07 or starts at -0.01 invites the reader to
#  wonder what lives out there, and nothing does.  Recall here is |topK and GT| / |GT|, so a state
#  with 6 unique markers whose top-20 contains all 6 scores exactly 1.0: the endpoints are real and
#  attainable, and the axis says so.  The n.s. circle and RECAST's rank live in their own narrow strip to the
#  right (GUTTER_MM), so no data axis is stretched to make room for an annotation.
#  Unbounded/count metrics keep a fitted range.  box_ticks = round ticks spanning the range.
# Width of the annotation strip beside every box column.  It must clear the WIDEST glyph it holds,
# which is a two-digit rank ("#11") at 6.2 pt ~= 4.9 mm; the strip is centred on 0.5*GUTTER_MM, so a
# narrower strip puts the "#" on top of the panel's right spine.
GUTTER_MM = 5.6
# The "Overall score" block prints its value OUTSIDE the right spine, like the n.s. circle and the
# rank glyph, so no bar is ever drawn over its own number.  "0.972" at 6.0 pt is ~5.3 mm wide, so the
# right strip has to be wider than GUTTER_MM when that block is present.
SCORE_STRIP_MM = 6.6
# A box column narrower than this cannot carry SIX tick labels at 6.0 pt without them touching, so a
# metric that defines `box_ticks_narrow` (the two 0-1 rate columns: 0 / 0.5 / 1) falls back to it. The
# count columns label four or five short ticks in the same width and are left alone.
NARROW_COL_MM = 19.0
METRICS = {
    "recall20":    dict(short="recall", title="CellMarker-unique\nRecall@20",  higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=None),
    "auroc":       dict(short="AUROC",  title="One-vs-rest\nAUROC",      higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        box_ticks_narrow=[0, 0.5, 1.0], scale="lin", ref=0.5),
    # ---- confusion-matrix companion figure (all four are rates in [0,1]) ----
    "precision":   dict(short="precision", title="Precision\n(top-20)",  higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], scale="lin", ref=None),
    "recall":      dict(short="recall", title="Recall\n(top-20)",        higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], scale="lin", ref=None),
    # plain "F1": a mathtext subscript would print at 0.7 x the title size, below 5 pt
    "f1":          dict(short="F1",     title="F1\n(top-20)",             higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], scale="lin", ref=None),
    # Accuracy on its TRUE axis is the point, not an accident: ~28,200 of the 28,231 genes are true
    # negatives whatever a method selects, so all 51 collapse against the right spine.  The note fills
    # the space the data cannot, and states the measured spread rather than leaving a blank panel.
    "accuracy":    dict(short="accuracy", title="Accuracy\n(28,231 genes)", higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], scale="lin", ref=None,
                        note="every method\n0.9989–0.9992\n\n~28,200 genes are\ntrue negatives for all;\naccuracy cannot\nseparate them"),
    "balanced_accuracy": dict(short="bal. acc.", title="Balanced\naccuracy", higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], scale="lin", ref=0.5),
    "marker_rank200": dict(short="rank", title="Unique-marker\nrank score (0-100)", higher=True,
                        axis=(-1.5, 113.5),  box_ticks=[0, 25, 50, 75, 100], scale="lin", ref=None),
    "ms_per_cell": dict(short="time",   title="Selection time\n(ms per cell, log)",         higher=False,
                        axis=(0.9, 1500.0),  box_ticks=[1, 10, 100, 1000], scale="log", ref=None),
    "contamination": dict(short="contam", title="Ribosomal & mito. genes\nin the top 200", higher=False,
                        axis=(-1.5, 101.5),  box_ticks=[0, 25, 50, 75, 100], scale="lin", ref=None),
    # Composite of the five blocks above (see overall_scores()).  Not a measurement: a summary of the
    # five measurements the figure already shows, on a 0-1 axis, higher = better on the field as a whole.
    "overall":     dict(short="overall", title="Overall score\n(5 metrics)", higher=True,
                        axis=(0.0, 1.0),  box_ticks=[0, 0.5, 1.0], scale="lin", ref=None),
}

# ----------------------------------------------------------------------------- overall score
# The five blocks the figure draws, in the order they are drawn, with the direction of "better" and the
# scale the block's own axis uses.  Selection time runs over three decades and is drawn on a log axis;
# it is normalised on log10 -- the scale its axis draws -- so the composite is not decided by the single
# slowest method.  (This is the same log convention is_ns() uses for that block.)
SCORE_BLOCKS = (("recall20", True, "lin"), ("auroc", True, "lin"), ("marker_rank200", True, "lin"),
                ("ms_per_cell", False, "log"), ("contamination", False, "lin"))

# Quality vs cost weights for the overall score.  The three higher-is-better blocks
# say what the genes are worth; the two lower-is-better blocks say what they cost.  Cost is kept in the
# composite because the figure reports it, but at a quarter of the weight, so that being fast and clean
# cannot substitute for recovering markers.  See overall_scores() for why equal weight was abandoned.
W_QUALITY, W_COST = 0.75, 0.25

def overall_scores(roster):
    """Overall score for every method in `roster` (the rows the figure draws).

    For each of the five blocks: take the method's mean over the matched units -- the same number the
    block already shows as its mean point, the same number MASTER holds and the rank glyph ranks --
    min-max normalise it ACROSS THE ROSTER (on log10 for the log-scaled block), and flip the two
    lower-is-better blocks (selection time, ribo/mito contamination) to 1 - x so that 1 is always the
    best value in the figure and 0 always the worst.  The three quality axes -- what the selected genes
    are worth -- carry W_QUALITY together, the two cost axes -- what the selection cost -- carry W_COST.

    They are not equally weighted.  A control that selects genes at random is by construction the
    worst method on all three quality axes and the best on both cost axes: it is instant, and random
    genes are not enriched for ribosomal or mitochondrial content.  Under min-max normalisation its five
    values are exactly (0, 0, 0, 1, 1), so an unweighted mean would give it 0.400 and rank it above two
    marker-selection methods; a score whose floor is reached by a method that recovers nothing would not
    measure marker quality.

    The score is therefore RELATIVE to the field shown: 1 would mean "best of these methods on all five
    axes", 0 "worst on all five".  It is not an absolute quality scale and cannot be compared across
    figures with different rosters.
    """
    sub = MASTER.loc[list(roster)]
    out = {}
    for key, higher, scale in SCORE_BLOCKS:
        v = sub[key].astype(float).to_numpy()
        t = np.log10(v) if scale == "log" else v
        lo, hi = float(np.nanmin(t)), float(np.nanmax(t))
        n = (t - lo) / (hi - lo) if hi > lo else np.zeros_like(t)
        out[key] = n if higher else (1.0 - n)
    N = pd.DataFrame(out, index=list(roster))
    q = [k for k, higher, _ in SCORE_BLOCKS if higher]
    c = [k for k, higher, _ in SCORE_BLOCKS if not higher]
    N["overall"] = W_QUALITY * N[q].mean(axis=1) + W_COST * N[c].mean(axis=1)
    return N


def recast_rank(key):
    """RECAST's 1-based rank on `key` across the shown field (1 = best), and the field size."""
    M = METRICS[key]
    s = MASTER[key].sort_values(ascending=not M["higher"])
    return list(s.index).index("recast_M0") + 1, len(s)

# ----------------------------------------------------------------------------- main-figure subset
# Taxonomy = the algorithmic approach each marker-selection method takes (field-standard split,
# after Pullin & McCarthy): single-cell DE tests | general-purpose DE | marker-scoring (non-DE effect-
# size/specificity scorers) | attribution (RECAST) | control.  scran findMarkers = pairwise DE test
# -> Single-cell DE; scran scoreMarkers = effect-size specificity scores -> Marker Scoring.
# Most of the 56 methods are the same method invoked differently ("Wilcoxon(tc)" beside "Wilcoxon(tc) (abs)", nine
# findMarkers settings, fifteen scoreMarkers settings), so the main figure shows five representatives per family and
# the full field is in Supplementary Fig. 3.  RECAST's ranks are unaffected: recast_rank() reads MASTER, the whole
# field, whichever subset is drawn.
#
# Two selection rules, both stated in the manuscript caption:
#   * Where one method was run in several configurations, only its BEST configuration is shown,
#     scored by CellMarker-unique Recall@20 (the panel's primary axis). Keeping each competitor's
#     best setting is the conservative choice -- it strengthens the field RECAST is measured against.
#   * Which five per family: single-cell DE is picked for familiarity (the tests a reader expects to
#     see) rather than for score, so the row set cannot be read as chosen to flatter RECAST. This
#     costs RECAST nothing to admit: the omitted linear model (0.2986) and Seurat wilcox (0.2960)
#     both outscore two of the five shown, and both are in the full field (Supplementary Fig. 3).
#     General-purpose DE has only three packages, so limma and edgeR keep both of their standard tests
#     (trend/voom, GLM/QL) to fill the family -- the one place a "same method, different call" pair
#     survives, because collapsing it would leave a three-row family.
# The five methods whose primary output is a marker panel rather than a ranked test statistic, in the order the family block lists them (attribution, forest, sparse hyperplane,
# filter cascade, linear program).
PANEL_SELECTION = ("scmarkergene", "nsforest", "rankcorr", "scmags", "scgenefit")

MAIN_MEMBERS = {
    "RECAST":          ["recast_M0"],
    # rank test | parametric test | discriminative | GLM | no test
    "Single-cell DE": ["scanpy_wilcoxontiecorrect_rankby_raw",   # best of 4 Wilcoxon configurations
                       "scanpy_tover_rankby_raw",                # best of 4 t-test configurations
                       "seurat_LR", "seurat_MAST",
                       "logfc_raw"],                             # best of 2 (raw vs abs)
    "Pseudobulk DE":  ["limma_trend", "limma_voom", "edger_glm", "edger_ql", "glmgampoi"],
    "Marker Scoring": ["cosg",
                       "scran_scoreMarkers_mean_logFC_cohen",    # best of 5 summary statistics
                       "scran_scoreMarkers_mean_AUC",            # best of 5
                       "scran_scoreMarkers_max_logFC_detected",  # best of 5
                       "cepo"],
    # attribution | forest | sparse hyperplane | filter cascade | linear program.  All five members
    # of the family are shown: there are exactly five, so no representative choice is being made and
    # none of them is left to the full field.
    "Dedicated panel selection": list(PANEL_SELECTION),
    "Control":        ["random"],
}
MAIN_NAME = {
    "recast_M0": "RECAST",
    "scanpy_wilcoxontiecorrect_rankby_raw": "Wilcoxon",
    "scanpy_tover_rankby_raw": "t-test",
    "seurat_LR": "Logistic regression", "seurat_MAST": "MAST",
    "logfc_raw": "Log fold-change",
    "limma_trend": "limma (trend)", "limma_voom": "limma (voom)",
    "edger_glm": "edgeR (GLM)", "edger_ql": "edgeR (QL)", "glmgampoi": "glmGamPoi",
    "cosg": "COSG",
    "scran_scoreMarkers_mean_logFC_cohen": "scoreMarkers (Cohen's d)",
    "scran_scoreMarkers_mean_AUC": "scoreMarkers (AUC)",
    "scran_scoreMarkers_max_logFC_detected": "scoreMarkers (detection)",
    "cepo": "Cepo", "random": "Random",
    "scmarkergene": "scMarkerGene", "nsforest": "NS-Forest", "rankcorr": "RankCorr",
    "scmags": "scMAGS", "scgenefit": "scGeneFit",
}

def main_name(mid):
    return MAIN_NAME[mid]

def main_groups():
    """The main-figure roster, ordered within each family by recall (as field_groups does)."""
    out = []
    for f in FAM_ORDER:
        ms = MAIN_MEMBERS.get(f, [])
        missing = [m for m in ms if m not in MASTER.index]
        if missing:
            raise KeyError(f"main-figure roster names methods absent from MASTER: {missing}")
        if ms:
            out.append((f, sorted(ms, key=lambda m: -MASTER.loc[m, "recall20"])))
    return out

# ----------------------------------------------------------------------------- the full field
def family_of(mid):
    if mid == "recast_M0": return "RECAST"
    if mid == "random":   return "Control"
    if mid in PANEL_SELECTION:
        return "Dedicated panel selection"                 # output is a marker panel, not a ranked test statistic
    if mid.startswith("scran_scoreMarkers") or mid in ("cosg", "cepo"):
        return "Marker Scoring"                            # effect-size / specificity scorers (no DE test)
    if mid in ("edger_glm", "edger_ql", "limma_trend", "limma_voom", "glmgampoi"):
        return "Pseudobulk DE"                             # key only: fitted per cell, one versus rest, no aggregation;
                                                           # the figures draw it as "General-purpose DE"
    return "Single-cell DE"                                # per-cell DE tests incl. scran findMarkers, Seurat GLMs

# Families run from the most comparable to RECAST to the least, with RECAST at the top and the control at the
# bottom.  The two families whose stated output is a marker panel or a
# marker score answer the same question RECAST answers, so they sit directly beneath it; the DE tests,
# which answer "is this gene differential" and are only then read as a ranking, follow.  "Dedicated
# panel selection" holds the algorithms whose stated output is a marker panel.  scMarkerGene belongs here and is the closest comparator RECAST has -- it too ranks genes by
# attributing a trained model.  Order is presentation only: the "#k" glyphs rank over the full 56-method
# field, the row order inside a block is recall-descending, and the overall score is order-invariant.
FAM_ORDER = ["RECAST", "Dedicated panel selection", "Marker Scoring",
             "Single-cell DE", "Pseudobulk DE", "Control"]

def field_name(mid):
    S = {"recast_M0": "RECAST", "cosg": "COSG", "cepo": "Cepo", "random": "Random",
         "logfc_raw": "log FC", "logfc_abs": "log FC (abs)", "lm_two_sample": "linear model",
         "glmgampoi": "glmGamPoi", "limma_trend": "limma (trend)", "limma_voom": "limma (voom)",
         "edger_glm": "edgeR (GLM)", "edger_ql": "edgeR (QL)",
         "scmarkergene": "scMarkerGene", "nsforest": "NS-Forest", "rankcorr": "RankCorr",
         "scmags": "scMAGS", "scgenefit": "scGeneFit"}
    if mid in S: return S[mid]
    s = mid
    s = s.replace("scran_scoreMarkers_", "scoreMarkers ").replace("scran_findMarkers_", "findMarkers ")
    s = s.replace("_logFC_cohen", " Cohen").replace("_logFC_detected", " det").replace("_AUC", " AUC")
    s = s.replace("seurat_", "Seurat ").replace("scanpy_", "")
    s = s.replace("wilcoxontiecorrect", "Wilcoxon(tc)").replace("wilcoxon", "Wilcoxon")
    s = s.replace("tover", "t(ov)").replace("_rankby_raw", "").replace("_rankby_abs", " (abs)")
    s = (s.replace("scoreMarkers mean", "scoreMarkers (mean)").replace("scoreMarkers median", "scoreMarkers (med)")
          .replace("scoreMarkers rank", "scoreMarkers (rank)").replace("scoreMarkers min", "scoreMarkers (min)")
          .replace("scoreMarkers max", "scoreMarkers (max)"))
    s = s.replace("findMarkers ", "findMarkers ").replace("_", " ")
    s = s.replace("t all", "t (all)").replace("t some", "t (some)").replace("t any", "t (any)")
    s = (s.replace("wilcox all", "wilcox (all)").replace("wilcox some", "wilcox (some)")
          .replace("wilcox any", "wilcox (any)").replace("binom all", "binom (all)")
          .replace("binom some", "binom (some)").replace("binom any", "binom (any)"))
    return s.strip()

def field_groups():
    g = {f: [] for f in FAM_ORDER}
    for mid in MASTER.index:
        g[family_of(mid)].append(mid)
    for f in g:  # order within family by recall desc
        g[f] = sorted(g[f], key=lambda m: -MASTER.loc[m, "recall20"])
    return [(f, g[f]) for f in FAM_ORDER if g[f]]

# ----------------------------------------------------------------------------- layout engine
def layout(groups, STEP=1.0, GAP=0.35, HEAD=1.05):
    """assign vertical depth (increases downward) to each method + each family header."""
    d = 0.0; mpos = {}; hpos = []
    for gi, (h, members) in enumerate(groups):
        if gi > 0: d += GAP
        hpos.append((h, d)); d += HEAD
        for m in members:
            mpos[m] = d; d += STEP
    return mpos, hpos, d  # d = total depth

# ----------------------------------------------------------------------------- renderer
HH, HC = 0.30, 0.17     # box half-height / whisker-cap half-height, in depth units

def _draw_box(ax, d, st, vals, is_f, ns_flag, seed, higher=True, ann_x=1.03):
    if is_f:
        fill, edge, medc, meanc, msz, dotc = RECAST, RECAST_D, RECAST_D, RECAST_D, 12, RECAST_D
        fill_a = 0.20
    else:
        fill, edge, medc, meanc, msz, dotc = G100, G550, G700, G900, 7, G550
        fill_a = 1.0
    ax.fill_betweenx([d - HH, d + HH], st["q1"], st["q3"], facecolor=fill, alpha=fill_a,
                     edgecolor="none", zorder=2)
    # individual lineage points (jittered), drawn under the box outline; deterministic jitter
    xr0, xr1 = ax.get_xlim()
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    m = (v >= xr0) & (v <= xr1)
    if m.any():
        rs = np.random.default_rng(seed)
        jit = (rs.random(len(v)) - 0.5) * (1.5 * HH)
        ax.scatter(v[m], d + jit[m], s=1.0, color=dotc, alpha=0.5, linewidths=0, zorder=2.4)
    ax.plot([st["q1"], st["q3"], st["q3"], st["q1"], st["q1"]],
            [d - HH, d - HH, d + HH, d + HH, d - HH], color=edge, lw=0.7, zorder=3)
    ax.plot([st["whislo"], st["q1"]], [d, d], color=edge, lw=0.7, zorder=3)
    ax.plot([st["q3"], st["whishi"]], [d, d], color=edge, lw=0.7, zorder=3)
    ax.plot([st["whislo"]] * 2, [d - HC, d + HC], color=edge, lw=0.7, zorder=3)
    ax.plot([st["whishi"]] * 2, [d - HC, d + HC], color=edge, lw=0.7, zorder=3)
    ax.plot([st["med"], st["med"]], [d - HH, d + HH], color=medc, lw=1.1, zorder=4)
    ax.scatter([st["mean"]], [d], s=msz, color=meanc, edgecolors="white",
               linewidths=0.4, zorder=5)
    if ns_flag:      # not significantly different from RECAST on this metric -> a hollow circle on this
        # method's row, in the annotation strip OUTSIDE the right spine.  Drawing it outside (rather than
        # in a slice of the data axis) is what allows bounded metrics to use their true (0, 1) range.
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.scatter([ann_x], [d], transform=trans, s=13, facecolors="none",
                   edgecolors=meanc, linewidths=0.7, zorder=6, clip_on=False)

def _draw_effect(ax, d, ef):
    if ef["lo"] > 0:      pc, cc = RECAST_D, RECAST      # CI entirely >0 -> RECAST better
    elif ef["hi"] < 0:    pc, cc = G900, G700          # CI entirely <0 -> comparator better
    else:                 pc, cc = G400, G300          # CI crosses 0 -> no clear difference
    ax.plot([ef["lo"], ef["hi"]], [d, d], color=cc, lw=1.4, solid_capstyle="round", zorder=3)
    ax.scatter([ef["mean"]], [d], s=11, color=pc, edgecolors="white", linewidths=0.35, zorder=4)

def _draw_score_bar(ax, d, s, is_f, ann_x):
    """One horizontal bar per method on a 0-1 axis, with the value printed in the strip outside the
    right spine (same strip the n.s. circle and the rank glyph use, so a bar never covers its number)."""
    if is_f:
        fill, edge, txtc, alpha = RECAST, RECAST_D, RECAST_D, 1.0
    else:
        fill, edge, txtc, alpha = G200, G550, G700, 1.0
    ax.fill_betweenx([d - HH, d + HH], 0.0, s, facecolor=fill, alpha=alpha,
                     edgecolor=edge, linewidth=0.7, zorder=3)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(ann_x, d, f"{s:.3f}", transform=trans, ha="left", va="center", fontsize=6.0,
            fontweight="bold" if is_f else "regular", color=txtc, zorder=9, clip_on=False)

# family headings as printed (the family names themselves are the keys of the tables above)
HEADER_CASE = {"Marker Scoring": "Marker scoring", "Pseudobulk DE": "General-purpose DE"}

LABEL_COL_MM = 27.0      # width of the method-label column (a caller may widen it for a longer key)


def draw(groups, namer, stem, slot_mm, columns, region_headers, show_fliers=True, fig_w=180.0):
    mpos, hpos, maxd = layout(groups)
    nbox = sum(1 for c in columns if c["kind"] == "box")
    neff = sum(1 for c in columns if c["kind"] == "effect")
    nsco = sum(1 for c in columns if c["kind"] == "score")
    SCORES = overall_scores(list(mpos)) if nsco else None
    L_edge, LW, gap_lbl = 2.0, LABEL_COL_MM, 2.5
    # Every box column carries a GUTTER_MM annotation strip OUTSIDE its right spine (the n.s. circle and RECAST's
    # rank), so bounded metrics keep exactly (0, 1).
    # The right strip has to hold the widest glyph the LAST column parks outside its spine: "#11" for a
    # box column, "0.972" for the score column.
    R_strip = SCORE_STRIP_MM if columns[-1]["kind"] == "score" else GUTTER_MM
    intra, gutter, R_edge = 1.6 + GUTTER_MM, 5.0 + GUTTER_MM, 2.0 + R_strip
    LBL_GAP = 2.0
    # The score block is added WITHOUT widening the figure: it and its region gap are taken out of the
    # same 180 mm, so every existing block shrinks by the same factor (21.7 -> 17.3 mm).
    wbox_rel, weff_rel, wsco_rel = 1.0, 0.80, 0.62
    regs = [c["region"] for c in columns]
    n_intra = sum(1 for i in range(1, len(columns)) if regs[i] == regs[i - 1])
    n_gut   = sum(1 for i in range(1, len(columns)) if regs[i] != regs[i - 1])
    avail = fig_w - L_edge - LW - gap_lbl - R_edge - n_intra * intra - n_gut * gutter
    unit = avail / (nbox * wbox_rel + neff * weff_rel + nsco * wsco_rel)
    W_REL = {"box": wbox_rel, "effect": weff_rel, "score": wsco_rel}

    x = L_edge + LW + gap_lbl
    for i, c in enumerate(columns):
        if i > 0:
            x += gutter if regs[i] != regs[i - 1] else intra
        c["_w"] = unit * W_REL[c["kind"]]
        # box / effect: glyph CENTRED in the strip.  score: number LEFT-aligned 1.1 mm past the spine.
        c["_annx"] = 1.0 + (1.1 if c["kind"] == "score" else 0.5 * GUTTER_MM) / c["_w"]
        c["_x"] = x
        x += c["_w"]

    # bottom margin holds, top-to-bottom: x-tick labels, then the per-region direction arrow + its
    # "Higher/Lower values are better" caption.
    top_m, ttl_h = 3.0, 8.0
    bot_m = 15.0                                   # arrows + captions + the mark key
    # The axes height equals the y range actually drawn, not the layout depth (which would leave empty axis below
    # the last method row).
    dmin, dmax = min(mpos.values()), max(mpos.values())
    ypad = 0.9
    ymin, ymax = dmin - ypad, dmax + ypad
    panels_h = (ymax - ymin) * slot_mm
    fig_h = top_m + ttl_h + panels_h + bot_m
    fig = plt.figure(figsize=(mm(fig_w), mm(fig_h)))
    yb, yh = bot_m / fig_h, panels_h / fig_h
    ed = ("full" in stem)

    # pre-compute x-limits per column across the shown methods
    for c in columns:
        M = METRICS[c["key"]]
        if c["kind"] == "score":
            c["_boxlim"] = (0.0, 1.0)                      # the score is defined on 0-1 by construction
        elif c["kind"] == "effect":
            vals = []
            for mid in mpos:
                if mid == "recast_M0":
                    continue
                ef = paired_effect(mid, c["key"], M["higher"])
                if ef:
                    vals += [ef["lo"], ef["hi"]]
            lo, hi = (min(vals), max(vals)) if vals else (-1, 1)
            span = max(hi - lo, 1e-9)
            c["_xlim"] = (min(lo, 0) - 0.06 * span, max(hi, 0) + 0.06 * span)
        else:  # box column: FIXED full-range axis (spans the whole data range so ticks label end-to-end);
            # the last ~10% is a gutter beyond every method's data that holds the n.s. circle / RECAST rank.
            lo_a, hi_a = M["axis"]
            v = DOTS[DOTS.block == c["key"]].value
            v = v[np.isfinite(v)]
            if M["scale"] == "log":                       # log metric (selection time): fit to the plotted values.
                # the times run 0.1-500 ms per cell (selection call vs selection call)
                if len(v):
                    vmin = float(v[v > 0].min()); vmax = max(float(v.max()), float(MASTER[c["key"]].max()))
                    lo_a = vmin / 1.6
                    hi_a = vmax * (vmax / lo_a) ** 0.12   # ~11 % of the log span as the gutter for the rank label
            elif hi_a != 1.0:                             # count metric: fit to the plotted values
                if len(v):
                    hi_a = max(float(v.max()), float(MASTER[c["key"]].max()))
                    lo_a = min(0.0, float(v.min()))
            c["_boxlim"] = (lo_a, hi_a)

    axes = []
    for c in columns:
        ax = fig.add_axes([c["_x"] / fig_w, yb, c["_w"] / fig_w, yh])
        ax.set_ylim(ymax, ymin)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color(G400); ax.spines[sp].set_linewidth(0.7)
        ax.set_yticks([]); ax.patch.set_alpha(0.0)
        ax.tick_params(axis="x", length=2.0, color=G550, labelsize=6.0, pad=2.0)
        M = METRICS[c["key"]]
        if c["kind"] == "score":
            lo, hi = c["_boxlim"]
            pad = 0.035 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_xticks(M["box_ticks"]); ax.set_xticklabels(["0", "0.5", "1"])
            title = M["title"]
        elif c["kind"] == "box":
            lo, hi = c["_boxlim"]
            if M["scale"] == "log":
                ax.set_xscale("log")
                ax.set_xlim(lo, hi)
                dec = [t for t in (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000) if lo <= t <= hi]
                # the column is narrow: label every other decade (0.1, 10, 1000) so "100" and "1000" cannot
                # touch, and mark the decades in between with unlabelled minor ticks
                major = [t for t in dec if int(round(np.log10(t))) % 2 == 1]
                minor = [t for t in dec if t not in major]
                ax.set_xticks(major); ax.set_xticklabels([("%g" % t) for t in major])
                ax.set_xticks(minor, minor=True); ax.tick_params(axis="x", which="minor", length=1.6, width=0.4)
                ax.xaxis.set_minor_locator(NullLocator())
            else:
                # Ticks still label the honest full range (lo..hi), but the DRAWN limits are padded
                # so a distribution sitting at an axis end (RECAST's AUROC, its contamination) does not merge with the
                # spine, and whiskers reaching exactly 1.0 are not cut off.
                pad = 0.035 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
                bt = M["box_ticks"] or []
                if c["_w"] < NARROW_COL_MM and M.get("box_ticks_narrow"):
                    bt = M["box_ticks_narrow"]          # too narrow for six labels: first / middle / last
                tk = [t for t in bt if lo <= t <= hi]   # keep nice ticks inside the fit
                if len(tk) >= 2:
                    ax.set_xticks(tk); ax.set_xticklabels([("%g" % t) for t in tk])
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
            if M["ref"] is not None and lo <= M["ref"] <= hi:          # faint biological anchor (chance / median)
                ax.axvline(M["ref"], color=G200, lw=0.6, ls=(0, (2, 2)), zorder=0)
            fv = MASTER.loc["recast_M0", c["key"]]                      # blue dashed RECAST reference, ON TOP
            if lo <= fv <= hi:
                ax.axvline(fv, color=RECAST_D, lw=1.0, ls=(0, (3, 2)), zorder=8, alpha=1.0)
            if M.get("note"):
                # A metric whose true axis is honest but nearly empty (accuracy: every method sits
                # against the right spine) leaves the reader wondering what went wrong.  The note
                # says what they are looking at, in the space the data does not use.
                ax.text(0.44, 0.5, M["note"], transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.0, color=G550, linespacing=1.45, zorder=7)
            title = M["title"]
        else:  # effect column
            ax.set_xlim(*c["_xlim"])
            ax.axvline(0.0, color=G550, lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
            title = "$\\Delta$ " + M["short"]
        ax.set_title(title, fontsize=6.8, color=INK, linespacing=0.95, pad=4)
        axes.append((ax, c))

    # label column (own axis), family headers on their own row, method labels right-aligned
    lab_w = (columns[0]["_x"] - LBL_GAP) - L_edge
    lab = fig.add_axes([L_edge / fig_w, yb, lab_w / fig_w, yh])
    lab.set_ylim(ymax, ymin); lab.axis("off"); lab.set_xlim(0, 1)
    for hname, d in hpos:
        if hname == "RECAST":
            continue
        lab.text(0.0, d + 0.12, HEADER_CASE.get(hname, hname), ha="left", va="center",
                 fontsize=6.9, fontweight="bold", color=INK)
    for mid, d in mpos.items():
        is_f = (mid == "recast_M0")
        lab.text(0.99, d, namer(mid), ha="right", va="center",
                 fontsize=6.2 if not is_f else 6.6,
                 fontweight="bold" if is_f else "regular",
                 color=RECAST_D if is_f else G700)
        if is_f:
            lab.plot([0.02, 0.02], [d - 0.34, d + 0.34], color=RECAST, lw=2.2,
                     solid_capstyle="butt")

    # draw glyphs
    for ax, c in axes:
        M = METRICS[c["key"]]
        for mid, d in mpos.items():
            is_f = (mid == "recast_M0")
            if c["kind"] == "score":
                _draw_score_bar(ax, d, float(SCORES.loc[mid, "overall"]), is_f, c["_annx"])
            elif c["kind"] == "box":
                vals = vals_of(mid, c["key"])
                st = box_stats(vals)
                if st is not None:
                    ns = (not is_f) and is_ns(mid, c["key"])
                    seed = zlib.crc32((mid + "|" + c["key"]).encode()) & 0xFFFFFFFF
                    _draw_box(ax, d, st, vals, is_f, ns, seed, higher=M["higher"],
                              ann_x=c["_annx"])
            else:
                if is_f:
                    continue                      # RECAST is the reference; no self-effect
                ef = paired_effect(mid, c["key"], M["higher"])
                if ef is not None:
                    _draw_effect(ax, d, ef)
        if c["kind"] == "box":                    # RECAST's own rank on this metric, in the right gutter of
            r, _ = recast_rank(c["key"])            # its row (aligned with the n.s. circles; RECAST has none)
            df = mpos["recast_M0"]
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(c["_annx"], df, f"#{r}", transform=trans, ha="center", va="center",
                    fontsize=6.2, fontweight="bold", color=RECAST_D, zorder=9, clip_on=False)

    return save_fig(fig, stem, OUT)

# ----------------------------------------------------------------------------- column specs
RH_A = ("Marker recovery and subtype discrimination", "higher is better  →")
RH_B = ("Efficiency and off-target signal", "←  lower is better")
RH_C = ("Overall score", "higher is better  →")

FIELD_BOX_COLUMNS = [
    dict(kind="box", key="recall20",       region="A"),
    dict(kind="box", key="auroc",          region="A"),
    dict(kind="box", key="marker_rank200", region="A"),
    dict(kind="box", key="ms_per_cell",    region="B"),
    dict(kind="box", key="contamination",  region="B"),
]
# Fig. 3 = the five measured blocks plus the overall score; the full field (Supplementary Fig. 3) shows the five.
MAIN_BOX_COLUMNS = FIELD_BOX_COLUMNS + [dict(kind="score", key="overall", region="C")]
