"""Colours of the methods and of the biological categories, for the panel modules of the main figures.

  * RECAST                          blue accent (the only saturated series)
  * conventional methods            dark grey
  * random / null / floor           light grey
  * reference-free attribution      mid grey
  * cell states, lineages, tissues  a muted categorical set
  * out-of-sample overlays          open markers over filled in-sample points

Also re-exports the helpers of figure_style, so a panel module needs a single `from method_colours import ...`.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from figure_style import (apply_style, mm, thin, save_fig, header, panel_label,
                      INK, SUB, G900, G700, G550, G400, G300, G200, G100, HAIR,
                      W1, W15, W2)

# ------------------------------------------------------------------ RECAST hero (sample1 blue)
RECAST    = "#4E86B0"   # hero accent — deepened sample1 steel-blue
RECAST_D  = "#2E5F82"   # dark end (emphasis)
RECAST_L  = "#A9C6DD"   # light tint (RECAST-unique / secondary)
RECAST_XL = "#DDE9F2"   # wash (fills / shaded bands)

# ------------------------------------------------------------------ conventional (DE / expression)
DE       = "#4D4D4D"   # dark neutral gray — the conventional / DE field
DE_L     = "#8A8A8A"
DE_XL    = "#C9C9C9"

# ------------------------------------------------------------------ null / random / floor
NULLC    = "#C4C4C4"   # light neutral
NULL_L   = "#E2E2E2"

# ------------------------------------------------------------------ reference-free attribution
REF_FREE = "#6F6F6F"     # mid neutral: reference-free attribution

# ------------------------------------------------------------------ restrained biological categorical
# muted, low-saturation; deliberately avoids pure RECAST-blue so the hero stays distinct.
BIO = ["#6E8CA0",  # slate
       "#B08268",  # clay
       "#7E9E77",  # sage
       "#9887B0",  # muted purple
       "#C2A25A",  # ochre
       "#7FA6A9",  # dusty teal
       "#B58BA0",  # mauve
       "#8F8F8F"]  # neutral

WARN = "#B06A4F"       # muted terracotta — wrong-reference / degradation (very sparing)

# ------------------------------------------------------------------ colormaps (single-hue)
def recast_cmap():
    """white -> RECAST blue -> dark: for RECAST-side single-hue heatmaps."""
    return LinearSegmentedColormap.from_list("recast_seq", ["#FFFFFF", RECAST_XL, RECAST, RECAST_D])

def neutral_cmap():
    """white -> gray -> ink: for reference-free / conventional single-hue heatmaps (reserves blue for RECAST)."""
    return LinearSegmentedColormap.from_list("neutral_seq", ["#FFFFFF", "#EAEAEA", "#9A9A9A", INK])

# ------------------------------------------------------------------ semantic role map
COL = {
    "recast": RECAST, "recast_d": RECAST_D, "recast_light": RECAST_L, "recast_unique": RECAST_L,
    "de": DE, "expr": DE, "conventional": DE, "baseline": DE, "competitor": DE, "cosg": DE,
    "reference_free": REF_FREE, "null": NULLC, "random": NULLC, "floor": NULLC, "field": NULL_L,
    "meanline": DE, "ink": INK, "sub": SUB, "hair": HAIR, "grid": G100, "warn": WARN,
}

# method identity: RECAST blue; everyone else neutral depth (NEVER per-method hues)
MCOL = {"RECAST": RECAST, "RECAST-unique": RECAST_L,
        "COSG": DE, "Wilcoxon": DE, "limma": DE, "scran": DE_L, "competitor": DE,
        "reference_free": REF_FREE, "field": NULL_L, "null": NULLC}

# ------------------------------------------------------------------ small numeric helpers
def rank_intensity(rank, cap=100.0):
    """rank (1=best) -> intensity in [0,1] with a log falloff (rank1=1, rank=cap=0)."""
    r = np.asarray(rank, dtype=float)
    out = 1.0 - np.log10(np.clip(r, 1, cap)) / np.log10(cap)
    return np.clip(out, 0.0, 1.0)

def hollow(**kw):
    """marker kwargs for an external / out-of-sample point: open face, coloured edge."""
    d = dict(facecolors="none", linewidths=0.8)
    d.update(kw)
    return d
