"""Colours, sizes and small helpers shared by the panel modules of the main figures.

Near-monochrome: ink and a grey ramp, with one accent colour reserved for RECAST.  Thin lines, small type, no
decoration.  apply_style() sets the matplotlib defaults and then installs the house style of panel_style.py.

    from figure_style import (ACCENT, ACCENT_D, ACCENT_L, INK, SUB, GRAY, WARN,
                              COL, MCOL, mm, apply_style, panel_label, header,
                              thin, save_fig, seq_cmap, ramp, W1, W15, W2)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------- column widths
W1, W15, W2 = 88.0, 120.0, 180.0            # mm — single / 1.5-col / double
def mm(x): return x / 25.4                   # mm -> inches

# ---------------------------------------------------------------- font
_avail = {f.name for f in fm.fontManager.ttflist}
for _cand in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
    if _cand in _avail:
        _FONT = _cand; break
else:
    _FONT = "DejaVu Sans"

# ---------------------------------------------------------------- palette
# The single accent (RECAST). A restrained deep gray-teal — the ONLY chromatic hue.
ACCENT   = "#2F6B74"     # RECAST
ACCENT_D = "#1E4A51"     # darker accent (emphasis / dark end of the accent ramp)
ACCENT_L = "#8FB4B8"     # light accent tint (RECAST-unique / secondary accent object)

# grayscale ramp, dark -> light
INK  = "#1A1A1A"         # primary text / darkest ink
G900 = "#333333"
G700 = "#565656"         # strong baseline / dark gray series
G550 = "#7C7C7C"         # mid gray series
G400 = "#A0A0A0"
G300 = "#BEBEBE"
G200 = "#D6D6D6"         # field / light series
G100 = "#EBEBEB"         # null / very light
HAIR = "#D9D9D9"         # hairline rules / gridlines
SUB  = "#6E6E6E"         # secondary text

# one muted "negative / mismatch" tone, used VERY sparingly (e.g. wrong-lineage, degradation)
WARN = "#B06A4F"         # muted terracotta

# semantic role map (keys kept stable so panel scripts read cleanly)
COL = {
    "recast": ACCENT, "recast_d": ACCENT_D, "recast_light": ACCENT_L, "recast_unique": ACCENT_L,
    "reference_free": G700, "expr": G550, "competitor": G550, "global_ref": G400,
    "null": G100, "field": G200, "meanline": G550, "warn": WARN,
    "ink": INK, "sub": SUB, "grid": G100, "hair": HAIR,
    # populations in schematics: target = accent, reference = mid gray (overridable per panel)
    "target": ACCENT, "reference": G550,
    # minimal muted categoricals — ONLY for unavoidable multi-cluster panels (e.g. UMAP)
    "state": [ACCENT, G550, ACCENT_L, G400, INK, G300],
}

# method identity: RECAST accent; everyone else grayscale depth
MCOL = {
    "RECAST": ACCENT, "RECAST-unique": ACCENT_L,
    "COSG": G550, "limma": G550, "Wilcoxon": G550, "competitor": G700,
    "reference_free": G700, "expr": G550, "field": G200, "null": G100,
    # compartments / lineages: grayscale depth ramp (disambiguate with direct labels)
    "B": G700, "CD4 T": G550, "CD8 T": G400, "monocyte": INK, "NK": G300, "ASC": G200,
}


def ramp(n, dark=INK, light=G200):
    """n grayscale steps dark->light, for ordered categoricals used WITH direct labels."""
    c0 = [int(dark.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    c1 = [int(light.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
    out = []
    for k in range(n):
        t = 0 if n == 1 else k/(n-1)
        out.append("#%02x%02x%02x" % tuple(round(c0[j] + (c1[j]-c0[j])*t) for j in range(3)))
    return out

# ---------------------------------------------------------------- rcParams
def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": [_FONT], "font.size": 7.0,
        "axes.titlesize": 7.2, "axes.titleweight": "regular", "axes.titlecolor": INK,
        "axes.labelsize": 7.0, "axes.labelcolor": INK,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
        "axes.edgecolor": G550, "axes.linewidth": 0.5,
        "xtick.color": G550, "ytick.color": G550, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "lines.antialiased": True, "patch.linewidth": 0.5,
        "mathtext.default": "regular",
    })
    # every panel then takes the house style of panel_style.py (the look of Fig. 6d), unless PANEL_STYLE_OFF=1
    import panel_style
    panel_style.activate()

# ---------------------------------------------------------------- helpers
def thin(ax, spines=("left", "bottom")):
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(s in spines)
        if s in spines:
            ax.spines[s].set_linewidth(0.5); ax.spines[s].set_color(G550)
    ax.tick_params(length=2.0, width=0.5, colors=G550, labelcolor=INK)
    return ax

def panel_label(ax, letter, dx_pt=-15.0, dy_pt=5.0, fontsize=8.5):
    """Bold lowercase panel letter just outside the axes' top-left corner."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt),
                textcoords="offset points", fontsize=fontsize, fontweight="bold",
                color=INK, ha="left", va="bottom", annotation_clip=False, zorder=50)

def header(ax, text, dx_pt=0.0, dy_pt=5.0, fontsize=7.0, color=None, weight="regular"):
    """Short, quiet phrase header for a panel (never a full-sentence conclusion)."""
    ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt),
                textcoords="offset points", fontsize=fontsize, fontweight=weight,
                ha="left", va="bottom", color=color or INK, annotation_clip=False, zorder=40)

RASTER_DPI = 600   # every export is 600 dpi: the PNG itself, and the bitmaps matplotlib embeds inside the
                   # vector PDF/SVG for rasterized artists (colorbars with >= 50 segments, imshow/pcolormesh
                   # heatmaps, `rasterized=True` scatters). Without an explicit dpi those bitmaps are rendered
                   # at figure.dpi (150) and ship as 150-ppi images inside an otherwise vector figure.


def save_fig(fig, stem, outdir):
    """Export at EXACT figure size: vector PDF + SVG (embedded rasters at RASTER_DPI) + 600-dpi PNG."""
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=RASTER_DPI, transparent=False)
    fig.savefig(f"{outdir}/{stem}.png", dpi=RASTER_DPI, transparent=False)
    plt.close(fig)
    return [f"{outdir}/{stem}.{e}" for e in ("pdf", "svg", "png")]
