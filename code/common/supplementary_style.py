"""Drawing rules of the Supplementary Figures: about 180 mm wide, text 8 pt, panel letters 9 pt bold.

  text           Liberation Sans 8 pt, #000000 (hues kept for method identity)
  panel letters  9 pt bold lowercase
  axis lines     black 0.5 pt, left and bottom; ticks black 0.5 pt, 2.0 pt long, outside
  reference line black 0.5 pt, dashed 4 pt on / 4 pt off
  hairlines      #D9D9D9 0.4 pt (row separators only)
Method colours are those of method_colours.py: RECAST blue, conventional dark grey, controls light grey.

Every size is set explicitly here.  Importing this module sets PANEL_STYLE_OFF=1, so that the panel modules of the main
figures, which some scripts import for their data and computations, do not install the 6.2 pt house style.
"""
import os

os.environ["PANEL_STYLE_OFF"] = "1"      # a main-figure panel module imported later must not install its style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 8.0                                  # pt, all text
FS_LETTER = 9.0                           # pt, bold panel letters
LW = 0.5                                  # pt, axis lines, ticks, reference lines
HAIR_LW, HAIR = 0.4, "#D9D9D9"
TICK, TICK_PAD, LABEL_PAD = 2.0, 1.8, 2.0
BLACK = "#000000"
DASH = (0, (4 / LW, 4 / LW))              # matplotlib scales dashes by line width: 4 pt on / 4 pt off
DPI = 600

# method colours (method_colours.py)
RECAST, RECAST_D, RECAST_L, RECAST_XL = "#4E86B0", "#2E5F82", "#A9C6DD", "#DDE9F2"
DE, DE_L, DE_XL = "#4D4D4D", "#8A8A8A", "#C9C9C9"
NULLC, NULL_L = "#C4C4C4", "#E2E2E2"
REF_FREE = "#6F6F6F"
BIO = ["#6E8CA0", "#B08268", "#7E9E77", "#9887B0", "#C2A25A", "#7FA6A9", "#B58BA0", "#8F8F8F"]

RC = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"], "font.size": FS,
    "axes.titlesize": FS, "axes.titleweight": "regular", "axes.titlecolor": BLACK, "axes.titlepad": 3.0,
    "axes.labelsize": FS, "axes.labelcolor": BLACK, "axes.labelpad": LABEL_PAD,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS, "legend.title_fontsize": FS,
    "axes.edgecolor": BLACK, "axes.linewidth": LW,
    "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelcolor": BLACK, "ytick.labelcolor": BLACK,
    "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": LW, "ytick.minor.width": LW,
    "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.major.pad": TICK_PAD, "ytick.major.pad": TICK_PAD,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": BLACK, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.dpi": 150, "savefig.dpi": DPI, "lines.antialiased": True, "patch.linewidth": LW,
    "mathtext.default": "regular", "legend.frameon": False, "axes.unicode_minus": True,
    "hatch.linewidth": 0.4,
}


def setup():
    """Install the rules (call after importing a panel module of the main figures, which sets its own rcParams)."""
    matplotlib.rcParams.update(RC)


def mm(x):
    return x / 25.4


class Page:
    """A figure laid out in millimetres from the top-left corner, at its final printed size."""

    def __init__(self, w_mm, h_mm):
        setup()
        self.W, self.H = float(w_mm), float(h_mm)
        self.fig = plt.figure(figsize=(mm(self.W), mm(self.H)))
        self.letters = []

    def ax(self, x, y, w, h, **kw):
        """Axes whose plotting area spans x..x+w (mm from the left) and y..y+h (mm from the top)."""
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FS)
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def letter(self, x, y, s):
        """Panel letter with its cap top at (x, y) mm."""
        t = self.fig.text(x / self.W, 1 - y / self.H, s, fontsize=FS_LETTER, fontweight="bold",
                          ha="left", va="top")
        self.letters.append((s, x, y))
        return t

    def save(self, outdir, stem):
        os.makedirs(outdir, exist_ok=True)
        out = []
        for ext in ("pdf", "svg", "png"):
            p = f"{outdir}/{stem}.{ext}"
            self.fig.savefig(p, dpi=DPI, transparent=False)
            out.append(p)
        plt.close(self.fig)
        return out


def frame(ax, spines=("left", "bottom")):
    for s in ("left", "bottom", "top", "right"):
        ax.spines[s].set_visible(s in spines)
        ax.spines[s].set_linewidth(LW)
        ax.spines[s].set_color(BLACK)
    ax.tick_params(length=TICK, width=LW, pad=TICK_PAD, labelsize=FS, colors=BLACK)
    return ax


def refline(ax, *, x=None, y=None, **kw):
    kw = dict(dict(color=BLACK, lw=LW, ls=DASH, zorder=0.5), **kw)
    return ax.axvline(x, **kw) if x is not None else ax.axhline(y, **kw)
