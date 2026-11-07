"""Shared by the marker-recovery panels of Fig. 2 (c, d, f, g): their y-axis label and the P-value brackets."""
import math

import matplotlib.text as mtext
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

from method_colours import INK

LABEL = "Marker recovery\n(20 selected genes)"


def ylabel(fig, ax):
    """Set the two-line y label (6.2 pt, line spacing 1.2) and keep >= 0.3 mm of canvas left of it: when it would
    come closer, the axes move right by the difference, their right edge fixed."""
    yl = ax.yaxis.label
    yl.set_text(LABEL)
    yl.set_fontsize(6.2)
    yl.set_linespacing(1.2)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = mtext.Text.get_window_extent(yl, rend)
    need = 0.3 - bb.x0 / fig.dpi * 25.4
    if need > 0:
        x0, y0, w, h = ax.get_position().bounds
        dx = need / (fig.get_figwidth() * 25.4)
        ax.set_position([x0 + dx, y0, w - dx, h])
        fig.canvas.draw()
        bb = mtext.Text.get_window_extent(yl, rend)
    ticks = [t.get_window_extent(rend) for t in ax.get_yticklabels() if t.get_text()]
    assert bb.x0 >= 0, "the y label runs off the canvas"
    assert (min(t.x0 for t in ticks) - bb.x1) / fig.dpi * 25.4 > 0.4, "the y label touches its tick labels"


def ptxt(p):
    """Two significant figures in decimal notation (P = 0.012, 0.00012, 0.44)."""
    assert 0 < p <= 1, p
    return f"{p:.{max(2, 1 - math.floor(math.log10(p)))}f}"


def p_label(ax, x, y, text, size=5.6):
    """'P = ...' with P in italics, centred on x, its bottom at y (data coordinates)."""
    box = HPacker(children=[TextArea("P", textprops=dict(fontsize=size, color=INK, style="italic")),
                            TextArea("= " + text, textprops=dict(fontsize=size, color=INK))], align="baseline", pad=0,
                  sep=size * 0.278)   # a space's width in Liberation Sans (a leading space is dropped)
    ax.add_artist(AnnotationBbox(box, (x, y), xycoords="data", frameon=False, box_alignment=(0.5, 0.0), pad=0,
                                 annotation_clip=False, zorder=7))


def bracket(ax, xa, xb, y, tick, text):
    """A bracket from xa to xb at height y, its ends `tick` lower, with 'P = text' above it."""
    ax.plot([xa, xa, xb, xb], [y - tick, y, y, y - tick], color=INK, lw=0.5, zorder=6, solid_capstyle="butt")
    p_label(ax, (xa + xb) / 2, y + tick * 0.6, text)
