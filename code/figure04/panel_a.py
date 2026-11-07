"""Fig. 4a: schematic of the choice of reference.

A target subtype T is attributed along the path from the mean expression profile of a reference population R to that
of T; the reference can be a sibling subtype, an embedding-based reference, the nearest population, the broad
lineage, the whole atlas, a random set of cells or an unrelated population.  The panel is one cell of the Fig. 4 grid
(58.5 x 62 mm), drawn in millimetre coordinates.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from method_colours import (apply_style, mm, save_fig, INK, SUB, HAIR, G550, G400,    # noqa: E402
                            RECAST, RECAST_D, RECAST_L, RECAST_XL, WARN)

apply_style()
BAND = "#F6F8FA"          # background of the reference lane


def _mk(ax):
    """Drawing helpers on an axes in millimetre coordinates: band, card (rounded box), txt, arr (arrow)."""
    def band(x0, y0, x1, y1):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=1.1",
                                    facecolor=BAND, edgecolor=HAIR, linewidth=0.6,
                                    zorder=0.5))

    def card(x0, y0, x1, y1, fc="white", ec=INK, lw=0.8, ls="solid", z=3):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.9",
                                    facecolor=fc, edgecolor=ec, linewidth=lw,
                                    linestyle=ls, zorder=z))

    def txt(x, y, s, fs=5.8, c=INK, bold=False, ha="center", va="center", rot=0, z=6):
        ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, rotation=rot,
                zorder=z, fontweight="bold" if bold else "normal")

    def arr(x0, y0, x1, y1, color=SUB, lw=0.9, ms=6.0, sa=0.0, sb=0.0, z=4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=ms, lw=lw, color=color,
                                     zorder=z, shrinkA=sa, shrinkB=sb))
    return band, card, txt, arr


def draw(stem, outdir, W=58.5, H=62.0):
    """Draw the panel on a W x H mm page; x positions are laid out on 46 mm and stretched to W."""
    X = lambda x: x * W / 46.0                                          # noqa: E731
    fig = plt.figure(figsize=(mm(W), mm(H)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    band, card, txt, arr = _mk(ax)

    card(X(3.0), 52.2, X(43.0), 61.4, fc=RECAST_XL, ec=RECAST_D, lw=1.0)
    _off = [(-5.4, 0.3), (-4.0, 1.5), (-3.4, -1.2), (-1.8, 0.7), (-0.6, -1.5),
            (0.3, 1.6), (1.6, -0.3), (2.8, 1.3), (3.2, -1.4), (4.8, 0.5), (5.8, -0.8)]
    ax.scatter([X(23.0) + dx for dx, _ in _off], [57.9 + dy for _, dy in _off], s=4.5, color=RECAST, lw=0, zorder=5)
    txt(X(23.0), 53.8, "target subtype $T$", fs=6.0, c=RECAST_D, bold=True)
    arr(X(23.0), 52.2, X(23.0), 49.75, color=G550, lw=1.0, ms=6.5)

    card(X(3.0), 40.2, X(43.0), 49.6, fc="white", ec=RECAST_L, lw=0.9)
    txt(X(23.0), 47.6, "Attribution path", fs=5.9, c=SUB)
    txt(X(23.0), 44.6, r"$C_R \rightarrow C_T$", fs=8.2, c=RECAST_D)
    txt(X(23.0), 41.8, "$C$ : mean expression profile", fs=5.8, c=SUB)

    Z0, Z1 = 0.5, 37.6
    band(X(0.5), Z0, X(45.5), Z1)
    txt(X(9.0), 35.9, "reference $R$ — a real population", fs=6.1, bold=True, ha="left")
    ax.add_patch(PathPatch(MPath([(X(23.0), 40.2), (X(23.0), 38.9), (X(5.5), 38.9), (X(5.5), 3.0)]),
                           fill=False, edgecolor=G550, lw=0.9, capstyle="round", joinstyle="round", zorder=2))
    labels = ["sibling (matched)", "geometry (embedding)", "nearest", "broad lineage",
              "global atlas", "random", "unrelated"]
    lcol = [RECAST_D, RECAST_D, INK, INK, SUB, SUB, WARN]
    lwt = [True, True, False, False, False, False, False]
    ys = [32.4, 27.5, 22.6, 17.7, 12.8, 7.9, 3.0]
    for i, (yy, lab) in enumerate(zip(ys, labels)):
        hot = i == 0
        card(X(9.0), yy - 2.0, X(43.5), yy + 2.0, fc=RECAST_XL if hot else "white",
             ec=RECAST_D if hot else G400, lw=0.9 if hot else 0.7)
        txt(X(26.25), yy, lab, fs=5.9, c=lcol[i], bold=lwt[i])
        if hot:
            arr(X(5.5), yy, X(8.9), yy, color=RECAST, lw=1.2, ms=6.5, z=5)
        else:
            ax.plot([X(5.5), X(8.9)], [yy, yy], color=G400, lw=0.7, zorder=2)
    return save_fig(fig, stem, outdir)
