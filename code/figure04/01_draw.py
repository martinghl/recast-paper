#!/usr/bin/env python
"""Fig. 4: the choice of reference.

Six panels on one grid at the 183-mm build width: three columns of CW = 58.5 mm shared by both rows, so the letters
a/d, b/e and c/f stand in one column; every panel's page is exactly its cell (CW x 62 mm) and the cells are tiled
without scaling or cropping.  Row 1: a (schematic), b, c; row 2: d, e, f.  The page is then scaled to 180 mm.

  a  panel_a.py     b-f  panels_b_to_f.py

Run     python 01_draw.py
Output  out/figure4.pdf (and the 183-mm composite and the six panels in out/)
"""
import os
import sys

import cairosvg
import lxml.etree as ET
import numpy as np
import svgutils.transform as sg
from PIL import Image
from pypdf import PdfReader, PdfWriter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import compose as CF                  # noqa: E402
import panel_a                        # noqa: E402
import panels_b_to_f as P             # noqa: E402

OUT = os.path.join(HERE, "out")
PAN = os.path.join(OUT, "panels")

# ---- the grid (mm at the 183-mm build width) ----
W183 = 183.0
M = 0.8                       # outer margin
STRIP = 3.0                   # letter strip above each row: the letter's baseline is the row's cell top
CW = 58.5                     # column width = every panel's page width
G = (W183 - 2 * M - 3 * CW) / 2          # 2.95 mm between columns
RH = 62.0                     # row height = every panel's page height
RG = -0.814                   # from the bottom of row 1's cells to row 2's letter strip
S = 180.0 / 183.0
LETTER_PT = 8.0
GRID = [["a", "b", "c"], ["d", "e", "f"]]


def draw_panels():
    os.makedirs(PAN, exist_ok=True)
    P.OUT = PAN
    for fn in (P.panel_d, P.panel_e, P.panel_b, P.panel_c, P.panel_f):
        fn()
    panel_a.draw("panel_a", PAN, W=CW, H=RH)
    # every page must be exactly its cell, and no ink may reach the page edge (nothing is cut by the cell clip)
    for L in "abcdef":
        a = np.asarray(Image.open(f"{PAN}/panel_{L}.png").convert("RGB"))
        h, w = a.shape[:2]
        assert abs(w / 600 * 25.4 - CW) < 0.05 and abs(h / 600 * 25.4 - RH) < 0.05, (L, w, h)
        ys, xs = np.nonzero(np.any(a < 250, axis=2))
        edge = min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max()) / 600 * 25.4
        assert edge >= 0.1, f"panel {L}: ink {edge:.2f} mm from the page edge"


def tile():
    H183 = M + STRIP + RH + RG + STRIP + RH + M
    elems = []
    for r, row in enumerate(GRID):
        y = M + STRIP + r * (RH + RG + STRIP)
        for c, L in enumerate(row):
            x = M + c * (CW + G)
            px, py = CF.mm(x), CF.mm(y)
            root = sg.fromfile(CF._namespace_svg(f"{PAN}/panel_{L}.svg", f"p{L}_")).getroot()
            root.moveto(px, py)
            clip = ET.Element(f"{{{CF.SVG_NS}}}clipPath")
            clip.set("id", f"cell_{L}"); clip.set("clipPathUnits", "userSpaceOnUse")
            rc = ET.SubElement(clip, f"{{{CF.SVG_NS}}}rect")
            rc.set("x", f"{px:.3f}"); rc.set("y", f"{py:.3f}")
            rc.set("width", f"{CF.mm(CW):.3f}"); rc.set("height", f"{CF.mm(RH):.3f}")
            wrap = ET.Element(f"{{{CF.SVG_NS}}}g"); wrap.set("clip-path", f"url(#cell_{L})")
            wrap.append(root.root)
            elems += [sg.FigureElement(clip), sg.FigureElement(wrap),
                      sg.TextElement(px, py, L, size=LETTER_PT, font=CF.LETTER_FONT, weight="bold")]
    Wp, Hp = CF.mm(W183), CF.mm(H183)
    fig = sg.SVGFigure(f"{Wp:.2f}pt", f"{Hp:.2f}pt")
    bg = ET.Element(f"{{{CF.SVG_NS}}}rect")
    bg.set("x", "0"); bg.set("y", "0"); bg.set("width", f"{Wp:.2f}"); bg.set("height", f"{Hp:.2f}")
    bg.set("fill", "#ffffff")
    fig.append([sg.FigureElement(bg)] + elems)
    fig.root.set("width", f"{Wp:.2f}pt"); fig.root.set("height", f"{Hp:.2f}pt")
    fig.root.set("viewBox", f"0 0 {Wp:.2f} {Hp:.2f}")
    svg = f"{OUT}/figure4_183mm.svg"
    fig.save(svg)
    cairosvg.svg2pdf(url=svg, write_to=f"{OUT}/figure4_183mm.pdf")
    pg = PdfReader(f"{OUT}/figure4_183mm.pdf").pages[0]          # 183 mm -> 180 mm
    pg.scale_by(S)
    wr = PdfWriter(); wr.add_page(pg)
    with open(f"{OUT}/figure4.pdf", "wb") as fh:
        wr.write(fh)


if __name__ == "__main__":
    draw_panels()
    tile()
