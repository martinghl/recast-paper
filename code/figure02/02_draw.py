#!/usr/bin/env python
"""Fig. 2: RECAST against reference-free attribution, and what its marker recovery comes from.

  a-c  panels_a_to_c.py     d, f, g  panels_d_f_g.py     e  panel_e.py
Run 01_marker_recovery.py first (panels c and d read its tables).

Layout at the 183-mm build width, then scaled to 180 mm: row 1 = a, b, c; row 2 = d, e, f, g; every panel at 100 %,
cropped to its ink plus 1.2 pt and 1.5 mm from its neighbour, both rows starting at one left edge (the wider row
centred).  Row 1 is as tall as a and c; the column labels of b hang below it.  The white band between the rows is
12 px at 600 dpi on the 180-mm page.  Panel letters are 8 pt bold, on the top edge of their panel.

Run     python 02_draw.py
Output  out/figure2.pdf (and the 183-mm composite), the panels in out/panels/
"""
import os
import sys

import cairosvg
import lxml.etree as ET
import svgutils.transform as sg
from pypdf import PdfReader, PdfWriter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import compose as CF                  # noqa: E402
import panels_a_to_c as ABC           # noqa: E402
import panels_d_f_g as DFG            # noqa: E402
import panel_e                        # noqa: E402

OUT = os.path.join(HERE, "out")
PAN = os.path.join(OUT, "panels")
ROWS = [["panel_a", "panel_b", "panel_c"], ["panel_d", "panel_e", "panel_f", "panel_g"]]

W183 = CF.TARGET_W                    # pt: 183 mm
MARGIN, STRIP, GAP = CF.MARGIN, CF.LETTER_STRIP, CF.COL_GAP     # 0.8 mm, 8.5 pt above each row, 1.5 mm
PAGE_W = float(f"{W183:.2f}")
SCALE = 180 * 72 / 25.4 / PAGE_W      # 183 mm -> 180 mm
ROW_SHIFT = 40 * 72 / 600 / SCALE     # pt: row 2 sits this much higher than 1.5 mm below row 1 (40 px at 600 dpi)


def _place(panel, px, py, idx, clip_h=None):
    """A cropped panel with the top-left corner of its crop at (px, py), clipped to the crop (or to clip_h)."""
    pel = panel.element(px, py, 1.0, f"p{idx}_")
    clip = ET.Element(f"{{{CF.SVG_NS}}}clipPath")
    clip.set("id", f"clip_{idx}"); clip.set("clipPathUnits", "userSpaceOnUse")
    rc = ET.SubElement(clip, f"{{{CF.SVG_NS}}}rect")
    rc.set("x", f"{px:.2f}"); rc.set("y", f"{py:.2f}")
    rc.set("width", f"{panel.cw:.2f}"); rc.set("height", f"{(clip_h or panel.ch):.2f}")
    wrap = ET.Element(f"{{{CF.SVG_NS}}}g"); wrap.set("clip-path", f"url(#clip_{idx})")
    wrap.append(pel.root)
    return [sg.FigureElement(clip), sg.FigureElement(wrap),
            sg.TextElement(px, py, panel.key[-1], size=CF.LETTER_SZ, font=CF.LETTER_FONT, weight="bold")]


def tile():
    CF.HERE = PAN                                                    # the panels' 600-dpi PNGs give their crops
    P = {k: CF.Panel(k, f"{PAN}/{k}.svg") for row in ROWS for k in row}
    widths = [sum(P[k].cw for k in row) + GAP * (len(row) - 1) for row in ROWS]
    assert max(widths) <= W183 - 2 * MARGIN, "a row is wider than the page"
    left = (W183 - max(widths)) / 2.0
    y1 = MARGIN + STRIP
    rh1 = max(P["panel_a"].ch, P["panel_c"].ch)
    y2 = y1 + rh1 + GAP + STRIP - ROW_SHIFT
    rh2 = max(P[k].ch for k in ROWS[1])
    elems, idx = [], 0
    for row, y in zip(ROWS, (y1, y2)):
        x = left
        for k in row:
            clip_h = P[k].h0 - P[k].cy if k == "panel_b" else None   # b: down to its canvas bottom (column labels)
            elems += _place(P[k], x, y, idx, clip_h)
            x += P[k].cw + GAP
            idx += 1
    H = y2 + rh2 + MARGIN
    fig = sg.SVGFigure(f"{W183:.2f}pt", f"{H:.2f}pt")
    bg = ET.Element(f"{{{CF.SVG_NS}}}rect")
    bg.set("x", "0"); bg.set("y", "0"); bg.set("width", f"{W183:.2f}"); bg.set("height", f"{H:.2f}")
    bg.set("fill", "#ffffff")
    fig.append([sg.FigureElement(bg)] + elems)
    fig.root.set("width", f"{W183:.2f}pt"); fig.root.set("height", f"{H:.2f}pt")
    fig.root.set("viewBox", f"0 0 {W183:.2f} {H:.2f}")
    svg = f"{OUT}/figure2_183mm.svg"
    fig.save(svg)
    cairosvg.svg2pdf(url=svg, write_to=f"{OUT}/figure2_183mm.pdf")
    w = PdfWriter()
    w.append(PdfReader(f"{OUT}/figure2_183mm.pdf"))
    w.pages[0].scale_by(SCALE)
    w.write(f"{OUT}/figure2.pdf")


if __name__ == "__main__":
    os.makedirs(PAN, exist_ok=True)
    for mod in (ABC, DFG):
        mod.OUT = PAN
    ABC.panel_a()
    ABC.panel_b()
    ABC.panel_c()
    DFG.panel_d()
    panel_e.draw("panel_e", PAN)
    DFG.panel_f()
    DFG.panel_g()
    tile()
