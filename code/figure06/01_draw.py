#!/usr/bin/env python
"""Fig. 6: reusing RECAST programs to score single cells.

  a  change in balanced accuracy and macro-F1, RECAST against ANS and reference-free programs (panels_a_to_c.py)
  b  one CD8 T subtype and its program scored by the three methods on one embedding (panels_a_to_c.py)
  c  program scores of each subtype's own cells and of all other cells (panels_a_to_c.py)
  d  change in balanced accuracy as genes are added to the curated panels (panel_d.R)

Layout at the 183-mm build width, then scaled to 180 mm: a fills row 1; b, c and d stand in row 2, each panel at
100 % and cropped to its ink plus 1.2 pt, 1.5 mm apart; d's page is placed whole at D_DX, D_DY from the row's top-left
corner.  The white band between the rows is 12 px at 600 dpi on the 180-mm page.  Panel letters are 8 pt bold.

Run     python 01_draw.py        (runs Rscript once; RSCRIPT names it, default "Rscript")
Output  out/figure6.pdf (and the 183-mm composite), the panels in out/panels/
"""
import os
import subprocess
import sys

import cairosvg
import lxml.etree as ET
import svgutils.transform as sg
from pypdf import PdfReader, PdfWriter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import compose as CF                  # noqa: E402
import panels_a_to_c as P             # noqa: E402

OUT = os.path.join(HERE, "out")
PAN = os.path.join(OUT, "panels")
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

PT_PER_MM = 72 / 25.4
W183 = CF.TARGET_W                    # pt: 183 mm
MARGIN, STRIP, GAP = CF.MARGIN, CF.LETTER_STRIP, CF.COL_GAP     # 0.8 mm, 8.5 pt above each row, 1.5 mm
ROW2_H = 163.5698                     # pt: height of row 2
D_DX, D_DY = 289.576893, -0.120114    # pt: panel d's page relative to the top-left corner of row 2
PAGE_W = float(f"{W183:.2f}")
SCALE = 180 * PT_PER_MM / PAGE_W      # 183 mm -> 180 mm
ROW_SHIFT = 55 * 72 / 600 / SCALE     # pt: row 2 sits this much higher than 1.5 mm below row 1 (55 px at 600 dpi)


def _place(panel, px, py, idx):
    """A cropped panel with the top-left corner of its crop at (px, py), clipped to the crop."""
    pel = panel.element(px, py, 1.0, f"p{idx}_")
    clip = ET.Element(f"{{{CF.SVG_NS}}}clipPath")
    clip.set("id", f"clip_{idx}"); clip.set("clipPathUnits", "userSpaceOnUse")
    rc = ET.SubElement(clip, f"{{{CF.SVG_NS}}}rect")
    rc.set("x", f"{px:.2f}"); rc.set("y", f"{py:.2f}")
    rc.set("width", f"{panel.cw:.2f}"); rc.set("height", f"{panel.ch:.2f}")
    wrap = ET.Element(f"{{{CF.SVG_NS}}}g"); wrap.set("clip-path", f"url(#clip_{idx})")
    wrap.append(pel.root)
    return [sg.FigureElement(clip), sg.FigureElement(wrap)]


def _letter(x, y, L):
    return sg.TextElement(x, y, L, size=CF.LETTER_SZ, font=CF.LETTER_FONT, weight="bold")


def tile():
    CF.HERE = PAN                                                    # the panels' 600-dpi PNGs give their crops
    a, b, c = (CF.Panel(k, f"{PAN}/{k}.svg") for k in ("panel_a", "panel_b", "panel_c"))
    left = (W183 - a.cw) / 2.0                                       # row 1 is the widest row
    y1 = MARGIN + STRIP
    y2 = y1 + a.ch + GAP + STRIP - ROW_SHIFT
    xb, xc = left, left + b.cw + GAP
    xd = xc + c.cw + GAP                                             # panel d's letter
    elems = _place(a, left, y1, 0) + [_letter(left, y1, "a")]
    elems += _place(b, xb, y2, 1) + [_letter(xb, y2, "b")] + _place(c, xc, y2, 2) + [_letter(xc, y2, "c")]
    # d: its whole page, clipped to the page
    f = sg.fromfile(f"{PAN}/panel_d.svg")
    dw, dh = CF._pt(f.width), CF._pt(f.height)
    dx, dy = left + D_DX, y2 + D_DY
    assert dx + dw <= W183 - MARGIN + 0.5, "panel d runs past the right margin"
    root = sg.fromfile(CF._namespace_svg(f"{PAN}/panel_d.svg", "p3_")).getroot()
    root.moveto(dx, dy)
    clip = ET.Element(f"{{{CF.SVG_NS}}}clipPath")
    clip.set("id", "clip_3"); clip.set("clipPathUnits", "userSpaceOnUse")
    rc = ET.SubElement(clip, f"{{{CF.SVG_NS}}}rect")
    rc.set("x", f"{dx:.4f}"); rc.set("y", f"{dy:.4f}"); rc.set("width", f"{dw:.4f}"); rc.set("height", f"{dh:.4f}")
    wrap = ET.Element(f"{{{CF.SVG_NS}}}g"); wrap.set("clip-path", "url(#clip_3)")
    wrap.append(root.root)
    elems += [sg.FigureElement(clip), sg.FigureElement(wrap), _letter(xd, y2, "d")]

    H = y2 + ROW2_H + MARGIN
    fig = sg.SVGFigure(f"{W183:.2f}pt", f"{H:.2f}pt")
    bg = ET.Element(f"{{{CF.SVG_NS}}}rect")
    bg.set("x", "0"); bg.set("y", "0"); bg.set("width", f"{W183:.2f}"); bg.set("height", f"{H:.2f}")
    bg.set("fill", "#ffffff")
    fig.append([sg.FigureElement(bg)] + elems)
    fig.root.set("width", f"{W183:.2f}pt"); fig.root.set("height", f"{H:.2f}pt")
    fig.root.set("viewBox", f"0 0 {W183:.2f} {H:.2f}")
    svg = f"{OUT}/figure6_183mm.svg"
    fig.save(svg)
    cairosvg.svg2pdf(url=svg, write_to=f"{OUT}/figure6_183mm.pdf")
    w = PdfWriter()
    w.append(PdfReader(f"{OUT}/figure6_183mm.pdf"))
    w.pages[0].scale_by(SCALE)
    w.write(f"{OUT}/figure6.pdf")


if __name__ == "__main__":
    os.makedirs(PAN, exist_ok=True)
    P.OUT = PAN
    P.panel_a()
    P.panel_b()
    P.panel_c()
    subprocess.run([RSCRIPT, os.path.join(HERE, "panel_d.R")], check=True)
    tile()
