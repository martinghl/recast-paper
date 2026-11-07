#!/usr/bin/env python
"""Fig. 5: biological programs recovered by RECAST and by COSG.

  a  enrichment of 18 curated programs per subtype, with the expected pairs outlined (panel_a.py)
  b  per expected pair, program genes in RECAST's top 30 minus those in COSG's (panels_b_c.R)
  c  expected pairs enriched by the genes only one method's top 30 holds (panels_b_c.R)

Layout: a fills the width (ink 0.4-179.6 mm); b and c share the row below it (4.4-179.6 mm), both at 100 %, with the
same plot size so that their columns share one pitch.  The white band between the two rows is 12 px at 600 dpi
(0.51 mm).  Panel letters are 10.5 pt bold.

Run     python 01_draw.py        (runs Rscript once; RSCRIPT names it, default "Rscript")
Output  out/figure5.pdf (and .svg, .png), the panels in out/panels/
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
import compose as CF                  # noqa: E402
import panel_a                        # noqa: E402

OUT = os.path.join(HERE, "out")
PAN = os.path.join(OUT, "panels")
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

LETTER_PT = 10.5
LETTER_BASE = 0.716 * LETTER_PT / CF.PT_PER_MM        # a letter's cap height: its baseline meets the panel's first ink
ROW_X0, ROW_X1 = 4.4, 179.6                           # b's ink starts, c's ink ends
ROW_GAP = 5.0 - 109 * 25.4 / 600                      # mm between the rows' inks: leaves a 12-px band at 600 dpi


def layout():
    """Place a over b | c.  Checked: b and c have the same plot height and width (one column pitch), their x axes
    on one line, the same ink height, nothing of b beyond its x axis, and 5-8 mm between their inks (c's letter sits
    in it, 4 mm left of c's ink)."""
    fb, fc = CF._axis_frame_mm("panel_b", PAN), CF._axis_frame_mm("panel_c", PAN)
    (bx0, by0, bx1, by1), _ = CF._ink_box_mm("panel_b", PAN)
    (cx0, cy0, cx1, cy1), _ = CF._ink_box_mm("panel_c", PAN)
    bw, cw = bx1 - bx0, cx1 - cx0
    c_left = ROW_X1 - cw
    gap = c_left - (ROW_X0 + bw)
    hb, hc = fb["x_axis_y"] - fb["y_top"], fc["x_axis_y"] - fc["y_top"]
    wb, wc = fb["x_right"] - fb["y_axis_x"], fc["x_right"] - fc["y_axis_x"]
    assert abs(hb - hc) <= 0.1, f"plot heights {hb:.2f} / {hc:.2f} mm"
    assert abs(wb - wc) <= 0.1, f"plot widths {wb:.2f} / {wc:.2f} mm"
    assert abs(fb["x_axis_y"] - fc["x_axis_y"]) <= 0.1, f"x axes at {fb['x_axis_y']:.2f} / {fc['x_axis_y']:.2f} mm"
    assert abs((by1 - by0) - (cy1 - cy0)) <= 0.1, f"ink heights {by1 - by0:.2f} / {cy1 - cy0:.2f} mm"
    assert bw - fb["x_right"] <= 0.2, f"panel b reaches {bw - fb['x_right']:.2f} mm beyond its x axis"
    assert 5.0 <= gap <= 8.0, f"{gap:.2f} mm between panels b and c"
    rows = [[("panel_a", "a", 0.4, 179.6, 0.4)],
            [("panel_b", "b", ROW_X0, None, 0.4), ("panel_c", "c", None, ROW_X1, c_left - 4.0)]]
    CF.OUT = OUT
    CF.compose_placed("figure5", 180.0, rows, margin_mm=0.4, row_gap_mm=ROW_GAP, letter_pt=LETTER_PT,
                      letter_base_mm=LETTER_BASE, pdir=PAN, verbose=False)


if __name__ == "__main__":
    os.makedirs(PAN, exist_ok=True)
    panel_a.draw("panel_a", PAN)
    subprocess.run([RSCRIPT, os.path.join(HERE, "panels_b_c.R")], check=True)
    layout()
