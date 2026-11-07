"""Panel composition for the main figures: place panel SVGs on one page and write SVG, PDF and a 600-dpi PNG.

  Panel(key, path)   a panel cropped to its ink (+ 1.2 pt), read from HERE/<key>.png; place it with .element()
  compose()          rows of panels across the 183-mm width, scaled down to fit, letters in a strip above each row
  compose_placed()   every panel at 100 % at an explicit x, rows a fixed distance apart ink to ink (Fig. 5)
  _ink_box_mm(), _axis_frame_mm()   a panel's ink box and axis lines, measured on its PNG
Panel SVG ids are namespaced per panel (matplotlib and svglite reuse clip and glyph ids), and every panel is clipped to
its crop.  Callers set HERE (where the PNGs are) and OUT (where the figure goes).
"""
import os, re, tempfile
import numpy as np
from PIL import Image
import svgutils.transform as sg
import lxml.etree as ET
import cairosvg

SVG_NS = "http://www.w3.org/2000/svg"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE

PT_PER_MM = 72.0 / 25.4
def mm(x): return x * PT_PER_MM

TARGET_W   = mm(183.0)           # full double-column text width
COL_GAP    = mm(1.5)             # between panels in a row (content-to-content)
ROW_GAP    = mm(1.5)             # between rows (content-to-content)
MARGIN     = mm(0.8)             # outer margin
LETTER_STRIP = 8.5               # pt reserved above each row for its letters
LETTER_SZ  = 8.0                 # pt, bold lowercase (constant, not scaled)
LETTER_FONT = "Arial"
TEXT_FLOOR_PT = 5.0              # the smallest text size the panels are drawn with
CROP_PAD_PT = 1.2                # keep a hair of white around each panel's ink

# ---- id namespacing: ids and references in either quote style (svglite writes id='...')
_ID_ATTR = re.compile(r'''\bid=["']([^"']+)["']''')
_ID_REFS = [re.compile(r'url\(#([^)]+)\)'), re.compile(r'''(?:xlink:)?href=["']#([^"']+)["']''')]
def _namespace_svg(path, pfx):
    with open(path) as f:
        txt = f.read()
    ids = set(_ID_ATTR.findall(txt))
    for rx in _ID_REFS:
        ids.update(rx.findall(txt))
    for _id in sorted(ids, key=len, reverse=True):
        new = pfx + _id
        txt = txt.replace(f'url(#{_id})', f'url(#{new})')
        for q in ('"', "'"):
            txt = txt.replace(f'id={q}{_id}{q}', f'id={q}{new}{q}')
            txt = txt.replace(f'xlink:href={q}#{_id}{q}', f'xlink:href={q}#{new}{q}')
            txt = txt.replace(f'href={q}#{_id}{q}', f'href={q}#{new}{q}')
    tmp = tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w")
    tmp.write(txt); tmp.close()
    return tmp.name

def _pt(s): return float(re.match(r"([\d.]+)", str(s)).group(1))

class Panel:
    def __init__(self, key, path):
        self.key, self.path = key, path
        f = sg.fromfile(path)
        self.w0, self.h0 = _pt(f.width), _pt(f.height)
        # crop to the panel's ink, so its blank margins do not widen the gaps between panels
        self.cx, self.cy, self.cw, self.ch = self._content_bbox(key)

    def _content_bbox(self, key):
        png = os.path.join(HERE, key + ".png")     # exact-size 600-dpi render (HERE is set by the caller)
        if not os.path.exists(png):
            return 0.0, 0.0, self.w0, self.h0
        im = Image.open(png).convert("L")
        a = np.asarray(im)
        mask = a < 250                              # non-white ink
        ys, xs = np.where(mask)
        if xs.size == 0:
            return 0.0, 0.0, self.w0, self.h0
        W, H = im.size
        x0, x1 = xs.min() / W * self.w0, (xs.max() + 1) / W * self.w0
        y0, y1 = ys.min() / H * self.h0, (ys.max() + 1) / H * self.h0
        # pad, clamp to panel
        x0 = max(0.0, x0 - CROP_PAD_PT); y0 = max(0.0, y0 - CROP_PAD_PT)
        x1 = min(self.w0, x1 + CROP_PAD_PT); y1 = min(self.h0, y1 + CROP_PAD_PT)
        return x0, y0, x1 - x0, y1 - y0

    def element(self, px, py, s, pfx):
        """Place so the panel's content-bbox top-left lands at (px,py) at scale s."""
        root = sg.fromfile(_namespace_svg(self.path, pfx)).getroot()
        if abs(s - 1.0) > 1e-6:
            root.scale(s)                           # -> transform: scale(s)
        root.moveto(px - self.cx * s, py - self.cy * s)   # translate outer
        return root

def _stamp_png_dpi(path, dpi):
    """Give a PNG an explicit resolution (pHYs chunk, pixels per metre). cairosvg renders at the requested dpi
    but omits the chunk, so readers report 0/72 dpi for a 600-dpi image. Byte-level chunk surgery: no re-encode,
    the decoded pixels (what the reproduction package checksums) are untouched."""
    import struct, zlib
    with open(path, "rb") as fh:
        b = fh.read()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", path
    ppm = int(round(dpi / 0.0254))
    data = struct.pack(">IIB", ppm, ppm, 1)
    phys = struct.pack(">I", len(data)) + b"pHYs" + data + struct.pack(">I", zlib.crc32(b"pHYs" + data) & 0xFFFFFFFF)
    out, i, done = [b[:8]], 8, False
    while i < len(b):                                   # walk the chunks: drop any existing pHYs, add ours after IHDR
        n = struct.unpack(">I", b[i:i + 4])[0]; typ = b[i + 4:i + 8]; chunk = b[i:i + 12 + n]; i += 12 + n
        if typ == b"pHYs":
            continue
        out.append(chunk)
        if typ == b"IHDR" and not done:
            out.append(phys); done = True
    with open(path, "wb") as fh:
        fh.write(b"".join(out))


def compose(fig_name, rows, letters, panels_dir=None, verbose=True, row_gap=None, letter_strip=None, margin=None):
    """Rows of panels across TARGET_W: each panel cropped to its ink, a row scaled down (never up) to fit the width,
    all rows starting at the left edge of the widest one; letters[key] stands on the top edge of its panel.
    row_gap, letter_strip and margin (pt) default to the module constants.  Writes OUT/fig_name.{svg,pdf,png}."""
    ROW_GAP_ = ROW_GAP if row_gap is None else row_gap
    LETTER_STRIP_ = LETTER_STRIP if letter_strip is None else letter_strip
    MARGIN_ = MARGIN if margin is None else margin
    pdir = panels_dir or HERE
    P = {}
    def load(k):
        if k not in P:
            P[k] = Panel(k, os.path.join(pdir, k + ".svg"))
        return P[k]
    for row in rows:
        for k in row:
            load(k)

    canvas_w = TARGET_W
    elems, pidx, y = [], 0, MARGIN_
    row_scales = []
    # one common left edge, from the widest row: column 1 and its letters align, the block stays centred
    _w = []
    for _row in rows:
        _n = len(_row)
        _sw = sum(load(k).cw for k in _row)
        _rs = min(1.0, (TARGET_W - 2 * MARGIN_ - COL_GAP * (_n - 1)) / _sw)
        _w.append(_sw * _rs + COL_GAP * (_n - 1))
    _COMMON_LEFT = (TARGET_W - max(_w)) / 2.0
    for row in rows:
        n = len(row)
        avail = TARGET_W - 2 * MARGIN_ - COL_GAP * (n - 1)
        sw = sum(load(k).cw for k in row)          # cropped content widths
        # fill the width but NEVER scale a row UP (that makes narrow rows tall);
        # a row narrower than the target is centered instead.
        rs = min(1.0, avail / sw)
        row_scales.append(rs)
        rh = max(load(k).ch * rs for k in row)     # cropped content heights
        row_w = sw * rs + COL_GAP * (n - 1)
        x = _COMMON_LEFT
        for k in row:
            P_k = load(k)
            w, h = P_k.cw * rs, P_k.ch * rs
            px, py = x, y + LETTER_STRIP_
            pel = P_k.element(px, py, rs, f"p{pidx}_")
            clip_id = f"clip_{pidx}"
            clip = ET.Element(f"{{{SVG_NS}}}clipPath")
            clip.set("id", clip_id); clip.set("clipPathUnits", "userSpaceOnUse")
            rc = ET.SubElement(clip, f"{{{SVG_NS}}}rect")
            rc.set("x", f"{px:.2f}"); rc.set("y", f"{py:.2f}")
            rc.set("width", f"{w:.2f}"); rc.set("height", f"{h:.2f}")
            wrap = ET.Element(f"{{{SVG_NS}}}g"); wrap.set("clip-path", f"url(#{clip_id})")
            wrap.append(pel.root)
            elems.append(sg.FigureElement(clip)); elems.append(sg.FigureElement(wrap))
            # the letter stands on the panel's top edge (y + LETTER_STRIP_; = y + LETTER_SZ + 0.5 by default)
            elems.append(sg.TextElement(x, y + LETTER_STRIP_, letters.get(k, "?"),
                                        size=LETTER_SZ, font=LETTER_FONT, weight="bold"))
            x += w + COL_GAP
            pidx += 1
        y += LETTER_STRIP_ + rh + ROW_GAP_
    canvas_h = y - ROW_GAP_ + MARGIN_

    fig = sg.SVGFigure(f"{canvas_w:.2f}pt", f"{canvas_h:.2f}pt")
    bg = ET.Element(f"{{{SVG_NS}}}rect")
    bg.set("x", "0"); bg.set("y", "0")
    bg.set("width", f"{canvas_w:.2f}"); bg.set("height", f"{canvas_h:.2f}")
    bg.set("fill", "#ffffff")
    fig.append([sg.FigureElement(bg)] + elems)
    fig.root.set("width", f"{canvas_w:.2f}pt")
    fig.root.set("height", f"{canvas_h:.2f}pt")
    fig.root.set("viewBox", f"0 0 {canvas_w:.2f} {canvas_h:.2f}")
    svg = os.path.join(OUT, f"{fig_name}.svg"); fig.save(svg)
    cairosvg.svg2pdf(url=svg, write_to=os.path.join(OUT, f"{fig_name}.pdf"))
    cairosvg.svg2png(url=svg, write_to=os.path.join(OUT, f"{fig_name}.png"), dpi=600)
    _stamp_png_dpi(os.path.join(OUT, f"{fig_name}.png"), 600)   # cairosvg writes no pHYs: declare 600 dpi
    min_pt = TEXT_FLOOR_PT * min(row_scales)
    if verbose:
        print(f"[{fig_name}] {canvas_w/PT_PER_MM:6.1f} x {canvas_h/PT_PER_MM:6.1f} mm  "
              f"aspect {canvas_w/canvas_h:4.2f}:1  rows {len(rows)}  "
              f"row-scales {[round(s,2) for s in row_scales]}  min-text ~{min_pt:.1f}pt")
    return canvas_w / PT_PER_MM, canvas_h / PT_PER_MM, min_pt


def _ink_box_mm(key, pdir):
    """(x0, y0, x1, y1) of the panel's ink on its own page, in mm, from its exact-size 600-dpi render."""
    f = sg.fromfile(os.path.join(pdir, key + ".svg"))
    w0, h0 = _pt(f.width) / PT_PER_MM, _pt(f.height) / PT_PER_MM
    im = Image.open(os.path.join(pdir, key + ".png")).convert("L")
    a = np.asarray(im)
    ys, xs = np.where(a < 250)
    W, H = im.size
    return (xs.min() / W * w0, ys.min() / H * h0, (xs.max() + 1) / W * w0, (ys.max() + 1) / H * h0), (w0, h0)



def compose_placed(fig_name, width_mm, rows, margin_mm, row_gap_mm, letter_pt, letter_base_mm, tol_mm=0.15,
                   pdir=None, verbose=True):
    """rows: [[(key, letter, x_left or None, x_right or None, letter_x[, dy]), ...], ...] -- each panel's ink is placed
    at 100 % with its left edge on x_left (or its right edge on x_right; given both, the ink width must match within
    tol_mm), its top on the row's top (or dy mm below it); rows are row_gap_mm apart
    ink to ink.  letter_x: the letter's left edge (mm); its baseline is letter_base_mm below the panel's top, so a
    10.5 pt letter's cap height (2.65 mm) meets the panel's first ink."""
    pdir = pdir or HERE
    elems, y, pidx, report = [], margin_mm, 0, []
    for row in rows:
        boxes = {}
        for item in row:
            key, L, xl, xr, lx = item[:5]
            dy = item[5] if len(item) > 5 else 0.0
            (x0, y0, x1, y1), (w0, h0) = _ink_box_mm(key, pdir)
            iw, ih = x1 - x0, y1 - y0
            if xl is not None and xr is not None:
                assert abs((xr - xl) - iw) <= tol_mm, f"{key}: ink {iw:.2f} mm wide, slot {xr - xl:.2f} mm"
            tx = xl if xl is not None else xr - iw
            ty = y + dy
            boxes[key] = (tx, ty, tx + iw, ty + ih)
            # the panel's page is moved so that its ink box lands on (tx, ty); clipped to its ink + the usual pad
            px, py = mm(tx - x0), mm(ty - y0)
            root = sg.fromfile(_namespace_svg(os.path.join(pdir, key + ".svg"), f"p{pidx}_")).getroot()
            root.moveto(px, py)
            pad = CROP_PAD_PT / PT_PER_MM
            cx0, cy0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
            cx1, cy1 = min(w0, x1 + pad), min(h0, y1 + pad)
            clip_id = f"clip_{pidx}"
            clip = ET.Element(f"{{{SVG_NS}}}clipPath")
            clip.set("id", clip_id); clip.set("clipPathUnits", "userSpaceOnUse")
            rc = ET.SubElement(clip, f"{{{SVG_NS}}}rect")
            rc.set("x", f"{px + mm(cx0):.3f}"); rc.set("y", f"{py + mm(cy0):.3f}")
            rc.set("width", f"{mm(cx1 - cx0):.3f}"); rc.set("height", f"{mm(cy1 - cy0):.3f}")
            wrap = ET.Element(f"{{{SVG_NS}}}g"); wrap.set("clip-path", f"url(#{clip_id})")
            wrap.append(root.root)
            elems += [sg.FigureElement(clip), sg.FigureElement(wrap),
                      sg.TextElement(mm(lx), mm(ty + letter_base_mm), L, size=letter_pt, font=LETTER_FONT,
                                     weight="bold")]
            report.append(f"{L} {key}: ink x {tx:.2f}-{tx + iw:.2f}, y {ty:.2f}-{ty + ih:.2f} mm ({iw:.2f} x {ih:.2f})")
            pidx += 1
        y = max(b[3] for b in boxes.values()) + row_gap_mm
    height_mm = y - row_gap_mm + margin_mm
    W, H = mm(width_mm), mm(height_mm)
    fig = sg.SVGFigure(f"{W:.2f}pt", f"{H:.2f}pt")
    bg = ET.Element(f"{{{SVG_NS}}}rect")
    bg.set("x", "0"); bg.set("y", "0"); bg.set("width", f"{W:.2f}"); bg.set("height", f"{H:.2f}")
    bg.set("fill", "#ffffff")
    fig.append([sg.FigureElement(bg)] + elems)
    fig.root.set("width", f"{W:.2f}pt"); fig.root.set("height", f"{H:.2f}pt")
    fig.root.set("viewBox", f"0 0 {W:.2f} {H:.2f}")
    svg = os.path.join(OUT, f"{fig_name}.svg"); fig.save(svg)
    cairosvg.svg2pdf(url=svg, write_to=os.path.join(OUT, f"{fig_name}.pdf"))
    cairosvg.svg2png(url=svg, write_to=os.path.join(OUT, f"{fig_name}.png"), dpi=600)
    _stamp_png_dpi(os.path.join(OUT, f"{fig_name}.png"), 600)
    if verbose:
        print(f"[{fig_name}] {width_mm:.1f} x {height_mm:.1f} mm, every panel at 100 %, letters {letter_pt} pt")
        for r in report:
            print("   " + r)
    return width_mm, height_mm


def _axis_frame_mm(key, pdir=None, min_run_mm=20.0):
    """A panel's two axis lines, in mm from the top left of its ink: the longest dark vertical run (the y axis: its
    x, top and bottom) and the longest dark horizontal run (the x axis: its y, left and right ends)."""
    pdir = pdir or HERE
    (x0, y0, _, _), (w0, _) = _ink_box_mm(key, pdir)
    im = np.asarray(Image.open(os.path.join(pdir, key + ".png")).convert("L"))
    px_mm = im.shape[1] / w0
    dark = (im < 80).astype(np.int8)

    def longest(a):                           # (length, start, line index) of the longest run of dark pixels
        best = (0, 0, 0)
        for i, line in enumerate(a):
            d = np.diff(np.concatenate([[0], line, [0]]))
            s, e = np.where(d == 1)[0], np.where(d == -1)[0]
            if s.size:
                j = int(np.argmax(e - s))
                if e[j] - s[j] > best[0]:
                    best = (int(e[j] - s[j]), int(s[j]), i)
        return best
    lv, sv, cv = longest(dark.T)
    lh, sh, rh = longest(dark)
    assert min(lv, lh) >= min_run_mm * px_mm, f"{key}: no axis line of {min_run_mm} mm"
    f = lambda p: p / px_mm
    return dict(y_axis_x=f(cv + 0.5) - x0, y_top=f(sv) - y0, y_bot=f(sv + lv) - y0,
                x_axis_y=f(rh + 0.5) - y0, x_left=f(sh) - x0, x_right=f(sh + lh) - x0)
