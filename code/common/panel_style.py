"""House style for the matplotlib panels of the main figures: the look of Fig. 6d, applied to every panel.

Fig. 6d is drawn in R with cowplot::theme_cowplot(font_size = 6.2, line_size = 0.5 * 6.2 / 14) (figure6/), which gives:
  axis lines       #000000, 0.47 pt, left and bottom only
  ticks            #000000, 0.47 pt, 1.55 pt long (font_size / 4), outside
  tick labels      6.2 pt, #000000, 1.33 pt from the tick
  axis titles      6.2 pt, #000000, 1.55 pt from the tick labels
  reference line   #000000, 0.47 pt, dashed 4 pt on / 4 pt off
  separator line   #D9D9D9, 0.38 pt
  text             Liberation Sans 6.2 pt, #000000
  panel letters    8 pt bold (set by compose.py)

activate() is called by style.apply_style(), i.e. whenever a panel module is imported.  It sets these values as
rcParams and wraps the matplotlib setters the panel modules use, so whatever size or grey a module asks for, the artist
already carries the house value -- before the module measures text to place labels or checks that nothing overlaps.
Figure.savefig applies the same rules once more, for values a module wrote into rcParams after import.

Coerced:
  text       size -> 6.2 pt (a panel letter -> 8 pt); a grey or near-black colour -> #000000.  Hues (method
             identity: RECAST blue, SIGnature sage, COSG orange ...) and white are kept.
  spines     -> #000000, 0.47 pt (a hued spine keeps its hue).  Which spines are visible stays the panel module's choice,
             so closed frames stay closed.
  ticks      -> #000000, 0.47 pt; a visible tick is 1.55 pt long (minor 0.9 pt); a hidden one (length 0) stays hidden.
  ref lines  axhline / axvline / axline, and any two-point dashed grey line at least 14 pt long: -> #000000 (a hued
             one keeps its hue),
             0.47 pt, and a dashed or dotted one -> 4 pt on / 4 pt off.  A solid light-grey separator line becomes
             6d's hairline (#D9D9D9, 0.38 pt); white stays white.
  keys       legend text 6.2 pt.  A legend handle or a hand-drawn two-point key sample styled exactly like a
             converted reference line gets the same new style; any other short sample (e.g. a dashed data curve's
             key) keeps its look, so a key always matches what it explains.
  arrows     a grey or black arrow (FancyArrowPatch, annotate arrows) -> #000000, 0.47 pt.
Not touched: data marks -- points, bars, boxes, violins, whiskers, data lines, heatmaps, fills, colour maps --, and
the other patches (schematic boxes, frames drawn as rectangles).  Those are set in the panel modules where needed.

keep(artist, color=True, size=False) exempts one artist and restores what the module asked for (e.g. grey text whose
greyness stands for a category in the key).
PANEL_STYLE_OFF=1 in the environment turns all of this off.  The Supplementary Figure scripts do that, because they set
their own sizes (supplementary_style.py).
"""
import os

import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import axis as maxis
from matplotlib import figure as mfigure
from matplotlib import legend as mlegend
from matplotlib import lines as mlines
from matplotlib import spines as mspines
from matplotlib import text as mtext
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch

FS, LETTER = 6.2, 8.0                       # pt: every text / a panel letter
LW, HAIR_LW = 0.47, 0.38                    # pt: axis lines, ticks, reference lines / 6d's grid hairline
TICK, TICK_MINOR = 1.55, 0.9                # pt
PAD, LABELPAD = 1.33, 1.55                  # pt: tick -> tick label, tick labels -> axis title
BLACK, HAIR = "#000000", "#D9D9D9"
DASH = (0, (4 / LW, 4 / LW))                # matplotlib scales dash lengths by the line width: 4 pt on / 4 pt off

ACTIVE = os.environ.get("PANEL_STYLE_OFF", "") != "1"
_done = False


# ---------------------------------------------------------------------------------------------------- colour tests
def _rgba(c):
    try:
        return mcolors.to_rgba(c)
    except (ValueError, TypeError):
        return None


def neutral(c, lum_max=0.80):
    """True for an opaque grey or black no lighter than lum_max (so white and the lightest greys are not neutral)."""
    v = _rgba(c)
    if v is None or v[3] == 0:
        return False
    r, g, b = v[:3]
    return max(r, g, b) - min(r, g, b) < 0.06 and max(r, g, b) <= lum_max


def light_grey(c):
    """A light grey but not white: a hairline colour.  The bound is G300 (#BEBEBE, luminance 0.745) and lighter: the
    panel modules draw separators, heatmap cell grids and subtle zero rules in G300/G200/HAIR, and those stay hairlines
    (no grid or rule is turned into an axis-weight black line)."""
    return neutral(c, 0.97) and not neutral(c, 0.74)


# ------------------------------------------------------------------------------------------------------------- text
_T_set_color = mtext.Text.set_color
_T_set_fontsize = mtext.Text.set_fontsize
_T_set_fontproperties = mtext.Text.set_fontproperties


def _is_letter(t, size):
    try:
        size = float(size)
    except (TypeError, ValueError):
        return False
    s = t.get_text()
    return len(s) == 1 and s.isascii() and s.isalpha() and s.islower() and size >= 7.9


def _want_size(t, size):
    if getattr(t, "_house_keep_size", False):
        return size
    return LETTER if _is_letter(t, size) else FS


def _set_fontsize(self, fontsize):
    self._house_req_size = fontsize
    _T_set_fontsize(self, _want_size(self, fontsize))


def _set_fontproperties(self, fp):
    _T_set_fontproperties(self, fp)
    sz = self._fontproperties.get_size_in_points()
    self._house_req_size = sz
    self._fontproperties.set_size(_want_size(self, sz))


def _set_color(self, color):
    self._house_req_color = color
    if not getattr(self, "_house_keep_color", False) and neutral(color):
        color = BLACK
    _T_set_color(self, color)


# ----------------------------------------------------------------------------------------------------------- spines
_S_set_edgecolor = mspines.Spine.set_edgecolor
_S_set_linewidth = mspines.Spine.set_linewidth


def _sp_set_edgecolor(self, color):
    self._house_req_color = color
    if not getattr(self, "_house_keep_color", False) and (color is None or neutral(color, 0.97)):
        color = BLACK
    _S_set_edgecolor(self, color)


def _sp_set_linewidth(self, w):
    self._house_req_lw = w
    _S_set_linewidth(self, LW if (w is None or w > 0) else w)


# ------------------------------------------------------------------------------------------------------------ ticks
_A_set_tick_params = maxis.Axis.set_tick_params


def _tick_kw(which, kw):
    kw = dict(kw)
    for k in ("size", "length"):
        if kw.get(k) is not None and kw[k] > 0:
            kw[k] = TICK_MINOR if which == "minor" else TICK
    if kw.get("width") is not None and kw["width"] > 0:
        kw["width"] = LW
    for k in ("color", "colors", "labelcolor"):
        if k in kw and neutral(kw[k], 0.97):
            kw[k] = BLACK
    if "labelsize" in kw:
        kw["labelsize"] = FS
    return kw


def _set_tick_params(self, which="major", reset=False, **kw):
    return _A_set_tick_params(self, which=which, reset=reset, **_tick_kw(which, kw))


# -------------------------------------------------------------------------------------------------- reference lines
_AX_axhline, _AX_axvline, _AX_axline = Axes.axhline, Axes.axvline, Axes.axline
_REF_MAP = {}       # style of a converted reference line -> its new style; legend keys drawn alike follow it


def _style_key(color, linestyle, linewidth):
    probe = mlines.Line2D([], [], color=color, linestyle=linestyle, linewidth=linewidth)
    return (tuple(round(v, 3) for v in mcolors.to_rgba(probe.get_color())), round(probe.get_linewidth(), 2),
            str(probe._unscaled_dash_pattern))


def _line_key(ln):
    return (tuple(round(v, 3) for v in mcolors.to_rgba(ln.get_color())), round(ln.get_linewidth(), 2),
            str(ln._unscaled_dash_pattern))


def _ref_kw(kw):
    kw = dict(kw)
    orig = (kw.get("color", kw.get("c")), kw.get("linestyle", kw.get("ls")), kw.get("linewidth", kw.get("lw")))
    c = kw.pop("color", kw.pop("c", None))
    ls = kw.pop("linestyle", kw.pop("ls", None))
    kw.pop("dashes", None)
    kw.pop("lw", None)
    solid = ls in (None, "-", "solid")
    if c is not None and _rgba(c) is not None and min(_rgba(c)[:3]) > 0.97:      # white: a gap, not a line
        kw.update(color=c, linestyle=ls or "-", linewidth=kw.pop("linewidth", LW))
        return kw
    if solid and c is not None and light_grey(c):
        kw.update(color=HAIR, linewidth=HAIR_LW, linestyle="-")
    else:
        kw.pop("linewidth", None)
        kw.update(color=BLACK if (c is None or neutral(c, 0.97)) else c, linewidth=LW,
                  linestyle="-" if solid else DASH)
    try:
        _REF_MAP[_style_key(*orig)] = (kw["color"], kw["linewidth"], kw["linestyle"])
    except (ValueError, TypeError):
        pass
    return kw


def _axhline(self, y=0, xmin=0, xmax=1, **kw):
    return _AX_axhline(self, y, xmin, xmax, **_ref_kw(kw))


def _axvline(self, x=0, ymin=0, ymax=1, **kw):
    return _AX_axvline(self, x, ymin, ymax, **_ref_kw(kw))


def _axline(self, xy1, xy2=None, *, slope=None, **kw):
    return _AX_axline(self, xy1, xy2, slope=slope, **_ref_kw(kw))


# ---------------------------------------------------------------------------------------------------------- legends
_L_init = mlegend.Legend.__init__


def _legend_init(self, parent, handles, labels, *args, **kw):
    if kw.get("fontsize") is not None:
        kw["fontsize"] = FS
    p = kw.get("prop")
    if isinstance(p, dict):
        kw["prop"] = {**p, "size": FS}
    elif isinstance(p, FontProperties):
        q = p.copy()
        q.set_size(FS)
        kw["prop"] = q
    if kw.get("title_fontsize") is not None:
        kw["title_fontsize"] = FS
    if "labelcolor" in kw and isinstance(kw["labelcolor"], str) and neutral(kw["labelcolor"]):
        kw["labelcolor"] = BLACK
    return _L_init(self, parent, handles, labels, *args, **kw)


# ---------------------------------------------------------------------------------------------- save-time enforce
def _enforce_axis(axis):
    for which, ticks, length in (("major", axis.get_major_ticks(), TICK), ("minor", axis.get_minor_ticks(), TICK_MINOR)):
        for t in ticks:
            for ln in (t.tick1line, t.tick2line):
                if ln.get_markersize() > 0:
                    ln.set_markersize(length)
                ln.set_markeredgewidth(LW)
                if neutral(ln.get_color(), 0.97):
                    ln.set_color(BLACK)
                    ln.set_markeredgecolor(BLACK)


REF_MIN_PT = 14.0          # a shorter two-point line is a hand-drawn key sample: it must keep its data line's look


def _length_pt(ln, fig):
    try:
        xy = ln.get_transform().transform(ln.get_xydata())
    except Exception:          # noqa: BLE001  (unusual transforms: treat as a reference line)
        return REF_MIN_PT
    d = xy[-1] - xy[0]
    return float((d[0] ** 2 + d[1] ** 2) ** 0.5) / fig.dpi * 72


def enforce(fig):
    """Re-apply the rules to every artist of fig (idempotent).  Called by Figure.savefig once activate() ran."""
    for t in fig.findobj(mtext.Text):
        if not getattr(t, "_house_keep_size", False):
            req = getattr(t, "_house_req_size", t.get_fontsize())
            t._fontproperties.set_size(_want_size(t, req))
        if not getattr(t, "_house_keep_color", False) and neutral(t.get_color()):
            _T_set_color(t, BLACK)
    for ax in fig.get_axes():
        for sp in ax.spines.values():
            if sp.get_linewidth() > 0:
                _S_set_linewidth(sp, LW)
            if not getattr(sp, "_house_keep_color", False) and neutral(sp.get_edgecolor(), 0.97):
                _S_set_edgecolor(sp, BLACK)
        _enforce_axis(ax.xaxis)
        _enforce_axis(ax.yaxis)
    # pass 1: a long two-point dashed grey line is a reference line drawn with plot()
    two_pt = [ln for ax in fig.get_axes() for ln in ax.get_lines()
              if len(ln.get_xdata()) == 2 and not getattr(ln, "_house_keep_color", False)]
    for ln in two_pt:
        if ln.get_linestyle() in ("--", ":", "-.") and neutral(ln.get_color(), 0.97) \
                and _length_pt(ln, fig) >= REF_MIN_PT:
            _REF_MAP[_line_key(ln)] = (BLACK, LW, DASH)
            ln.set_color(BLACK)
            ln.set_linewidth(LW)
            ln.set_linestyle(DASH)
            ln._house_done = True
    # pass 2: a hand-drawn key sample drawn exactly like a converted reference line (any colour) follows it;
    # every other short sample keeps its data line's look
    for ln in two_pt:
        if getattr(ln, "_house_done", False):
            continue
        new = _REF_MAP.get(_line_key(ln))
        if new is not None:
            ln.set_color(new[0])
            ln.set_linewidth(new[1])
            ln.set_linestyle(new[2])
            ln._house_done = True
    legends = [a.get_legend() for a in fig.get_axes() if a.get_legend() is not None] + list(fig.legends)
    for leg in legends:                     # a legend key drawn like a converted reference line follows it
        for ln in leg.get_lines():
            new = _REF_MAP.get(_line_key(ln))
            if new is not None:
                ln.set_color(new[0])
                ln.set_linewidth(new[1])
                ln.set_linestyle(new[2])
    arrows = list(fig.findobj(FancyArrowPatch))
    arrows += [a.arrow_patch for a in fig.findobj(mtext.Annotation) if getattr(a, "arrow_patch", None) is not None]
    for ar in arrows:
        if not getattr(ar, "_house_keep_color", False) and neutral(ar.get_edgecolor(), 0.97):
            ar.set_color(BLACK)
            ar.set_linewidth(LW)


_F_savefig = mfigure.Figure.savefig


def _savefig(self, *a, **kw):
    enforce(self)
    return _F_savefig(self, *a, **kw)


# ------------------------------------------------------------------------------------------------------- interface
RC = {
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.titlecolor": BLACK, "axes.edgecolor": BLACK,
    "xtick.color": BLACK, "ytick.color": BLACK, "xtick.labelcolor": BLACK, "ytick.labelcolor": BLACK,
    "axes.linewidth": LW, "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": LW,
    "ytick.minor.width": LW, "xtick.major.size": TICK, "ytick.major.size": TICK, "xtick.minor.size": TICK_MINOR,
    "ytick.minor.size": TICK_MINOR, "xtick.major.pad": PAD, "ytick.major.pad": PAD, "xtick.minor.pad": PAD,
    "ytick.minor.pad": PAD, "axes.labelpad": LABELPAD, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
}


def activate():
    """Install the house style (no-op when PANEL_STYLE_OFF=1 or when already installed)."""
    global _done
    if not ACTIVE:
        return False
    mpl.rcParams.update(RC)
    if not _done:
        mtext.Text.set_fontsize = _set_fontsize
        mtext.Text.set_size = _set_fontsize
        mtext.Text.set_fontproperties = _set_fontproperties
        mtext.Text.set_font_properties = _set_fontproperties
        mtext.Text.set_color = _set_color
        mtext.Text.set_c = _set_color
        mspines.Spine.set_edgecolor = _sp_set_edgecolor
        mspines.Spine.set_linewidth = _sp_set_linewidth
        mspines.Spine.set_lw = _sp_set_linewidth
        maxis.Axis.set_tick_params = _set_tick_params
        Axes.axhline, Axes.axvline, Axes.axline = _axhline, _axvline, _axline
        mlegend.Legend.__init__ = _legend_init
        mfigure.Figure.savefig = _savefig
        _done = True
    return True


def keep(artist, color=True, size=False):
    """Exempt one artist; restore the colour (and size) the module asked for."""
    if color:
        artist._house_keep_color = True
        req = getattr(artist, "_house_req_color", None)
        if req is not None:
            if isinstance(artist, mtext.Text):
                _T_set_color(artist, req)
            elif isinstance(artist, mspines.Spine):
                _S_set_edgecolor(artist, req)
            elif isinstance(artist, mlines.Line2D):
                artist.set_color(req)
    if size and isinstance(artist, mtext.Text):
        artist._house_keep_size = True
        req = getattr(artist, "_house_req_size", None)
        if req is not None:
            _T_set_fontsize(artist, req)
    return artist


