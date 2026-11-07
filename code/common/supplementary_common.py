"""Helpers shared by the Supplementary Figure scripts: the code/ folder, and checks that labels stay clear of the data."""
import os

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the code/ folder


def px_per_mm(fig):
    return fig.dpi / 25.4


def text_clear_of_points(fig, ax, texts, xy, pad_mm=0.3, what=""):
    """No data point (xy in data coordinates) may fall inside a text box padded by pad_mm."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.asarray(xy, float))
    pad = pad_mm * px_per_mm(fig)
    for t in texts:
        bb = t.get_window_extent(r).padded(pad)
        n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
        assert n == 0, f"{what}: text {t.get_text()!r} covers {n} data points"


def clearance_mm(fig, ax, items, step_mm=0.05, cap_mm=3.0):
    """Smallest distance (mm, to step_mm, capped) between any item bbox and any line vertex or filled band of ax."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = px_per_mm(fig)
    bands = [ax.transData.transform_path(p) for pc in ax.collections for p in pc.get_paths()]
    lines = [ax.transData.transform(ln.get_xydata()) for ln in ax.get_lines() if len(ln.get_xydata()) > 2]
    boxes = [it.get_window_extent(r) for it in items]
    c = 0.0
    while c < cap_mm:
        hit = False
        for b in boxes:
            bb = b.padded(c * k)
            if any(p.intersects_bbox(bb, filled=True) for p in bands):
                hit = True
                break
            for xy in lines:
                if ((xy[:, 0] > bb.x0) & (xy[:, 0] < bb.x1) & (xy[:, 1] > bb.y0) & (xy[:, 1] < bb.y1)).any():
                    hit = True
                    break
            if hit:
                break
        if hit:
            return round(c - step_mm, 2)
        c += step_mm
    return cap_mm
