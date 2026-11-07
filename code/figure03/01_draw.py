#!/usr/bin/env python
"""Fig. 3: the marker benchmark on the 97 subtypes, 22 methods.

One row per method, grouped by family; box columns for marker recovery, one-vs-rest AUROC, marker rank, selection
time and contamination, and the overall score.  Boxes, means, jittered subtype dots, open circles (95% paired bootstrap
interval of RECAST minus the method includes zero) and "#k" ranks come from common/marker_benchmark_panels.py.
A method run in several configurations is shown in its best configuration by marker recovery.
The panel is drawn 183 mm wide, composed (cropped to its ink) and scaled to 180 mm.

Run     python 01_draw.py
Output  out/figure3.pdf (and the panel and the composite at 183 mm in out/)
"""
import os
import sys

import numpy as np
import pypdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import matplotlib.text as mtext                     # noqa: E402

import marker_benchmark_panels as B                 # noqa: E402
import method_colours                               # noqa: E402
import compose as CF                                # noqa: E402

OUT = os.path.join(HERE, "out")
PANEL, FIG = "benchmark_panel", "figure3_183mm"

# ---------------------------------------------------------------- display strings
B.METRICS["recall20"]["title"] = "Subtype-marker recovery\n(20 selected genes)"
B.METRICS["marker_rank200"]["title"] = "Marker-prioritization score\n(first 200 genes)"
B.METRICS["contamination"]["title"] = "Ribosomal or mito. genes\n(first 200)"
B.MAIN_NAME["random"] = "Alphabetical control"      # "random" returns genes in alphabetical order
FAMILIES = tuple(B.HEADER_CASE.get(f, f) for f in
                 ("Dedicated panel selection", "Marker Scoring", "Single-cell DE", "Pseudobulk DE", "Control"))
HEADER_PT = 6.5                                     # family headings


def configurations(index):
    """method -> its configurations (4, 4, 2, 5, 5, 5)."""
    sm = [m for m in index if m.startswith("scran_scoreMarkers_")]
    g = {"Wilcoxon": [m for m in index if m.startswith("scanpy_wilcoxon")],
         "t-test": [m for m in index if m.startswith("scanpy_t")],
         "log FC": [m for m in index if m.startswith("logfc_")],
         "scoreMarkers Cohen": [m for m in sm if m.endswith("_logFC_cohen")],
         "scoreMarkers AUC": [m for m in sm if m.endswith("_AUC")],
         "scoreMarkers detection": [m for m in sm if m.endswith("_logFC_detected")]}
    assert [len(v) for v in g.values()] == [4, 4, 2, 5, 5, 5], {k: len(v) for k, v in g.items()}
    return g


def apply_best_configuration_rule():
    """Show each multi-configuration method in its best configuration by mean marker recovery."""
    shown = {m for ms in B.MAIN_MEMBERS.values() for m in ms}
    changed = []
    for meth, cfgs in configurations(B.MASTER.index).items():
        r = B.MASTER.loc[cfgs, "recall20"].sort_values(ascending=False)
        assert r.iloc[0] > r.iloc[1], f"{meth}: tie for the best configuration"
        old = [m for m in cfgs if m in shown]
        assert len(old) == 1, (meth, old)
        old, new = old[0], r.index[0]
        if new != old:
            for fam, ms in B.MAIN_MEMBERS.items():
                B.MAIN_MEMBERS[fam] = [new if m == old else m for m in ms]
            B.MAIN_NAME[new] = B.MAIN_NAME[old]
            changed.append(f"{meth}: {old} -> {new}")
    return changed


def save_with_checks(fig, stem, outdir):
    """Set the family headings to HEADER_PT, check they stay left of the first column, then save the panel."""
    heads = [t for t in fig.findobj(mtext.Text) if t.get_text() in FAMILIES]
    assert len(heads) == len(FAMILIES), len(heads)
    for t in heads:
        t.set_fontsize(HEADER_PT)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_mm = 25.4 / fig.dpi
    data_axes = [a for a in fig.axes if a.get_title()]
    x_first = min(a.get_window_extent(rend).x0 for a in data_axes)
    for t in heads:
        gap = (x_first - t.get_window_extent(rend).x1) * px_mm
        assert gap > 0.8, f"header {t.get_text()} is {gap:.2f} mm from the first column"
    return method_colours.save_fig(fig, PANEL, OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    assert len(B.MASTER) == 56
    per = B.DOTS[B.DOTS.block == "recall20"].groupby("method").unit.nunique()
    assert per["recast_M0"] == 97 and per.min() >= 94, per.describe()
    for blk in ("recall20", "auroc", "marker_rank200", "contamination", "ms_per_cell"):
        mean = B.DOTS[B.DOTS.block == blk].groupby("method").value.mean()
        assert np.allclose(mean.loc[B.MASTER.index], B.MASTER[blk]), blk
    ranks = {k: B.recast_rank(k)[0] for k in ("recall20", "auroc", "marker_rank200", "ms_per_cell", "contamination")}
    assert ranks == {"recall20": 1, "auroc": 5, "marker_rank200": 1, "ms_per_cell": 2, "contamination": 11}, ranks
    print("configurations changed to the best by recovery:", apply_best_configuration_rule() or "none")

    B.save_fig = save_with_checks
    mg = B.main_groups()
    assert sum(len(v) for _, v in mg) == 22
    B.draw(mg, B.main_name, PANEL, slot_mm=4.60, columns=B.MAIN_BOX_COLUMNS,
           region_headers={"A": B.RH_A, "B": B.RH_B, "C": B.RH_C})

    CF.OUT, CF.HERE = OUT, OUT                       # the composer crops the panel to the ink of OUT/<panel>.png
    CF.compose(FIG, [[PANEL]], {PANEL: ""}, panels_dir=OUT)

    w = pypdf.PdfWriter()                            # 183 mm -> 180 mm
    w.add_page(pypdf.PdfReader(f"{OUT}/{FIG}.pdf").pages[0])
    w.pages[0].scale_by(180 / 25.4 * 72 / float(w.pages[0].mediabox.width))
    w.add_metadata({"/Producer": "pypdf"})
    with open(f"{OUT}/figure3.pdf", "wb") as fh:
        w.write(fh)


if __name__ == "__main__":
    main()
