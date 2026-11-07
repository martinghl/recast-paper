"""Score the Wilcoxon arm of the label-noise runs: ps_de_<lin>_top50.csv -> ps_de_all_metrics{tag}.csv.

select_wilcoxon.py writes ranked gene lists per (condition, state); the figure reads one row per (lineage, state,
mode, eps, seed), with the definitions label_noise_runs.readouts uses for RECAST:

    jaccard20 = |top20(perturbed) n top20(clean)| / |top20(perturbed) u top20(clean)|
    recall20  = |top20 n UNIQ(state)| / |UNIQ(state)|,  UNIQ = CellMarker subtype markers n this lineage's genes,
                minus any marker shared by two or more sibling states

    python score_comparators_label_noise.py _v2     # ps_de_all_metrics_v2.csv

Env SC."""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FSB = "/path/to/research/analysis/subtype_benchmark"
LIN = ["tf_cd8_dom", "tf_b_yosh", "tf_mono_steph", "fs_lung_hlca"]
K = 20
CLEAN = "none_e000_s-1"

GT = json.load(open(f"{FSB}/results/cellmarker_gt_fine.json"))
GO = set(l.strip() for l in open(f"{FSB}/results/go_genes.txt") if l.strip())


def parse_condition(tag):
    """'boundary_e025_s0' -> ('boundary', 0.025, 0);  'none_e000_s-1' -> ('none', 0.0, -1)."""
    mode, e, s = tag.rsplit("_", 2)
    return mode, int(e[1:]) / 1000.0, int(s[1:])


def one(lin, tag):
    meta = np.load(f"{HERE}/ps_export_{lin}_meta{tag}.npz", allow_pickle=True)
    gi = set(meta["genes"].astype(str))
    states = [str(s) for s in meta["states"]]
    markers = {c: (set(GT.get(c, [])) & GO) for c in states}
    shared = {g for g, n in Counter(g for c in states for g in markers[c]).items() if n >= 2}
    uniq = {c: [g for g in (markers[c] - shared) if g in gi] for c in states}

    d = pd.read_csv(f"{HERE}/ps_de_{lin}_top50{tag}.csv").sort_values(["condition", "state", "rank"])
    top20 = {(c, s): list(g.gene.astype(str))[:K] for (c, s), g in d.groupby(["condition", "state"])}
    rows = []
    for (cond, state), top in top20.items():
        base = top20.get((CLEAN, state))
        if base is None:
            continue
        mode, eps, seed = parse_condition(cond)
        mk = uniq.get(state, [])
        rows.append(dict(lineage=lin, state=state, mode=mode, eps=eps, seed=seed,
                         jaccard20=len(set(top) & set(base)) / len(set(top) | set(base)),
                         recall20=(len(set(top) & set(mk)) / len(mk)) if mk else np.nan))
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    # States with no unique CellMarker marker keep a NaN recall row -- the RECAST table carries them too,
    # and dropping them here would silently change the row count the figures join on.
    got = pd.DataFrame([r for l in LIN for r in one(l, tag)])
    got = got.sort_values(["lineage", "state", "mode", "eps", "seed"]).reset_index(drop=True)
    p = f"{HERE}/ps_de_all_metrics{tag}.csv"
    got.to_csv(p, index=False)
    print(f"wrote {p}  ({len(got)} rows, {got.lineage.nunique()} lineages, {got.state.nunique()} states)")


if __name__ == "__main__":
    main()
