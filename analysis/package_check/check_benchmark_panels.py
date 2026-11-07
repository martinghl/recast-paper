"""Fig. 3, Supplementary Fig. 3 and Fig. 5: the RECAST gene panels of the 33 benchmark lineages
(97 subtype units), recomputed with recast.cluster_attribution and compared with the panels the
benchmark scored (top 200, or top 50 where only that was kept).

    python check_benchmark_panels.py [lineage ...]

Writes panels/topN200_<lineage>_recast.csv and benchmark_panels.csv (one row per state)."""
import os
import sys

import pandas as pd

import common as C
import recast

PANELS = os.path.join(C.FULL_BENCHMARK, "scoring", "in", "panels")


def lineages():
    old = C.config("lineage_registry").LINEAGES
    new = C.config("added_datasets", os.path.join(C.FULL_BENCHMARK, "added_lineages")).LINEAGES
    out = {}
    for f in sorted(os.listdir(PANELS)):
        lin = f.replace("topN200_", "").replace("topN_", "").replace("_recast_M0.csv", "")
        if lin in out:
            continue
        if lin in new:
            out[lin] = dict(new[lin], p2=os.path.join(C.FULL_BENCHMARK, "data", "p2"))
        else:
            out[lin] = dict(old[lin], p2=C.ATLASES)
    return out


def original(lin):
    deep = os.path.join(PANELS, f"topN200_{lin}_recast_M0.csv")
    return C.panel_lists(deep if os.path.exists(deep) else os.path.join(PANELS, f"topN_{lin}_recast_M0.csv"))


def run(lin, spec, enc):
    labels = C.read_labels(os.path.join(spec["p2"], spec["label_file"]))
    a, states = C.lineage_adata(spec["p2"], spec["ds"], labels, spec["states"], spec["min_cells"])
    res = recast.cluster_attribution(enc, a, "state", reference="rest", gate="dC",
                                     centroid="mean_lognorm", qc="silent")
    os.makedirs(os.path.join(C.OUT, "panels"), exist_ok=True)
    pd.DataFrame([dict(cluster=s, rank=i + 1, gene=g) for s in states for i, g in enumerate(res.top(s, 200))]
                 ).to_csv(os.path.join(C.OUT, "panels", f"topN200_{lin}_recast.csv"), index=False)
    stored = original(lin)
    assert sorted(stored) == sorted(states), (lin, sorted(stored), states)
    rows = []
    for s in states:
        r = C.compare_lists(res.genes[s], stored[s])
        rows.append(dict(lineage=lin, state=s, n_cells=int((a.obs["state"] == s).sum()), **r))
        print(f"  {lin} | {s}: identical={r['identical']} depth={r['depth']} first_diff={r['first_diff_rank']}",
              flush=True)
    return rows


if __name__ == "__main__":
    todo = lineages()
    names = sys.argv[1:] or list(todo)
    enc = C.scimilarity()
    rows = []
    for lin in names:
        rows += run(lin, todo[lin], enc)
        C.assert_package_only()
    tag = "" if not sys.argv[1:] else "_" + "_".join(names)
    C.write(pd.DataFrame(rows), f"benchmark_panels{tag}.csv")
