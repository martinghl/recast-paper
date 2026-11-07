"""Marker recall per lineage against CellMarker (cellmarker_panels.py), RECAST and the comparators: per subtype
recall@K = |top-K ∩ markers| / |markers ∩ vocabulary|, mean over subtypes with markers.  Uses only gene lists (the
comparator lists come as CSV exports, fs_genesets/<id>.csv), so it runs in seconds.  UNIQUE_MARKERS=1 keeps only the
markers unique to one subtype within its lineage.  Writes results/fs_cm_recovery[_uniq]_<id><tag>.csv.  Env SC."""
import os, sys, json
import numpy as np, pandas as pd

FSB = "/path/to/research/analysis/subtype_benchmark"
OUT = f"{FSB}/results"
# GENESETS_DIR: comparator gene lists (fs_genesets_v2 for the integer-count runs); RECAST_RESULT_TAG: result tag.
GENESETS = os.environ.get("GENESETS_DIR") or f"{FSB}/fs_genesets"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
PANELS = f"{FSB}/panels"
sys.path.insert(0, FSB)
import lineage_registry as C
from cluster_key import pick   # tolerate make.names-renamed cluster labels (see module docstring)

GT = json.load(open(f"{OUT}/cellmarker_gt_fine.json"))
GO = set(l.strip() for l in open(f"{OUT}/go_genes.txt") if l.strip())   # go gene universe (28231), for GT ∩ genes
KS = (10, 20, 30)
# UNIQUE_MARKERS=1 -> keep only markers unique to one subtype within its lineage (drop any CellMarker gene shared by
# >= 2 sibling subtypes, e.g. CD3E / CD8A across CD8 subtypes): shared cell-type markers would credit any method that
# ranks by expression magnitude, so the unique markers test subtype-specific selection.
UNIQUE = os.environ.get("UNIQUE_MARKERS", "0") == "1"
TAG = "uniq_" if UNIQUE else ""


def m0_geneset(fsid):
    df = pd.read_csv(f"{PANELS}/topN_{fsid}_recast_M0.csv").sort_values(["cluster", "rank"])
    return {str(c): list(g["gene"].astype(str)) for c, g in df.groupby("cluster")}


def de_genesets(fsid):
    df = pd.read_csv(f"{GENESETS}/{fsid}.csv").sort_values(["method", "cluster", "rank"])
    return {str(m): {str(c): list(g["gene"].astype(str)) for c, g in gm.groupby("cluster")}
            for m, gm in df.groupby("method")}


def recovery(gs, markers, clusters, K):
    recs = []
    for c in clusters:
        mk = markers.get(c, set())
        if not mk:
            continue
        top = list(pick(gs, c))[:K]
        recs.append(len(set(top) & mk) / len(mk))
    return float(np.mean(recs)) if recs else np.nan


def run(fsid):
    m0 = m0_geneset(fsid)
    de = de_genesets(fsid)
    clusters = sorted(m0.keys())                       # canonical >=min_cells states RECAST selected over
    markers = {c: (set(GT.get(c, [])) & GO) for c in clusters}
    if UNIQUE:                                          # keep only genes unique to one state within this lineage
        from collections import Counter
        shared = {g for g, n in Counter(g for c in clusters for g in markers[c]).items() if n >= 2}
        markers = {c: (markers[c] - shared) for c in clusters}
    n_arm2 = sum(1 for c in clusters if markers[c])
    rows = []

    def add(name, gs):
        for K in KS:
            rows.append(dict(method=name, k=K, marker_recall=recovery(gs, markers, clusters, K)))

    add("recast_M0", m0)
    for m, gs in de.items():
        add(m, gs)
    rc = pd.DataFrame(rows); rc.insert(0, "lineage", fsid)
    os.makedirs(OUT, exist_ok=True)
    rc.to_csv(f"{OUT}/fs_cm_recovery_{TAG}{fsid}{VTAG}.csv", index=False)

    ngt = {c: len(markers[c]) for c in clusters}
    r20 = rc[rc.k == 20].dropna(subset=["marker_recall"]).sort_values("marker_recall", ascending=False).reset_index(drop=True)
    if len(r20) == 0 or "recast_M0" not in set(r20.method):
        print(f"[{fsid}] {len(clusters)} states ({n_arm2} w/ GT) | recovery@20: NO scorable arm2 states "
              f"(all GT empty) | GT sizes {ngt}", flush=True)
        return -1, 0
    rank = int((r20.method == "recast_M0").values.argmax()) + 1
    m0v = float(r20.loc[r20.method == "recast_M0", "marker_recall"].iloc[0])
    top = r20.iloc[0]
    print(f"[{fsid}] {len(clusters)} states ({n_arm2} w/ GT) | recovery@20: recast_M0 rank {rank}/{len(r20)} "
          f"val={m0v:.3f} | top={top.method}({top.marker_recall:.3f}) | GT sizes {ngt}", flush=True)
    return rank, len(r20)


if __name__ == "__main__":
    ids = sys.argv[1:] or C.NEW_IDS
    for fsid in ids:
        run(fsid)
    print("[recovery done]", flush=True)
