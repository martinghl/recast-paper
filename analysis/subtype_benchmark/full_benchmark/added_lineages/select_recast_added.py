"""RECAST panels for the added lineages with select_recast.py.  The top 50 come from select_recast.select() (its module
globals C / ATLASES / OUT pointed at the added lineages); the top 200 repeat the loop of export_recast_top200.py (which
runs over its own lineage list) on the cells load_lineage() returns.  reference="rest", gate="dC",
centroid="mean_lognorm".
Writes full_benchmark/data/panels/topN_<id>_recast_M0.csv and topN200_<id>_recast_M0.csv.

Env SC, GPU.  Usage: CUDA_VISIBLE_DEVICES=0 python -I select_recast_added.py [dsid ...]"""
import os
import sys
import types

import anndata as ad
import pandas as pd

SB = "/path/to/research/analysis"
FULL_BENCHMARK = f"{SB}/subtype_benchmark/full_benchmark"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark", f"{FULL_BENCHMARK}/added_lineages"):
    if p not in sys.path:
        sys.path.insert(0, p)
import added_datasets as D  # noqa: E402
import select_recast as S  # noqa: E402

S.C = types.SimpleNamespace(LINEAGES=D.LINEAGES, NEW_IDS=D.NEW_IDS)
S.ATLASES = f"{FULL_BENCHMARK}/data/atlases"
S.OUT = f"{FULL_BENCHMARK}/data/panels"
TOPN_DEEP = 200


def deep(dsid, enc):
    Craw_go, lab_keep, keepstates, go = S.load_lineage(dsid)
    import recast
    a_raw = ad.AnnData(Craw_go); a_raw.var_names = go; a_raw.obs["state"] = lab_keep
    ar = recast.cluster_attribution(enc, a_raw, "state", reference="rest", gate="dC", centroid="mean_lognorm")
    rows = [dict(cluster=str(cl), rank=i + 1, gene=str(g))
            for cl in keepstates for i, g in enumerate(list(ar.top(cl, k=TOPN_DEEP)))]
    pd.DataFrame(rows, columns=["cluster", "rank", "gene"]).to_csv(f"{S.OUT}/topN200_{dsid}_recast_M0.csv", index=False)
    print(f"[{dsid}] deep: {len(rows)} rows / {len(keepstates)} states", flush=True)


def main():
    ids = sys.argv[1:] or D.NEW_IDS
    os.makedirs(S.OUT, exist_ok=True)
    import recast
    from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(S.SCIMILARITY_MODEL, device="cuda"))
    for dsid in ids:
        S.select(dsid)
        deep(dsid, enc)
        t50 = pd.read_csv(f"{S.OUT}/topN_{dsid}_recast_M0.csv")
        t200 = pd.read_csv(f"{S.OUT}/topN200_{dsid}_recast_M0.csv")
        for cl, g in t50.groupby("cluster"):
            head = t200[t200.cluster == cl].sort_values("rank").gene.head(len(g)).tolist()
            if head != g.sort_values("rank").gene.tolist():
                n = len(set(head) & set(g.gene))
                print(f"  [{dsid}] {cl}: top-50 and the first 50 of top-200 differ ({n}/50 shared)", flush=True)


if __name__ == "__main__":
    main()
