"""RECAST top-200 ranking per subtype lineage, for the depth-200 endpoints (marker rank, contamination).  Same
selection as select_recast; only TOPN differs.
-> panels/topN200_<id>_recast_M0.csv   [cluster, rank, gene]   (env SC, GPU)"""
import os, sys
import numpy as np, pandas as pd, anndata as ad
SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path:
        sys.path.insert(0, p)
import lineage_registry as C
from select_recast import _load_raw
from scoring_benchmark.datasets import _go_gene_order, SCIMILARITY_MODEL
ATLASES = f"{SB}/atlases"; OUT = f"{SB}/subtype_benchmark/panels"; TOPN = 200

LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]

import recast
from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder
enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda"))
go = _go_gene_order()

for dsid in LIN20:
    e = C.LINEAGES[dsid]; stem = e["ds"]
    lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    obs = np.asarray(att.obs_names).astype(str); del att
    keepstates = sorted([s for s in e["states"] if (lab_all == s).sum() >= e["min_cells"]])
    keep = np.isin(lab_all, keepstates)
    a_raw = ad.AnnData(_load_raw(stem, obs[keep])); a_raw.var_names = go
    a_raw.obs["state"] = lab_all[keep].astype(str)
    # centroid="mean_lognorm" is set explicitly: C_g = mean_i log1p(1e4 * x_ig / sum_g x_ig), per-cell lognorm,
    # then the mean over cells.
    ar = recast.cluster_attribution(enc, a_raw, "state", reference="rest", gate="dC", centroid="mean_lognorm")
    rows = [dict(cluster=str(cl), rank=i + 1, gene=str(g))
            for cl in keepstates for i, g in enumerate(list(ar.top(cl, k=TOPN)))]
    pd.DataFrame(rows, columns=["cluster", "rank", "gene"]).to_csv(
        f"{OUT}/topN200_{dsid}_recast_M0.csv", index=False)
    print(f"[{dsid}] {len(rows)} rows / {len(keepstates)} states", flush=True)
print("[recast deep export done]")
