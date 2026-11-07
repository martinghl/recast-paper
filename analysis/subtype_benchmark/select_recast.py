# subtype_benchmark/select_recast.py
"""RECAST selection per subtype lineage -> panels/topN_<id>_recast_M0.csv.  Env SC, GPU.  Cells and labels are those of
the comparator exports (same attribution h5ad, same bench_labels subset, same >= 50-cell subtypes)."""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path: sys.path.insert(0, p)
import lineage_registry as C
from scoring_benchmark.datasets import _align_to_go, _go_gene_order, SCIMILARITY_MODEL
ATLASES = f"{SB}/atlases"; OUT = f"{SB}/subtype_benchmark/panels"; TOPN = 50
# fullgene h5ad stems: reuse the same stem; raw counts come from <stem>_fullgene.h5ad, matched by obs_names
def _load_raw(stem, cells_obs):
    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pmap = {b: i for i, b in enumerate(np.asarray(fg.obs_names).astype(str))}
    order = np.array([pmap[b] for b in cells_obs]); Craw = fg.X.tocsr()[order]
    rawvar = np.asarray(fg.var_names).astype(str); del fg
    return _align_to_go(Craw, rawvar, _go_gene_order())

# module-level: the 4 zhao-reuse lineages use zhao's OWN raw vocabulary, labels from att.obs['label']
EXT_ZHAO = {
    "ext_zhao_CD8":  ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"],
    "ext_zhao_CD4":  ["CCR7+ CD4", "CXCR3+ CD4", "Treg"],
    "ext_zhao_B":    ["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"],
    "ext_zhao_Mono": ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"],
}

def load_lineage(dsid):
    """(Craw_go, labels, keepstates, go) for one lineage: the cells, labels and gene order the RECAST panels are
    selected from.  select() and the analyses on the same cells (aggregation_and_encoder.py,
    target_sample_size.py) all read them from here.
    """
    e = C.LINEAGES.get(dsid)
    if e is not None:                          # new lineage: labels from bench_labels file
        stem = e["ds"]; states = e["states"]; min_cells = e["min_cells"]
        lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
    elif dsid in EXT_ZHAO:                      # zhao-reuse: raw labels from zhao attribution obs['label']
        stem = "zhao"; states = EXT_ZHAO[dsid]; min_cells = 50; lab_all = None   # filled after att load
    else:
        raise SystemExit(f"unknown id {dsid}")
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    obs = np.asarray(att.obs_names).astype(str); go = _go_gene_order()
    assert list(np.asarray(att.var_names).astype(str)) == list(go)
    if lab_all is None:                         # ext_zhao path pulls labels from att.obs['label']
        lab_all = att.obs["label"].astype(str).to_numpy()
    assert len(lab_all) == len(obs), f"{dsid}: labels {len(lab_all)} != cells {len(obs)}"
    keepstates = sorted([s for s in states if (lab_all == s).sum() >= min_cells])
    keep = np.isin(lab_all, keepstates)
    Craw_go = _load_raw(stem, obs[keep])
    del att
    return Craw_go, lab_all[keep].astype(str), keepstates, go


def select(dsid, device="cuda"):
    Craw_go, lab_keep, keepstates, go = load_lineage(dsid)
    import recast
    from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device=device))
    a_raw = ad.AnnData(Craw_go); a_raw.var_names = go; a_raw.obs["state"] = lab_keep
    # centroid="mean_lognorm" is set explicitly: C_g = mean_i log1p(1e4 * x_ig / sum_g x_ig), per-cell lognorm,
    # then the mean over cells.
    ar = recast.cluster_attribution(enc, a_raw, "state", reference="rest", gate="dC",
                                   centroid="mean_lognorm")
    rows = [dict(cluster=str(cl), rank=i + 1, gene=str(g))
            for cl in keepstates for i, g in enumerate(list(ar.top(cl, k=TOPN)))]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows, columns=["cluster", "rank", "gene"]).to_csv(f"{OUT}/topN_{dsid}_recast_M0.csv", index=False)
    print(f"[{dsid}] {len(rows)} rows over {len(keepstates)} states", flush=True)

if __name__ == "__main__":
    ids = sys.argv[1:] or (C.NEW_IDS + list(EXT_ZHAO))
    for i in ids: select(i)
