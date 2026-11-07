"""Load a fine-subtype or broad dataset for the gene-set scoring comparisons, with the lineage table,
curated marker sets and state-name mapping of recast_subtypes.py (copied below, so that loading data does not load
SCimilarity: importing recast_subtypes.py starts the encoder on the GPU through sibling_targets.py).

Raw counts are re-indexed onto SCimilarity's gene_order.tsv (28,231 genes), with zeros for genes the dataset lacks
(for the Hao PBMC subset 11,802 of its 12,127 genes are in the panel, no duplicated symbols, so a plain scatter).
The encoder takes exactly 28,231 inputs, and recast_subtypes.py's pseudobulk step does the same re-indexing
(`population_ig.cluster_countsums` ... `.reindex(go).fillna(0.0)`), so `recast.pseudobulk_centroid(a_raw.X, mask)`
equals recast_subtypes.py's centroid for the same cluster.  Per-cell embeddings need log-normalized input; that is
done by the encoder wrapper in run_scoring_benchmark.py.  Re-indexing these counts and log-normalizing per cell
(log1p(1e4 * proportion)) reproduces the stored attribution file's X (a_log.X here) to float32 rounding (largest
difference 4.8e-7 on 30 cells of the Hao PBMC subset)."""
import numpy as np
import anndata as ad
import scipy.sparse as sp

ATLASES = "/path/to/research/analysis/atlases"
SCIMILARITY_MODEL = "/path/to/scimilarity/model_v1.1"
# the model_v1.1 folder itself (it holds gene_order.tsv and encoder.ckpt), as in attribution_methods/contrastive_ig.py

# ============================================================================================
# Copied from subtype_benchmark/recast_subtypes.py (CFG and MARKERS; see the module docstring for why this is a
# copy rather than an import). Keep the two in step.
# ============================================================================================
CFG = {
    "pbmcbmn_l2": ("pbmcbmn", "bench_labels_pbmcbmn_l2.txt",
                   ["B naive", "B intermediate", "B memory"]),
    "yoshida": ("yoshida", "bench_labels_yoshida.txt",
                ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells", "Memory B cells"]),
    "helperT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Regulatory T cells"]),
    "cytoT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
              ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
}
MARKERS = {
    "naive":        {"TCL1A", "IGHD", "FCER2", "IL4R", "YBX3", "TSPAN13", "PLPP5", "BACH2", "SELL", "VPREB3", "CD72", "IGHM", "APBB2", "PCDH9"},
    "memory":       {"CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH", "CRTAM", "CD80", "ITGB1", "TEX9", "GRIP2", "LINC01781", "BANK1"},
    "switched":     {"IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHGP", "CD27", "TNFRSF13B", "AIM2", "S100A10"},
    "nonswitched":  {"IGHM", "IGHD", "CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH"},
    "intermediate": {"TNFRSF13B", "COCH", "AIM2", "CRTAM", "MARCKS", "RALGPS2", "SSPN", "SCIMP", "CD72"},
    "plasma":       {"MZB1", "XBP1", "PRDM1", "DERL3", "JCHAIN", "TNFRSF17", "CD38", "SLAMF7", "ITM2C", "POU2AF1", "SEC11C", "FKBP11"},
    "naiveT":       {"CCR7", "SELL", "TCF7", "LEF1", "IL7R", "TXK", "NOSIP", "MAL", "TRABD2A", "ADTRP", "PLAC8", "LDHB", "CD27", "ACTN1"},
    "eff_helper":   {"GZMK", "GZMA", "CCL5", "KLRB1", "ANXA1", "S100A4", "IL32", "AHNAK", "CCR6", "GPR183", "CRTAM", "TNFRSF4", "CCR4"},
    "treg":         {"FOXP3", "IL2RA", "IKZF2", "CTLA4", "TNFRSF4", "TNFRSF18", "TIGIT", "RTKN2", "TBC1D4", "LAYN", "IL32", "IL10RA"},
    "eff_cyto":     {"GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "FGFBP2", "FCGR3A", "KLRD1", "CX3CR1", "SPON2", "KLRG1", "CMC1", "GZMA", "KLRF1"},
    "trm":          {"ITGAE", "ITGA1", "ZNF683", "CXCR6", "XCL1", "XCL2", "CD69", "GZMK", "DUSP2", "CD8A", "CD160", "KLRC1"},
    "gdt":          {"TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "KLRC1", "KLRD1", "TRGV9", "TRDV3", "KLRG1", "NKG7"},
}
TMAP = {"Regulatory T cells": "treg", "Tcm/Naive helper T cells": "naiveT", "Tem/Effector helper T cells": "eff_helper",
        "Tcm/Naive cytotoxic T cells": "naiveT", "Tem/Temra cytotoxic T cells": "eff_cyto",
        "Tem/Trm cytotoxic T cells": "trm", "gdT": "gdt"}


def state_key(lbl):
    if lbl in TMAP:
        return TMAP[lbl]
    l = lbl.lower()
    if "plasma" in l:
        return "plasma"
    if "switched" in l and "non" not in l and "unswitched" not in l:
        return "switched"
    if "non-switched" in l or "unswitched" in l:
        return "nonswitched"
    if "intermediate" in l:
        return "intermediate"
    if "naive" in l:
        return "naive"
    if "memory" in l:
        return "memory"
    return "memory"
# ============================================================================================
# End of the copy from recast_subtypes.py.
# ============================================================================================


def _go_gene_order():
    """SCimilarity's own gene order (28231 genes for model_v1.1), read directly from the model dir's
    gene_order.tsv. A static text file -- no model weights loaded, no GPU touched."""
    with open(f"{SCIMILARITY_MODEL}/gene_order.tsv") as fh:
        return [l.strip() for l in fh if l.strip()]


def _align_to_go(Craw, rawvar, go):
    """Reindex a (cells x rawvar) raw-count matrix onto SCimilarity's `go` gene order, zero-padding
    genes present in `go` but absent from `rawvar`. Sparse-safe; returns csr. See module docstring for
    why this must happen (SCimilarityEncoder needs exactly len(go) go-ordered features) and why it is
    numerically the right thing to do (matches recast_subtypes's own population_ig.cluster_countsums reindex).

    Assumes no duplicate gene symbols in `rawvar` (verified true for pbmcbmn's fullgene panel: 12127
    unique / 12127 total) -- a straight column scatter, not a groupby-sum. If some future dataset's
    rawvar *does* have duplicates, scipy's coo_matrix construction sums duplicate (row, col) entries at
    `.tocsr()` time, so a duplicated gene symbol mapping two source columns onto the same go column
    would still sum correctly; only a same-symbol collision at the SAME source column index is
    impossible by construction (one row per source index).
    """
    go_idx = {g: i for i, g in enumerate(go)}
    src = [j for j, g in enumerate(rawvar) if g in go_idx]
    dst = [go_idx[rawvar[j]] for j in src]
    M = sp.coo_matrix((np.ones(len(src), dtype="float32"), (src, dst)),
                       shape=(len(rawvar), len(go))).tocsr()
    return (Craw.astype("float32") @ M).tocsr()


def load_scoring_dataset(key):
    """-> (a_raw, a_log, labels, clusters, signatures, truth_map).

    a_log.X: dense float32 SCimilarity-ready lognorm expression (go order, 28231 genes) read straight
      from `{stem}_attribution_scimilarity.h5ad`.
    a_raw.X: sparse float32 RAW counts, reindexed onto the SAME go order (see `_align_to_go`) so
      a_raw.var_names == a_log.var_names == go, and a_raw is directly usable by
      recast.SCimilarityEncoder / recast.attribute()'s pseudobulk_centroid.
    labels: per-cell state string array (same order as a_raw/a_log obs).
    clusters: sorted list of states kept (>=60 cells each, per CFG's states list for `key`).
    signatures: {state_key(c): sorted(MARKERS[state_key(c)])} for c in clusters -- UNFILTERED marker sets
      (baselines.score_cells / recast.score_gene_set_recast each filter to their own gene universe
      internally).
    truth_map: {c: state_key(c)} for c in clusters.
    """
    stem, finef, STATES = CFG[key]
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    genes = np.asarray(att.var_names).astype(str)
    bc = np.asarray(att.obs_names).astype(str)
    fine = np.array([l.strip() for l in open(f"{ATLASES}/{finef}") if l.strip()])
    assert len(fine) == att.n_obs, f"[{key}] label/obs mismatch {len(fine)} vs {att.n_obs}"
    del att
    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pmap = {b: i for i, b in enumerate(np.asarray(fg.obs_names).astype(str))}
    Craw = fg.X.tocsr()[np.array([pmap[b] for b in bc])]
    rawvar = np.asarray(fg.var_names).astype(str)
    del fg

    STATES = [s for s in STATES if (fine == s).sum() >= 60]
    keep = np.isin(fine, STATES)
    Xln, labels, Craw = Xln[keep], fine[keep], Craw[keep]
    clusters = sorted(STATES)

    go = _go_gene_order()
    Craw_aligned = _align_to_go(Craw, rawvar, go)

    a_log = ad.AnnData(Xln)
    a_log.var_names = genes
    a_log.obs["state"] = labels
    a_raw = ad.AnnData(Craw_aligned)
    a_raw.var_names = go
    a_raw.obs["state"] = labels

    signatures = {state_key(c): sorted(MARKERS[state_key(c)]) for c in clusters}
    truth_map = {c: state_key(c) for c in clusters}
    return a_raw, a_log, labels, clusters, signatures, truth_map
