"""SCimilarity integrated-gradients attributions for pbmc3k, with SIGnature's own data and attribution
code (external/Signature/src/{data,attributions}.py).

`sig_data.harmonize(adata_raw, gene_order)` aligns raw counts to SCimilarity's 28,231-gene panel and log-normalizes;
`sig_attr.compute_attributions()` runs integrated gradients with multiply_by_inputs=True, takes |.| and scales each
row to sum 1000.  `gene_order` comes from `sig_data.read_gene_order(MODEL_DIR)` (the gene_order.tsv the SCimilarity
wrapper reads), so `ad_h.var_names` lines up with the gene order inside `compute_attributions`."""
import sys
import time
import numpy as np
import anndata

sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data
import attributions as sig_attr

IN = "/path/to/research/analysis/atlases/pbmc3k_fullgene.h5ad"
OUT = "/path/to/research/analysis/atlases/pbmc3k_attribution_scimilarity.h5ad"
MODEL_DIR = sig_attr.DEFAULT_MODEL_DIR

t0 = time.time()
adata = anndata.read_h5ad(IN)
adata.layers["counts"] = adata.X.copy()          # harmonize reads raw counts
n_before = adata.n_obs
obs_before = list(adata.obs_names)
n_genes_in = adata.n_vars
print(f"input: {adata.shape} | obs_names[:3]={obs_before[:3]}", flush=True)

gene_order = sig_data.read_gene_order(MODEL_DIR)
print(f"SCimilarity gene_order: {len(gene_order)} genes (from {MODEL_DIR})", flush=True)

ad_h = sig_data.harmonize(adata, gene_order)     # -> 28,231-gene order, X = lognorm (CP10k+log1p)

# --- Check 1: harmonize must not drop
# or reorder cells. Attribution rows are matched to the benchmark SCE by
# position downstream, so this is asserted, not eyeballed. ---
assert ad_h.n_obs == n_before, f"n_obs changed: {n_before} -> {ad_h.n_obs}"
assert list(ad_h.obs_names) == obs_before, "obs_names order changed by harmonize()"
print(f"[gate1] n_obs preserved: {ad_h.n_obs} == {n_before}; obs_names order identical to input: True", flush=True)

# --- Check 3: genes matched to the encoder's gene order ---
matched = set(adata.var_names) & set(gene_order)
n_matched = len(matched)
n_present_mask = int(ad_h.var["present"].sum())
assert n_matched == n_present_mask, (n_matched, n_present_mask)
n_hybrid = sum(1 for g in adata.var_names if g not in matched and "-ENSG" in g)
print(
    f"[panel match] {n_matched} / {n_genes_in} input genes matched the SCimilarity "
    f"{len(gene_order)}-gene panel ({n_matched/n_genes_in:.1%}); "
    f"{n_genes_in - n_matched} unmatched become zero columns (expected -- includes "
    f"{n_hybrid} SYMBOL-ENSEMBLID hybrid ids that cannot match by construction).",
    flush=True,
)

t1 = time.time()
print(f"harmonize done in {t1 - t0:.1f}s; running IG attribution (batch_size=500, use_gpu=True)...", flush=True)
attr = sig_attr.compute_attributions(ad_h, batch_size=500, use_gpu=True)   # IG, |.|, row-sum 1000
t2 = time.time()
print(f"attribution done in {t2 - t1:.1f}s", flush=True)

ad_h.layers["attribution_ig"] = attr
ad_h.write_h5ad(OUT)
t3 = time.time()
print(f"wrote {OUT} in {t3 - t2:.1f}s", flush=True)

rs = np.asarray(attr.sum(axis=1)).ravel()
print("attr shape", attr.shape, "| row-sum min/max", rs.min(), rs.max(),
      "| min value", attr.min(), "| finite", np.isfinite(attr.data).all(), flush=True)
print(f"total wall time {t3 - t0:.1f}s", flush=True)
