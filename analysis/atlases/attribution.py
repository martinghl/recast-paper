"""SCimilarity integrated-gradients attributions for one atlas, as attribution_pbmc3k.py (SIGnature's
data and attribution code unchanged: harmonize to the 28,231-gene panel, log-normalize, integrated gradients with
multiply_by_inputs=True, |.|, rows scaled to sum 1000).

Input and output paths follow the dataset id.  Checks: shape (n_cells, 28231); non-negative; finite; every row sums
to 1000 unless the cell expressed no gene of the panel (reported).

Usage: python attribution.py <dataset_id>"""
import sys
import time
import numpy as np
import anndata

sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data
import attributions as sig_attr

if len(sys.argv) != 2:
    raise SystemExit("usage: attribution.py <dataset_id>")
DATASET_ID = sys.argv[1]

ATLAS_DIR = "/path/to/research/analysis/atlases"
IN = f"{ATLAS_DIR}/{DATASET_ID}_fullgene.h5ad"
OUT = f"{ATLAS_DIR}/{DATASET_ID}_attribution_scimilarity.h5ad"
MODEL_DIR = sig_attr.DEFAULT_MODEL_DIR
EXPECTED_PANEL_WIDTH = 28231

t0 = time.time()
adata = anndata.read_h5ad(IN)
adata.layers["counts"] = adata.X.copy()          # harmonize reads raw counts
n_before = adata.n_obs
obs_before = list(adata.obs_names)
n_genes_in = adata.n_vars
print(f"[{DATASET_ID}] input: {adata.shape} | obs_names[:3]={obs_before[:3]}", flush=True)

gene_order = sig_data.read_gene_order(MODEL_DIR)
assert len(gene_order) == EXPECTED_PANEL_WIDTH, \
    f"unexpected SCimilarity panel width: {len(gene_order)} != {EXPECTED_PANEL_WIDTH}"
print(f"[{DATASET_ID}] SCimilarity gene_order: {len(gene_order)} genes (from {MODEL_DIR})", flush=True)

ad_h = sig_data.harmonize(adata, gene_order)     # -> 28,231-gene order, X = lognorm (CP10k+log1p)

# --- Check 2: harmonize must not drop or reorder cells. Attribution
# rows are matched to the benchmark cells by POSITION downstream, so this is
# asserted, not eyeballed. ---
assert ad_h.n_obs == n_before, f"[GATE FAIL: cell count] n_obs changed: {n_before} -> {ad_h.n_obs}"
assert list(ad_h.obs_names) == obs_before, "[GATE FAIL: cell order] obs_names order changed by harmonize()"
print(f"[{DATASET_ID}][gate: cell order] n_obs preserved: {ad_h.n_obs} == {n_before}; "
      f"obs_names order identical to input: True", flush=True)

# --- Check 3: genes matched to the encoder's gene order ---
matched = set(adata.var_names) & set(gene_order)
n_matched = len(matched)
n_present_mask = int(ad_h.var["present"].sum())
assert n_matched == n_present_mask, (n_matched, n_present_mask)
n_hybrid = sum(1 for g in adata.var_names if g not in matched and "-ENSG" in g)
panel_coverage_frac = n_matched / n_genes_in
print(
    f"[{DATASET_ID}][panel match] {n_matched} / {n_genes_in} input genes matched the SCimilarity "
    f"{len(gene_order)}-gene panel ({panel_coverage_frac:.1%}); "
    f"{n_genes_in - n_matched} unmatched become zero columns (expected -- includes "
    f"{n_hybrid} SYMBOL-ENSEMBLID hybrid ids that cannot match by construction).",
    flush=True,
)

t1 = time.time()
print(f"[{DATASET_ID}] harmonize done in {t1 - t0:.1f}s; running IG attribution "
      f"(batch_size=500, use_gpu=True)...", flush=True)
attr = sig_attr.compute_attributions(ad_h, batch_size=500, use_gpu=True)   # IG, |.|, row-sum 1000
t2 = time.time()
print(f"[{DATASET_ID}] attribution done in {t2 - t1:.1f}s", flush=True)

# --- Check 1: shape (n_cells, 28231); non-negative; finite; row
# sums ~= 1000. Asserted, not eyeballed. ---
assert attr.shape == (n_before, EXPECTED_PANEL_WIDTH), \
    f"[GATE FAIL: shape] {attr.shape} != {(n_before, EXPECTED_PANEL_WIDTH)}"
mn = attr.min()
mx = attr.max()
assert mn >= 0, f"[GATE FAIL: non-negative] min value = {mn}"
assert np.isfinite(attr.data).all(), "[GATE FAIL: finite] non-finite values present"

rs = np.asarray(attr.sum(axis=1)).ravel()
degenerate = rs == 0
n_degenerate = int(degenerate.sum())
ok_rowsum = np.isclose(rs[~degenerate], 1000, rtol=1e-3, atol=1e-6) if (~degenerate).any() else np.array([], dtype=bool)
assert ok_rowsum.all(), \
    f"[GATE FAIL: row sum] {(~ok_rowsum).sum()} / {len(rs) - n_degenerate} non-degenerate rows not ~=1000"

print(
    f"[{DATASET_ID}][gate: shape/nonneg/finite/rowsum] shape={attr.shape} OK; "
    f"min={mn:.6g} max={mx:.6g}; finite=True; "
    f"row-sum(non-degenerate) min/max="
    f"{(rs[~degenerate].min() if (~degenerate).any() else float('nan')):.6f}/"
    f"{(rs[~degenerate].max() if (~degenerate).any() else float('nan')):.6f}; "
    f"degenerate all-zero rows={n_degenerate}/{len(rs)}",
    flush=True,
)
if n_degenerate:
    print(f"[{DATASET_ID}] NOTE: {n_degenerate} cell(s) have all-zero attribution "
          f"(no gene the cell expressed overlaps the SCimilarity panel) -- "
          f"flagged, not counted as a gate failure.", flush=True)

# --- Cell-order-preservation re-check against the exported input file itself
# (not just the in-memory adata object), matching the task's explicit gate 2
# wording ("assert obs_names match the exported input element-by-element"). ---
input_only = anndata.read_h5ad(IN)
assert list(ad_h.obs_names) == list(input_only.obs_names), \
    "[GATE FAIL: cell order vs exported input] obs_names differ from the exported input file"
print(f"[{DATASET_ID}][gate: cell order vs exported input] obs_names match the exported "
      f"input h5ad element-by-element: True", flush=True)

ad_h.layers["attribution_ig"] = attr
ad_h.write_h5ad(OUT)
t3 = time.time()
print(f"[{DATASET_ID}] wrote {OUT} in {t3 - t2:.1f}s", flush=True)

print(f"[{DATASET_ID}] total wall time {t3 - t0:.1f}s", flush=True)
print(f"[{DATASET_ID}] GATES: ALL PASS "
      f"(shape={attr.shape}, panel_coverage={panel_coverage_frac:.1%}, "
      f"degenerate_rows={n_degenerate})", flush=True)
