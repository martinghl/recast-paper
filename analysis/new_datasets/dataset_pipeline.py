#!/usr/bin/env python
"""Prepare one more single-cell dataset for the analyses: counts on SCimilarity's gene order, the
reference-free SCimilarity attributions, per-cell labels, and a check that the attributions rank curated markers
above housekeeping genes.

Given a raw-counts .h5ad, an obs label column and the SCimilarity model folder, it writes:

  dataset_files/<id>/<id>_fullgene.h5ad
      raw counts in the dataset's own (HGNC-symbol) gene space + obs['label'].

  dataset_files/<id>/<id>_attribution_scimilarity.h5ad
      cells x 28231 SCimilarity `gene_order` (HGNC), the layout of atlases/*_attribution_scimilarity.h5ad:
        .X                        = per-cell tp10k log-normalized expression (gene_order)
        layers['counts']          = raw counts on gene_order (zero for genes outside the panel)
        layers['attribution_ig']  = reference-free SCimilarity attribution (row sum ~1000, non-negative)
        var['present']            = gene measured in this dataset and in the panel
        obs['label']              = per-cell label
        uns['orig_genes']         = the dataset's original gene list (from align_dataset)

  dataset_files/<id>/bench_labels_<id>.txt
      per-cell labels, one per line, in obs order.

  dataset_files/<id>/trust_gate.json
      AUROC of curated markers against housekeeping genes in the mean attribution, and the pass / fail verdict.

The attributions come from SIGnature's own code (external/Signature/src: data.harmonize +
attributions.compute_attributions, i.e. SCimilarityWrapper.calculate_attributions(method="ig",
multiply_by_inputs=True, target_sum=1e3)): integrated gradients of the cell's own embedding from a zero baseline,
|.|, each cell's attributions scaled to sum 1000 -- as atlases/attribution.py does for the atlases.

The marker check: one-sided Mann-Whitney U of the mean attribution of the curated markers (--markers) against
housekeeping genes (RPL*, RPS*, MT-*); AUROC = U / (n_marker * n_house); pass if AUROC > --auc-threshold (0.7).
For reference, SCimilarity reaches 0.767 on the Lawlor islets.

CLI
---
  python new_datasets/dataset_pipeline.py \
      --h5ad <raw.h5ad> --id <name> --label-col <obs col> \
      [--layer <counts layer>] [--gene-id symbol|ensembl] \
      [--symbol-col <var col with HGNC symbols>] [--gene-map <ens->symbol tsv/csv>] \
      [--markers <json {state:[genes]}>] [--device cuda|cpu] [--gpu 0] \
      [--model-dir <scimilarity model dir>] [--out-dir <dir>] \
      [--auc-threshold 0.7] [--batch-size 500]

Prints the verdict, % genes aligned, and the numbers of cells and states."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# --- Pre-scan argv to pin the GPU BEFORE any torch-bearing import happens. The SIGnature
# recipe loads torch lazily (only when compute_attributions -> get_wrapper is called), but we
# set CUDA_VISIBLE_DEVICES here so there is no window in which torch could grab the wrong GPU.
def _prescan_gpu(argv):
    gpu, device = "0", "cuda"
    for i, a in enumerate(argv):
        if a == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
        elif a.startswith("--gpu="):
            gpu = a.split("=", 1)[1]
        elif a == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
        elif a.startswith("--device="):
            device = a.split("=", 1)[1]
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    return gpu, device


_prescan_gpu(sys.argv[1:])

# The frozen SIGnature recipe (read-only import; do not modify these packages).
sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data          # noqa: E402  -> read_gene_order, harmonize
import attributions as sig_attr  # noqa: E402  -> compute_attributions, mean_attribution_per_gene

DEFAULT_MODEL_DIR = sig_attr.DEFAULT_MODEL_DIR   # .../scimilarity_model/model_v1.1
EXPECTED_PANEL_WIDTH = 28231
HERE = os.path.dirname(os.path.abspath(__file__))

# Housekeeping genes: ribosomal RPL*/RPS* and mitochondrial MT-*.
DEFAULT_HOUSEKEEPING_PREFIXES = ("RPL", "RPS", "MT-")

# General immune marker panel,
# used ONLY as a labeled SUBSTITUTE when the caller passes no --markers.
GENERAL_IMMUNE_SUBSTITUTE = [
    "CD3D", "CD3E", "CD8A", "IL7R", "MS4A1", "CD79A", "CD14", "LYZ",
    "FCGR3A", "NKG7", "GNLY", "PPBP", "FCER1A", "CST3",
]

# Common var-column names that carry HGNC symbols alongside Ensembl var_names.
_SYMBOL_COL_CANDIDATES = [
    "feature_name", "gene_symbols", "gene_symbol", "gene_name", "gene_names",
    "symbol", "symbols", "hgnc_symbol", "SYMBOL", "Symbol",
]


# ----------------------------------------------------------------------------------------
# Gene-ID handling
# ----------------------------------------------------------------------------------------
def _strip_ens_version(x: str) -> str:
    """ENSG00000123456.7 -> ENSG00000123456 (Ensembl IDs carry a version suffix)."""
    x = str(x)
    return x.split(".", 1)[0] if x.startswith("ENSG") else x


def resolve_symbols(adata, gene_id, symbol_col, gene_map_path):
    """Return (symbols: np.ndarray[str], info: dict) -- the HGNC symbols to use as the
    dataset's gene space. Handles --gene-id symbol|ensembl.

    symbol  : var_names are already HGNC symbols -> use as-is.
    ensembl : map Ensembl IDs -> symbols, in priority order:
                1. --gene-map <tsv/csv> two columns (ensembl, symbol)
                2. --symbol-col <var col> carrying per-gene symbols (e.g. cellxgene 'feature_name')
                3. auto-detected symbol-like var column
                4. none available -> WARN and keep Ensembl IDs (they will not align;
                   % aligned is reported so that the failure is explicit).
    """
    var_names = np.asarray(adata.var_names).astype(str)
    info = {"gene_id": gene_id, "n_input_genes": int(len(var_names))}

    if gene_id == "symbol":
        info["mapping_source"] = "none (var_names already HGNC symbols)"
        return var_names, info

    # ----- gene_id == "ensembl" -----
    mapping = None
    source = None
    if gene_map_path:
        sep = "," if gene_map_path.endswith(".csv") else "\t"
        mp = pd.read_csv(gene_map_path, sep=sep, header=None, dtype=str, comment="#")
        mapping = {_strip_ens_version(e): str(s) for e, s in zip(mp.iloc[:, 0], mp.iloc[:, 1])}
        source = f"--gene-map {gene_map_path} ({len(mapping)} pairs)"
    else:
        col = symbol_col
        if col is None:
            for c in _SYMBOL_COL_CANDIDATES:
                if c in adata.var.columns:
                    col = c
                    break
        if col is not None and col in adata.var.columns:
            syms = adata.var[col].astype(str).to_numpy()
            info["symbol_col"] = col
            source = f"var['{col}']"
            n_go = _panel_overlap_count(syms, gene_map_path=None,
                                        go=sig_data.read_gene_order(DEFAULT_MODEL_DIR))
            info["mapping_source"] = source
            info["n_mapped"] = int(np.sum(syms != "nan"))
            info["n_symbols_in_panel"] = int(n_go)
            return syms, info

    if mapping is not None:
        syms = np.array([mapping.get(_strip_ens_version(g), g) for g in var_names], dtype=object).astype(str)
        n_mapped = int(np.sum([_strip_ens_version(g) in mapping for g in var_names]))
        info["mapping_source"] = source
        info["n_mapped"] = n_mapped
        return syms, info

    # Nothing to map with -> keep Ensembl, warn loudly.
    info["mapping_source"] = "NONE -- no --gene-map, no --symbol-col, no symbol-like var column found"
    info["n_mapped"] = 0
    print("[new_datasets][WARN] --gene-id ensembl but no symbol mapping available; keeping Ensembl "
          "IDs as-is. They will NOT match the HGNC gene panel -- pass --symbol-col or "
          "--gene-map. Reporting % aligned so this is explicit.", flush=True)
    return var_names, info


def _panel_overlap_count(symbols, gene_map_path, go):
    goset = set(go)
    return int(sum(1 for s in set(map(str, symbols)) if s in goset))


# ----------------------------------------------------------------------------------------
# Artifact 1: fullgene (raw counts, own gene space, obs['label'])
# ----------------------------------------------------------------------------------------
def build_fullgene(adata, counts, symbols, labels):
    """Assemble the <id>_fullgene.h5ad AnnData: raw counts in the dataset's own (symbol)
    gene space + obs['label']. Cell order and barcodes preserved from the input."""
    X = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(np.asarray(counts))
    if X.min() < 0:
        raise SystemExit("[new_datasets] input counts contain negative values -- not raw counts.")
    fg = ad.AnnData(X=X)
    fg.obs_names = np.asarray(adata.obs_names).astype(str)
    fg.var_names = np.asarray(symbols).astype(str)
    fg.obs["label"] = pd.Categorical([str(x) for x in labels])
    return fg


# ----------------------------------------------------------------------------------------
# Artifact 2: reference-free SCimilarity attribution h5ad
# ----------------------------------------------------------------------------------------
def compute_attribution_h5ad(fullgene, model_dir, batch_size, use_gpu):
    """Run the frozen SIGnature recipe on the fullgene AnnData and return the harmonized,
    attributed AnnData -- mirrors atlases/attribution.py exactly,
    with the same asserted gates (shape / non-negative / finite / row-sum ~1000 / cell order).
    """
    n_before = fullgene.n_obs
    obs_before = list(fullgene.obs_names)
    n_genes_in = fullgene.n_vars

    adata = fullgene.copy()
    adata.layers["counts"] = adata.X.copy()          # harmonize reads raw counts

    gene_order = sig_data.read_gene_order(model_dir)
    assert len(gene_order) == EXPECTED_PANEL_WIDTH, \
        f"unexpected SCimilarity panel width: {len(gene_order)} != {EXPECTED_PANEL_WIDTH}"

    ad_h = sig_data.harmonize(adata, gene_order)     # X = tp10k lognorm; layers['counts'] go-aligned raw

    # Gate: harmonize must not drop or reorder cells (attributions matched by position downstream).
    assert ad_h.n_obs == n_before, f"[GATE FAIL cell count] {n_before} -> {ad_h.n_obs}"
    assert list(ad_h.obs_names) == obs_before, "[GATE FAIL cell order] obs_names reordered by harmonize()"

    # Gate: panel-match accounting.
    matched = set(fullgene.var_names) & set(gene_order)
    n_matched = len(matched)
    assert n_matched == int(ad_h.var["present"].sum()), (n_matched, int(ad_h.var["present"].sum()))
    panel_coverage_frac = n_matched / max(1, n_genes_in)

    t1 = time.time()
    attr = sig_attr.compute_attributions(ad_h, model_dir=model_dir,
                                         batch_size=batch_size, use_gpu=use_gpu)
    print(f"[new_datasets] IG attribution done in {time.time() - t1:.1f}s "
          f"(batch_size={batch_size}, use_gpu={use_gpu})", flush=True)

    # Gate: shape / non-negative / finite / row-sum ~1000 (degenerate all-zero rows flagged).
    assert attr.shape == (n_before, EXPECTED_PANEL_WIDTH), \
        f"[GATE FAIL shape] {attr.shape} != {(n_before, EXPECTED_PANEL_WIDTH)}"
    assert attr.min() >= 0, f"[GATE FAIL non-negative] min={attr.min()}"
    assert np.isfinite(attr.data).all(), "[GATE FAIL finite] non-finite values present"
    rs = np.asarray(attr.sum(axis=1)).ravel()
    degenerate = rs == 0
    n_degenerate = int(degenerate.sum())
    if (~degenerate).any():
        ok = np.isclose(rs[~degenerate], 1000, rtol=1e-3, atol=1e-6)
        assert ok.all(), f"[GATE FAIL row-sum] {int((~ok).sum())}/{int((~degenerate).sum())} rows not ~1000"

    ad_h.layers["attribution_ig"] = attr
    if "label" in fullgene.obs.columns:
        ad_h.obs["label"] = fullgene.obs["label"].values

    gate = {
        "shape": list(attr.shape),
        "attr_min": float(attr.min()),
        "attr_max": float(attr.max()),
        "row_sum_min": float(rs[~degenerate].min()) if (~degenerate).any() else float("nan"),
        "row_sum_max": float(rs[~degenerate].max()) if (~degenerate).any() else float("nan"),
        "n_degenerate_rows": n_degenerate,
        "n_genes_matched": n_matched,
        "panel_coverage_frac": float(panel_coverage_frac),
    }
    print(f"[new_datasets][gate] shape={attr.shape} min={attr.min():.4g} max={attr.max():.4g} "
          f"row-sum[min/max]={gate['row_sum_min']:.4f}/{gate['row_sum_max']:.4f} "
          f"degenerate={n_degenerate}/{n_before} panel_coverage={panel_coverage_frac:.1%}", flush=True)
    if n_degenerate:
        print(f"[new_datasets] NOTE: {n_degenerate} cell(s) have all-zero attribution (no expressed "
              f"gene overlaps the panel) -- flagged, not a gate failure.", flush=True)
    return ad_h, gate


# ----------------------------------------------------------------------------------------
# Artifact 4: trust gate (curated markers vs housekeeping)
# ----------------------------------------------------------------------------------------
def trust_gate(ad_h, marker_genes, housekeeping_prefixes, auc_threshold,
               scimilarity_reference=0.767):
    """Marker-vs-housekeeping AUROC of the mean attribution (see the module docstring).

    marker_genes : flat list (union of the caller's {state:[genes]}).
    Returns a dict matching results/micro_trust_check.json's shape + GO/NO-GO fields.
    """
    from scipy.stats import mannwhitneyu

    mean_attr = sig_attr.mean_attribution_per_gene(ad_h.layers["attribution_ig"])
    genes = np.asarray(ad_h.var_names).astype(str)
    gi = {g: i for i, g in enumerate(genes)}

    marker_set = list(dict.fromkeys(str(g) for g in marker_genes))       # dedupe, keep order
    present_markers = [m for m in marker_set if m in gi]
    mk_idx = np.array([gi[m] for m in present_markers], dtype=int)

    house_mask = np.array([g.startswith(tuple(housekeeping_prefixes)) for g in genes])
    hk_idx = np.where(house_mask)[0]

    n_genes = len(genes)
    order = np.argsort(-mean_attr)
    rank = np.empty(n_genes, dtype=int); rank[order] = np.arange(n_genes)
    pct = 1.0 - rank / (n_genes - 1)                                     # 1.0 = highest attribution

    if mk_idx.size == 0 or hk_idx.size == 0:
        verdict = "NO-GO"
        result = {
            "auc_marker_vs_housekeeping": None,
            "verdict": verdict, "go": False,
            "reason": "empty marker or housekeeping set in the 28231-gene panel",
            "n_markers_requested": len(marker_set),
            "n_markers_present": int(mk_idx.size),
            "n_housekeeping": int(hk_idx.size),
        }
        return result

    U, p = mannwhitneyu(mean_attr[mk_idx], mean_attr[hk_idx], alternative="greater")
    auc = float(U / (mk_idx.size * hk_idx.size))

    top20 = [genes[i] for i in order[:20]]
    top100 = set(genes[order[:100]])
    m_pct, hk_pct = pct[mk_idx], pct[hk_idx]

    go = auc > auc_threshold
    meets_ref = auc >= (scimilarity_reference - 0.01)   # within rounding of the 0.767 reference
    verdict = ("GO-STRONG" if (go and meets_ref) else "GO" if go else "NO-GO")

    return {
        "auc_marker_vs_housekeeping": auc,
        "p_value": float(p),
        "go": bool(go),
        "verdict": verdict,
        "auc_threshold": float(auc_threshold),
        "scimilarity_reference_auc": float(scimilarity_reference),
        "meets_scimilarity_reference": bool(meets_ref),
        "marker_pct_median": float(np.median(m_pct)),
        "marker_pct_min": float(m_pct.min()),
        "housekeeping_pct_median": float(np.median(hk_pct)),
        "n_housekeeping": int(hk_idx.size),
        "housekeeping_prefixes": list(housekeeping_prefixes),
        "n_markers_requested": len(marker_set),
        "n_markers_present": int(mk_idx.size),
        "markers_present": present_markers,
        "markers_in_top100": sorted(top100 & set(present_markers)),
        "marker_median_attr": float(np.median(mean_attr[mk_idx])),
        "housekeeping_median_attr": float(np.median(mean_attr[hk_idx])),
        "top20_by_attribution": top20,
        "n_cells": int(ad_h.n_obs),
        "n_genes_aligned": int(ad_h.n_vars),
    }


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------
def load_marker_genes(markers_path):
    """--markers: JSON {state: [genes]} -> flat union list. None -> documented substitute."""
    if markers_path is None:
        print("[new_datasets][WARN] no --markers given; using the documented general-immune SUBSTITUTE "
              "panel for the trust gate (label it SUBSTITUTE in any report). Pass tissue-curated "
              "markers via --markers for a real gate.", flush=True)
        return list(GENERAL_IMMUNE_SUBSTITUTE), "general-immune SUBSTITUTE (no --markers supplied)"
    with open(markers_path) as fh:
        d = json.load(fh)
    if isinstance(d, dict):
        flat = []
        for v in d.values():
            flat.extend(list(v) if isinstance(v, (list, tuple, set)) else [v])
        source = f"--markers {os.path.basename(markers_path)} ({len(d)} states -> {len(set(flat))} genes)"
    elif isinstance(d, (list, tuple)):
        flat = list(d)
        source = f"--markers {os.path.basename(markers_path)} (flat list -> {len(set(flat))} genes)"
    else:
        raise SystemExit(f"[new_datasets] --markers must be a JSON object or list, got {type(d)}")
    return flat, source


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5ad", required=True, help="raw-counts .h5ad (X or --layer) + a label column")
    ap.add_argument("--id", required=True, help="dataset id (names the output folder + files)")
    ap.add_argument("--label-col", required=True, help="obs column holding per-cell labels")
    ap.add_argument("--layer", default=None, help="counts layer name (default: adata.X)")
    ap.add_argument("--gene-id", default="symbol", choices=["symbol", "ensembl"],
                    help="var_names identifier type (default symbol)")
    ap.add_argument("--symbol-col", default=None, help="var column with HGNC symbols (for --gene-id ensembl)")
    ap.add_argument("--gene-map", default=None, help="ensembl->symbol map file (tsv/csv, 2 cols, no header)")
    ap.add_argument("--markers", default=None, help="JSON {state:[genes]} of curated markers for the trust gate")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="SCimilarity model dir (gene_order.tsv/encoder)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "dataset_files"),
                    help="output root; artifacts go to <out-dir>/<id>/")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cuda (default) or cpu")
    ap.add_argument("--gpu", default="0", help="GPU index to pin via CUDA_VISIBLE_DEVICES (default 0)")
    ap.add_argument("--batch-size", type=int, default=500, help="cells per IG batch (default 500)")
    ap.add_argument("--auc-threshold", type=float, default=0.7,
                    help="trust-gate GO threshold on marker>housekeeping AUROC (default 0.7)")
    ap.add_argument("--housekeeping-prefixes", default=",".join(DEFAULT_HOUSEKEEPING_PREFIXES),
                    help="comma-separated housekeeping gene prefixes (default RPL,RPS,MT-)")
    a = ap.parse_args(argv)

    use_gpu = (a.device == "cuda")
    hk_prefixes = tuple(x for x in a.housekeeping_prefixes.split(",") if x)
    out_id_dir = os.path.join(a.out_dir, a.id)
    os.makedirs(out_id_dir, exist_ok=True)
    fullgene_path = os.path.join(out_id_dir, f"{a.id}_fullgene.h5ad")
    attr_path = os.path.join(out_id_dir, f"{a.id}_attribution_scimilarity.h5ad")
    labels_path = os.path.join(out_id_dir, f"bench_labels_{a.id}.txt")
    trust_path = os.path.join(out_id_dir, "trust_gate.json")

    print("=" * 80)
    print(f"[new_datasets] dataset id = {a.id}")
    print(f"[new_datasets] input      = {a.h5ad}  (layer={a.layer or 'X'})")
    print(f"[new_datasets] model dir  = {a.model_dir}")
    print(f"[new_datasets] device     = {a.device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")
    print(f"[new_datasets] out dir    = {out_id_dir}")
    print("=" * 80, flush=True)

    if not os.path.exists(a.h5ad):
        raise SystemExit(f"[new_datasets] h5ad not found: {a.h5ad}")
    adata = ad.read_h5ad(a.h5ad)
    if a.label_col not in adata.obs.columns:
        raise SystemExit(f"[new_datasets] --label-col '{a.label_col}' not in obs. Have: {list(adata.obs.columns)}")
    counts = adata.layers[a.layer] if a.layer else adata.X
    if counts is None:
        raise SystemExit("[new_datasets] no counts found (adata.X is None and no --layer given).")
    labels = adata.obs[a.label_col].astype(str).to_numpy()
    n_cells = adata.n_obs
    states, state_counts = np.unique(labels, return_counts=True)
    print(f"[new_datasets] {n_cells} cells x {adata.n_vars} genes; {len(states)} states in '{a.label_col}'", flush=True)

    # --- gene-ID resolution ---
    go = sig_data.read_gene_order(a.model_dir)
    symbols, gene_info = resolve_symbols(adata, a.gene_id, a.symbol_col, a.gene_map)
    n_in_panel = _panel_overlap_count(symbols, gene_map_path=None, go=go)
    pct_aligned = 100.0 * n_in_panel / len(go)
    pct_of_yours = 100.0 * n_in_panel / max(1, len(set(map(str, symbols))))
    gene_info.update({"n_symbols_in_panel": int(n_in_panel),
                      "pct_of_panel_covered": round(pct_aligned, 2),
                      "pct_of_your_genes_in_panel": round(pct_of_yours, 2)})
    print(f"[new_datasets] gene-ID: {gene_info['mapping_source']}", flush=True)
    print(f"[new_datasets] gene alignment: {n_in_panel} unique symbols in the {len(go)}-gene panel "
          f"({pct_aligned:.1f}% of panel; {pct_of_yours:.1f}% of your genes)", flush=True)
    if n_in_panel < 5000:
        raise SystemExit(
            f"[new_datasets] only {n_in_panel} of your genes are in the SCimilarity HGNC panel "
            f"(SIGnature.align_dataset requires >=5000). Are your var_names HGNC symbols? "
            f"For Ensembl input pass --gene-id ensembl with --symbol-col or --gene-map.")
    if n_in_panel < 0.2 * len(go):
        print(f"[new_datasets][WARN] <20% of the panel covered ({pct_aligned:.1f}%) -- attributions will "
              f"be dominated by zero columns. Check your gene IDs.", flush=True)

    # --- artifact 1: fullgene ---
    fg = build_fullgene(adata, counts, symbols, labels)
    fg.write_h5ad(fullgene_path)
    print(f"[new_datasets] wrote {fullgene_path}  {fg.shape}", flush=True)

    # --- artifact 3: bench labels (obs order) ---
    with open(labels_path, "w") as fh:
        fh.write("\n".join(str(x) for x in labels) + "\n")
    print(f"[new_datasets] wrote {labels_path}  ({len(labels)} labels)", flush=True)

    # --- artifact 2: reference-free attribution ---
    ad_h, gate = compute_attribution_h5ad(fg, a.model_dir, a.batch_size, use_gpu)
    ad_h.write_h5ad(attr_path)
    print(f"[new_datasets] wrote {attr_path}  {ad_h.shape}", flush=True)

    # --- artifact 4: trust gate ---
    marker_genes, marker_source = load_marker_genes(a.markers)
    trust = trust_gate(ad_h, marker_genes, hk_prefixes, a.auc_threshold)
    trust.update({
        "dataset_id": a.id,
        "marker_source": marker_source,
        "gene_id_handling": gene_info,
        "attribution_gate": gate,
        "input_h5ad": os.path.abspath(a.h5ad),
        "n_states": int(len(states)),
        "states": {str(s): int(c) for s, c in zip(states, state_counts)},
    })
    with open(trust_path, "w") as fh:
        json.dump(trust, fh, indent=2)
    print(f"[new_datasets] wrote {trust_path}", flush=True)

    # --- verdict ---
    auc = trust.get("auc_marker_vs_housekeeping")
    print("\n" + "=" * 80)
    print(f"[new_datasets] TRUST GATE: {trust['verdict']}")
    if auc is not None:
        print(f"[new_datasets]   marker>housekeeping AUROC = {auc:.4f}  "
              f"(threshold {a.auc_threshold}; SCimilarity reference 0.767; "
              f"p={trust['p_value']:.3g})")
        print(f"[new_datasets]   markers present {trust['n_markers_present']}/{trust['n_markers_requested']} "
              f"({marker_source}); housekeeping n={trust['n_housekeeping']}")
    else:
        print(f"[new_datasets]   {trust.get('reason')}")
    print(f"[new_datasets]   {n_in_panel}/{len(go)} genes aligned ({pct_aligned:.1f}% of panel); "
          f"{n_cells} cells; {len(states)} states")
    decision = "GO" if trust.get("go") else "NO-GO"
    print(f"\n[new_datasets] ===> {decision} <===")
    print("=" * 80, flush=True)
    return 0 if trust.get("go") else 2


if __name__ == "__main__":
    sys.exit(main())
