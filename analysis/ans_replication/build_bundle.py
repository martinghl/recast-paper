"""Per-dataset bundles for the gene-set scoring comparisons with ANS.

A bundle is:
    a_log    AnnData, log-normalized expression (the scoring methods score a_log.X per cell)
    a_raw    AnnData or None, raw counts on SCimilarity's 28,231-gene order (RECAST scores from a_raw)
    truth    per-cell label (signature names), aligned to a_log / a_raw
    clusters sorted unique truth labels (= the signatures used)
    signatures {name: [genes]}
    cluster_key  obs column holding `truth`
    flags    dict describing the data (printed and saved)

Builders:
  build_neural_diff  -- the induced pluripotent stem cell-to-midbrain data of Jerber et al., as in ANS's
      neuronal_differentation_dataset.py: read h5ad -> subsample 40% (random_state=123) -> sc.pp.log1p -> a_log;
      celltype = truth = clusters; signatures = top-100 sc.tl.rank_genes_groups(method='wilcoxon') per celltype;
      a_raw = the matrix before log1p on SCimilarity's gene order.
  build_fig2         -- ANS's B-cell subtype and lung adenocarcinoma data (pp_pbmc_b_subtypes / pp_luad_kim_malignant).
  build_fig2_pbmcbmn -- the B-cell subtypes on the Hao PBMC subset.

Every builder prints: (i) whether .X holds counts, normalized or log values; (ii) the gene-ID namespace and the share
of genes found in SCimilarity's gene_order.tsv."""
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

# _go_gene_order / _align_to_go: SCimilarity gene-space alignment for RECAST a_raw (see datasets.py).
from scoring_benchmark.datasets import _go_gene_order, _align_to_go

AR = "/path/to/research/analysis/ans_replication"
NEURAL_H5 = f"{AR}/data/neural_diff/all_timepoints_subsampled.h5"
PP_DIR = f"{AR}/data/preprocessed"
ANNOT_DIR = f"{AR}/data/annotations"

# ---------------------------------------------------------------------------------------------
# ANS signature loaders (ported: logic identical to ANS data/load_data.py + data/constants.py).
# ---------------------------------------------------------------------------------------------
# constants.py::PBMC_DATASET_SIGS_CONFIGS (subset actually used here).
PBMC_SIGS_CONFIG = {
    "pbmc_b_subtypes": {
        "B intermediate": ["B intermediate kappa", "B intermediate lambda"],
        "B memory": ["B memory kappa", "B memory lambda"],
        "B naive": ["B naive kappa", "B naive lambda"],
    },
    "pbmc_b_mono_nk": {
        "B": ["B naive kappa", "B memory kappa", "B naive lambda", "B memory lambda",
              "B intermediate kappa", "B intermediate lambda", "Plasma", "Plasmablast"],
        "Mono": ["CD14 Mono", "CD16 Mono"],
        "NK": ["NK_2", "NK_4", "NK_1", "NK_3", "NK_CD56bright", "NK Proliferating"],
    },
}


def _remove_overlapping_signature_genes(sig_genes):
    """ANS _remove_overlapping_signature_genes: drop any gene shared across >=2 signatures."""
    sig_old = {k: list(v) for k, v in sig_genes.items()}
    out = {}
    for ct, genes in sig_genes.items():
        g = set(genes)
        for ct2, genes2 in sig_old.items():
            if ct2 == ct:
                continue
            g -= set(genes2)
        out[ct] = sorted(g)
    return out


def _load_pbmc_signatures(dataset, de_csv, filter_genes=True,
                          threshold_pval=0.01, threshold_log2fc=0.5,
                          remove_overlapping_genes=False):
    """ANS _load_pbmc_signatures: filter DE_by_celltype.csv (P Value<=0.01 & Avg Log FC>=0.5, the
    CODE default -- not the paper's 0.05), then union of the level-3 subtypes per config."""
    de = pd.read_csv(de_csv)
    if filter_genes:
        de = de[(de["P Value"] <= threshold_pval) &
                (de["Average Log Fold Change"] >= threshold_log2fc)].copy().reset_index(drop=True)
    cfg = PBMC_SIGS_CONFIG[dataset]
    sigs = {}
    for cell_type, subtypes in cfg.items():
        genes = de[de["Cell Type"].isin(subtypes)]["Gene"].tolist()
        sigs[cell_type] = sorted(set(genes))
    if remove_overlapping_genes:
        sigs = _remove_overlapping_signature_genes(sigs)
    return sigs


def _load_cancer_csv_signatures(csv, remove_overlapping_genes=False):
    """ANS _process_csv_signatures: one gene column per malignant state; drop NaNs, sort, dedupe."""
    df = pd.read_csv(csv)
    sigs = {k: sorted(set(v.dropna().astype(str).tolist())) for k, v in df.to_dict("series").items()}
    if remove_overlapping_genes:
        sigs = _remove_overlapping_signature_genes(sigs)
    return sigs


# ---------------------------------------------------------------------------------------------
# Provenance heuristics.
# ---------------------------------------------------------------------------------------------
def _matrix_stats(X, n=2000):
    """Cheap stats on a (sub)sample of a sparse/dense matrix: (min, max, frac_integer, has_neg)."""
    if sp.issparse(X):
        sub = X[: min(n, X.shape[0])]
        data = np.asarray(sub.data, dtype="float64")
        if data.size == 0:
            data = np.zeros(1)
        mn = float(min(0.0, data.min()))
        mx = float(data.max())
    else:
        sub = np.asarray(X[: min(n, X.shape[0])], dtype="float64")
        data = sub[sub != 0] if sub.size else sub
        if data.size == 0:
            data = np.zeros(1)
        mn = float(sub.min())
        mx = float(data.max())
    frac_int = float(np.mean(np.isclose(data, np.round(data))))
    return mn, mx, frac_int, (mn < -1e-6)


def _classify_matrix(X):
    mn, mx, frac_int, has_neg = _matrix_stats(X)
    if has_neg:
        kind = "scaled/z-scored (has negatives)"
    elif frac_int > 0.98 and mx > 30:
        kind = "raw counts (integer-valued)"
    elif mx < 20:
        kind = "normalized-log (non-integer, small range)"
    else:
        kind = "normalized-linear (non-integer, large range)"
    return dict(kind=kind, min=round(mn, 3), max=round(mx, 3),
                frac_integer=round(frac_int, 3), has_negative=has_neg)


def _gene_namespace(var_names, var_df=None):
    vn = [str(g) for g in var_names]
    n_ens = sum(g.upper().startswith("ENSG") for g in vn)
    frac_ens = n_ens / max(1, len(vn))
    symbol_col = None
    if frac_ens > 0.5 and var_df is not None:
        for c in var_df.columns:
            vals = var_df[c].astype(str)
            if (~vals.str.upper().str.startswith("ENSG")).mean() > 0.8 and vals.str.len().mean() < 15:
                symbol_col = c
                break
    return dict(frac_ensembl=round(frac_ens, 3),
                looks_like_ensembl=frac_ens > 0.5,
                symbol_col=symbol_col)


def _pct_in_go(symbols, go_set):
    syms = [str(s) for s in symbols]
    hit = sum(s in go_set for s in syms)
    return round(100.0 * hit / max(1, len(syms)), 2), hit, len(syms)


# ---------------------------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------------------------
def _resolve_symbols(adata, ns):
    """Return an HGNC-symbol array for adata.var (mapping from Ensembl via a symbol column if
    needed). Falls back to var_names unchanged when already symbols or no mapping available."""
    if ns["looks_like_ensembl"] and ns["symbol_col"] is not None:
        return adata.var[ns["symbol_col"]].astype(str).values
    return np.asarray(adata.var_names, dtype=str)


def build_neural_diff(subsample=0.4, seed=123, verbose=True):
    go = _go_gene_order()
    go_set = set(go)
    flags = {"dataset": "neural_diff", "source_h5": NEURAL_H5}

    data = sc.read_h5ad(NEURAL_H5)
    flags["n_obs_full"] = int(data.n_obs)
    flags["n_var"] = int(data.n_vars)
    flags["obs_columns"] = list(map(str, data.obs.columns))
    flags["X_kind_preLog"] = _classify_matrix(data.X)
    ns = _gene_namespace(data.var_names, data.var)
    flags["gene_namespace"] = ns

    # subsample 40% (random_state=123) -- exactly ANS.
    n = int(round(data.n_obs * subsample))
    sampled = data.obs.sample(n=n, random_state=seed).index
    data = data[sampled].copy()
    flags["n_obs_subsampled"] = int(data.n_obs)

    if "celltype" not in data.obs.columns:
        raise KeyError(f"'celltype' not in obs columns {list(data.obs.columns)}")
    truth = data.obs["celltype"].astype(str).values
    clusters = sorted(pd.unique(truth).tolist())
    flags["celltypes"] = clusters
    flags["n_celltypes"] = len(clusters)

    # --- a_raw (RECAST): raw-like counts BEFORE log1p, symbol-mapped, reindexed to go order. ---
    symbols = _resolve_symbols(data, ns)
    pct_go, hit, tot = _pct_in_go(symbols, go_set)
    flags["pct_genes_in_go"] = pct_go
    flags["genes_in_go"] = f"{hit}/{tot}"
    Xraw = data.X.tocsr() if sp.issparse(data.X) else sp.csr_matrix(data.X)
    # Integrality check, as in build_fig2.  `all_timepoints_subsampled.h5` has 0 % integer values and a per-cell
    # total of 10,085: it is library-normalized, with no .raw and no counts layer, so raw counts would need the
    # original Jerber et al. deposit.  The developmental analyses use this matrix (ALLOW_NONINTEGER_ARAW=1).
    _k = _classify_matrix(Xraw)
    flags["a_raw_kind"] = _k
    if not _k["kind"].startswith("raw counts"):
        if os.environ.get("ALLOW_NONINTEGER_ARAW") != "1":
            raise ValueError(
                f"neural_diff a_raw is {_k['kind']!r}, not integer counts "
                f"(frac_integer={_k.get('frac_integer')}, max={_k.get('max')}). "
                f"{NEURAL_H5} has no .raw and no counts layer; re-download the raw Jerber deposit, or "
                "set ALLOW_NONINTEGER_ARAW=1 to build the bundle the analyses used.")
        flags["a_raw_integrality_override"] = True
    Craw_aligned = _align_to_go(Xraw, list(map(str, symbols)), go)
    a_raw = ad.AnnData(Craw_aligned)
    a_raw.var_names = go
    a_raw.obs["celltype"] = truth
    flags["a_raw_source"] = ("pre-log1p .X (" + flags["X_kind_preLog"]["kind"] + ")")

    # --- a_log (baselines): sc.pp.log1p on the subsampled .X, exactly as ANS. ---
    a_log = data  # reuse; will log1p in place
    sc.pp.log1p(a_log)
    a_log.obs["celltype"] = truth
    flags["a_log_kind"] = "sc.pp.log1p(subsampled .X)"

    # --- signatures: top-100 wilcoxon markers per celltype (on a_log), exactly as ANS. ---
    sc.tl.rank_genes_groups(a_log, groupby="celltype", method="wilcoxon", n_genes=100)
    res = a_log.uns["rank_genes_groups"]
    names = res["names"]
    signatures = {}
    for grp in names.dtype.names:
        # already ranked by wilcoxon; take the top-100 gene names for this celltype.
        signatures[str(grp)] = [str(g) for g in list(names[grp])]
    flags["signature_lengths"] = {k: len(v) for k, v in signatures.items()}

    if verbose:
        print(json.dumps({k: v for k, v in flags.items() if k != "obs_columns"}, indent=2,
                         default=str), flush=True)
    return dict(a_log=a_log, a_raw=a_raw, truth=truth, clusters=clusters,
                signatures=signatures, cluster_key="celltype", flags=flags)


def _find_truth_col(obs, wanted):
    """Find the obs column whose value set best matches `wanted` (signature names)."""
    wanted = set(map(str, wanted))
    best, best_ov = None, -1
    for c in obs.columns:
        vals = set(map(str, pd.unique(obs[c].astype(str))))
        ov = len(vals & wanted)
        if ov > best_ov:
            best, best_ov = c, ov
    return best, best_ov


def build_fig2(name, verbose=True):
    """name in {'pbmc_b_subtypes', 'luad_kim_malignant'}."""
    go = _go_gene_order()
    go_set = set(go)
    flags = {"dataset": name}

    if name == "pbmc_b_subtypes":
        h5ad = f"{PP_DIR}/pp_pbmc_b_subtypes.h5ad"
        signatures = _load_pbmc_signatures("pbmc_b_subtypes", f"{ANNOT_DIR}/DE_by_celltype.csv")
    elif name == "luad_kim_malignant":
        h5ad = f"{PP_DIR}/pp_luad_kim_malignant.h5ad"
        signatures = _load_cancer_csv_signatures(f"{ANNOT_DIR}/kim_3.csv")
    else:
        raise ValueError(f"unknown fig2 dataset {name}")
    flags["source_h5ad"] = h5ad
    flags["signature_lengths"] = {k: len(v) for k, v in signatures.items()}

    adata = sc.read_h5ad(h5ad)
    flags["n_obs"] = int(adata.n_obs)
    flags["n_var"] = int(adata.n_vars)
    flags["obs_columns"] = list(map(str, adata.obs.columns))
    flags["X_kind"] = _classify_matrix(adata.X)
    ns = _gene_namespace(adata.var_names, adata.var)
    flags["gene_namespace"] = ns

    truth_col, ov = _find_truth_col(adata.obs, signatures.keys())
    flags["truth_col"] = truth_col
    flags["truth_col_overlap_with_signatures"] = ov
    truth_all = adata.obs[truth_col].astype(str).values

    # keep only cells whose truth label is one of the signature classes (ANS scores all cells, but its
    # metrics are over the annotated classes).
    keep = np.isin(truth_all, list(signatures.keys()))
    adata = adata[keep].copy()
    truth = adata.obs[truth_col].astype(str).values
    clusters = sorted([c for c in signatures.keys() if (truth == c).sum() > 0])
    flags["n_obs_kept"] = int(adata.n_obs)
    flags["class_counts"] = {c: int((truth == c).sum()) for c in clusters}

    a_log = adata
    a_log.obs["celltype"] = truth

    # --- a_raw for RECAST: only if raw counts are recoverable from this pp_ file. ---
    a_raw = None
    raw_src = None
    cand = None
    if adata.raw is not None:
        cand = ("raw", adata.raw.X, np.asarray(adata.raw.var_names, dtype=str))
    elif "counts" in getattr(adata, "layers", {}):
        cand = ("layers['counts']", adata.layers["counts"], np.asarray(adata.var_names, dtype=str))
    else:
        xk = flags["X_kind"]["kind"]
        if xk.startswith("raw counts"):
            cand = (".X", adata.X, np.asarray(adata.var_names, dtype=str))

    if cand is not None:
        raw_src, Xr, rv = cand
        # Integrality check: `SIGnature.utils.lognorm_counts` aliases layers["counts"] instead of copying it, so
        # a layer named "counts" may hold log-normalized values.  a_raw must be integer counts.
        _k = _classify_matrix(Xr)
        flags["a_raw_kind"] = _k
        if not _k["kind"].startswith("raw counts"):
            if os.environ.get("ALLOW_NONINTEGER_ARAW") != "1":
                raise ValueError(
                    f"a_raw candidate {raw_src!r} is {_k['kind']!r}, not integer counts "
                    f"(frac_integer={_k.get('frac_integer')}, max={_k.get('max')}). RECAST requires counts. "
                    "Point at the true-counts file, or set ALLOW_NONINTEGER_ARAW=1 to override "
                    "deliberately (the override is recorded in flags).")
            flags["a_raw_integrality_override"] = True
        # map to symbols if the raw var is ensembl and a symbol col exists (rare for pp files).
        if ns["looks_like_ensembl"] and ns["symbol_col"] is not None and raw_src != "raw":
            rv = adata.var[ns["symbol_col"]].astype(str).values
        pct_go, hit, tot = _pct_in_go(rv, go_set)
        flags["pct_genes_in_go"] = pct_go
        flags["genes_in_go"] = f"{hit}/{tot}"
        Xr = Xr.tocsr() if sp.issparse(Xr) else sp.csr_matrix(Xr)
        a_raw = ad.AnnData(_align_to_go(Xr, list(map(str, rv)), go))
        a_raw.var_names = go
        a_raw.obs["celltype"] = truth
        flags["a_raw_source"] = raw_src
    else:
        flags["a_raw_source"] = None
        flags["recast_note"] = ("no raw counts in pp_ file (.X is " + flags["X_kind"]["kind"] +
                               ", no .raw, no counts layer) -> RECAST a_raw unavailable; "
                               "baselines-only for this dataset.")

    if verbose:
        print(json.dumps({k: v for k, v in flags.items() if k != "obs_columns"}, indent=2,
                         default=str), flush=True)
    return dict(a_log=a_log, a_raw=a_raw, truth=truth, clusters=clusters,
                signatures=signatures, cluster_key="celltype", flags=flags)


def build_fig2_pbmcbmn(verbose=True):
    """The three B-cell subtypes (naive, intermediate, memory) of ANS's B-cell benchmark on the Hao PBMC subset:
    ANS's eight per-cell methods and ANS's B-subtype signatures (DE_by_celltype.csv), with these cells.
    a_log = SCimilarity tp10k log-normalized values (the scoring methods score per cell); a_raw = raw counts on
    SCimilarity's gene order (RECAST).  Loaded with scoring_benchmark.datasets.
    """
    from scoring_benchmark.datasets import load_scoring_dataset
    a_raw, a_log, labels, _clusters, _sig_mark, _truth_mark = load_scoring_dataset("pbmcbmn_l2")
    truth = np.asarray(labels).astype(str)
    signatures = _load_pbmc_signatures("pbmc_b_subtypes", f"{ANNOT_DIR}/DE_by_celltype.csv")
    clusters = sorted([c for c in signatures if (truth == c).sum() > 0])
    for A in (a_log, a_raw):
        A.obs["celltype"] = truth
        A.obs["state"] = truth
    flags = {
        "dataset": "pbmcbmn_bsub",
        "source": "local atlases pbmcbmn (ANS pp_pbmc_b_subtypes.h5ad is Drive quota-blocked)",
        "n_obs": int(a_log.n_obs),
        "classes": clusters,
        "class_counts": {c: int((truth == c).sum()) for c in clusters},
        "a_log_kind": "SCimilarity tp10k-lognorm (go order, 28231 genes)",
        "a_raw_source": "go-aligned raw counts (RECAST available)",
        "signature_source": "ANS DE_by_celltype.csv, pbmc_b_subtypes config",
        "signature_lengths": {k: len(v) for k, v in signatures.items()},
        "note": ("NOT ANS's exact pp_ cells: same three B-subtype classes + ANS DE signatures + "
                 "ANS 8-method per-cell protocol, on the local pbmcbmn B/Mono/NK CITE-seq lineage, "
                 "because ANS's pp_pbmc_b_subtypes.h5ad is Drive quota-blocked."),
    }
    if verbose:
        print(json.dumps(flags, indent=2, default=str), flush=True)
    return dict(a_log=a_log, a_raw=a_raw, truth=truth, clusters=clusters,
                signatures=signatures, cluster_key="celltype", flags=flags)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "neural_diff"
    if which == "neural_diff":
        b = build_neural_diff()
    else:
        b = build_fig2(which)
    os.makedirs(f"{AR}/bundles", exist_ok=True)
    with open(f"{AR}/bundles/{b['flags']['dataset']}_flags.json", "w") as fh:
        json.dump(b["flags"], fh, indent=2, default=str)
    with open(f"{AR}/bundles/{b['flags']['dataset']}_signatures.json", "w") as fh:
        json.dump(b["signatures"], fh, indent=2)
    print(f"\n[build_bundle] {b['flags']['dataset']}: a_log={b['a_log'].shape}, "
          f"a_raw={None if b['a_raw'] is None else b['a_raw'].shape}, "
          f"clusters={b['clusters']}", flush=True)
