"""Cell-cycle comparisons defined by cell identity: target = a proliferating population annotated by the
dataset's authors, reference = its non-proliferating counterpart in the same lineage and dataset.  Both groups are
put on SCimilarity's 28,231 genes, so RECAST and the comparator methods use the same cells and genes.

Sources:
  * atlases/ h5ads       -- on SCimilarity's gene order; X log-normalized; counts from *_fullgene.h5ad
  * CELLxGENE h5ads      -- Ensembl var_names (symbols in var['feature_name']); raw counts in .raw.X
  * neural_diff bundle   -- its own 32,738-gene space, symbols as var_names

Per comparison, under cc_contrasts/<name>/:
  counts.npz   raw counts, CSR, cells x 28231 (zero for genes outside the dataset)
  labels.txt   "target" / "reference" per cell
  meta.json    source file, label column, group names, n, gene coverage"""
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data  # noqa: E402

MODEL = "/path/to/scimilarity/model_v1.1"
ROOT = "/path/to/research/analysis"
ATLASES = f"{ROOT}/atlases"
CR = f"{ROOT}/cell_cycle"
DATA = f"{CR}/data"
OUT = f"{CR}/cc_contrasts"
TUREP = f"{ROOT}/turep_case/nat_med_2019"
GO = sig_data.read_gene_order(MODEL)

# name -> (source kind, path, label column, target labels, reference labels)
#   kind "p2"  local atlases attribution h5ad (obs['label'] or an external labels file)
#   kind "cx"  CELLxGENE h5ad (Ensembl var_names, counts in .raw)
#   kind "nd"  neural_diff bundle (own gene space, symbol var_names)
#   kind "sym" any local h5ad whose var_names are already symbols and whose .X is raw counts
BM = f"{DATA}/bonemarrow_dev.h5ad"
CONTRASTS = {
    # ---- haematopoietic development (one dataset, five contrasts) ----
    "bm_cycling_progenitor": ("cx", BM, "AuthorCellType", ["Cycling Progenitor"],
                              ["HSC", "MPP-MkEry", "MPP-MyLy", "LMPP"]),
    "bm_gmp_cycle":          ("cx", BM, "AuthorCellType", ["GMP-Cycle"],
                              ["Early GMP", "GMP-Mono", "GMP-Neut"]),
    "bm_prepdc_cycling":     ("cx", BM, "AuthorCellType", ["Pre-pDC Cycling"], ["Pre-pDC"]),
    "bm_prob_cycling":       ("cx", BM, "AuthorCellType", ["Pro-B Cycling"],
                              ["Pro-B VDJ", "Pre-ProB"]),
    "bm_nk_prolif":          ("cx", BM, "AuthorCellType", ["NK Proliferating"],
                              ["NK", "NK CD56high"]),
    "bm_t_prolif":           ("cx", BM, "AuthorCellType", ["T Proliferating"],
                              ["CD4 Naive", "CD8 Naive", "CD4 Central Memory",
                               "CD8 Central Memory"]),
    # ---- fetal development ----
    "gut_g2ms_enterocyte":   ("cx", f"{DATA}/dev_gut.h5ad", "cell_name_detailed",
                              ["G2M/S enterocytes"],
                              ["Early enterocyte", "Enterocyte", "Colonocytes"]),
    "gonad_prolif_stromal":  ("cx", f"{DATA}/fetal_gonad.h5ad", "cell_type_authors",
                              ["Proliferating stromal cells"],
                              ["Stromal cells from the ovary",
                               "Stromal cells from mesonephroi/Fallopian tube"]),
    # ---- iPSC -> midbrain differentiation ----
    "nd_pfpp":               ("nd", None, "celltype", ["P_FPP"], ["FPP"]),
    "nd_psert":              ("nd", None, "celltype", ["P_Sert"], ["Sert"]),
    # ---- tumour / immune (already local) ----
    "zhao_cycling_asc":      ("p2", f"{ATLASES}/zhao_attribution_scimilarity.h5ad", "label",
                              ["cycling ASC"], ["noncycling ASC"]),
    "dominguez_cycling":     ("p2", f"{ATLASES}/dominguez_conde_attribution_scimilarity.h5ad", "label",
                              ["Cycling immune mix"], None),   # None -> all other labels
    "bcc_tcell_prolif":      ("p2", f"{TUREP}/bcc_allcells_attribution_scimilarity.h5ad", "label",
                              ["Tcell_prolif"],
                              ["CD8_mem_T_cells", "CD4_T_cells", "CD8_act_T_cells", "Tregs"]),
    # ---- colorectal tumour epithelium (the only non-immune tumour system here): 100 % integer counts,
    # symbol var_names, no .raw and no layers -- hence kind "sym".
    "pelka_prolif_epi":      ("sym", f"{ROOT}/dataset_files/pelka_crc_epi/pelka_crc_epi_raw.h5ad",
                              "label", ["cE03 (Stem/TA-like prolif)"], ["cE01 (Stem/TA-like)"]),
}
MIN_TARGET = 100


def _align(counts_adata, tag):
    """Align a raw-count AnnData (symbol var_names) to the 28,231-gene space, RAW COUNTS OUT.

    Deliberately does NOT go through sig_data.harmonize(). That helper calls
    SIGnature.utils.lognorm_counts, which does `data.X = data.layers["counts"]` -- an alias, not a
    copy -- and then lets scanpy normalise X *in place*. The counts layer is destroyed as a side
    effect, so harmonize()'s layers['counts'] comes back log-normalised. (The same aliasing already
    makes layers['counts'] of every atlases/*_attribution_scimilarity.h5ad a copy of the log-normalized X;
    the benchmark reads counts from *_fullgene.h5ad.)

    align_dataset alone does the gene-space alignment without touching values, so we take its X.
    """
    from SIGnature.utils import align_dataset

    a = counts_adata.copy()
    a.var_names_make_unique()
    measured = {str(s) for s in a.var_names}.intersection(GO)
    h = align_dataset(a, GO)
    counts = sp.csr_matrix(h.X).copy()
    d = counts.data
    integer = bool(np.all(d == np.round(d))) if counts.nnz else False
    print(f"    aligned -> {counts.shape}, measured-in-panel {len(measured)}/{len(GO)}, "
          f"counts integer={integer} max={d.max() if counts.nnz else 0:g}", flush=True)
    if not integer:
        raise RuntimeError(f"{tag}: aligned matrix is not integer counts -- wrong source matrix")
    return counts, len(measured)


def load_cx(path, col, targets, refs):
    """CELLxGENE: Ensembl var_names, symbols in var['feature_name'], counts in .raw.X."""
    A = ad.read_h5ad(path, backed="r")
    lab = A.obs[col].astype(str).to_numpy()
    keep_t = np.isin(lab, targets)
    keep_r = np.isin(lab, refs) if refs is not None else ~keep_t
    idx = np.flatnonzero(keep_t | keep_r)
    print(f"    subset {len(idx)} cells ({keep_t.sum()} target / {keep_r.sum()} reference)", flush=True)
    sub = A[idx].to_memory()
    A.file.close()
    y = np.where(np.isin(sub.obs[col].astype(str).to_numpy(), targets), "target", "reference")

    if sub.raw is not None:
        X = sub.raw.X
        rv = sub.raw.var
        syms = (rv["feature_name"].astype(str).to_numpy() if "feature_name" in rv.columns
                else sub.var["feature_name"].astype(str).to_numpy())
    else:
        X = sub.X
        syms = sub.var["feature_name"].astype(str).to_numpy()
    raw = ad.AnnData(X=sp.csr_matrix(X), var=pd.DataFrame(index=pd.Index(syms, name=None)))
    raw.var_names_make_unique()
    return raw, y


def load_p2(path, col, targets, refs):
    """Local h5ad already in gene_order.

    Its layers['counts'] cannot be trusted: for every atlases/*_attribution_scimilarity.h5ad the
    layer is a byte-identical copy of the log-normalised X (see _align for the aliasing bug that
    caused it). So the layer is validated as integer first, and when it fails we fall back to the
    matching *_fullgene.h5ad -- which does hold real counts -- joined on barcode and aligned.
    """
    A = ad.read_h5ad(path, backed="r")
    lab = A.obs[col].astype(str).to_numpy()
    keep_t = np.isin(lab, targets)
    keep_r = np.isin(lab, refs) if refs is not None else ~keep_t
    idx = np.flatnonzero(keep_t | keep_r)
    print(f"    subset {len(idx)} cells ({keep_t.sum()} target / {keep_r.sum()} reference)", flush=True)
    sub = A[idx].to_memory()
    A.file.close()
    y = np.where(np.isin(sub.obs[col].astype(str).to_numpy(), targets), "target", "reference")
    assert list(map(str, sub.var_names)) == list(GO), "h5ad not in gene_order!"

    C = sp.csr_matrix(sub.layers["counts"])
    if C.nnz and bool(np.all(C.data == np.round(C.data))):
        print("    counts layer validated as integer", flush=True)
        return C, y, len(GO)

    stem = os.path.basename(path).replace("_attribution_scimilarity.h5ad", "")
    fg_path = f"{ATLASES}/{stem}_fullgene.h5ad"
    print(f"    !! counts layer is NOT integer (lognorm alias); falling back to "
          f"{os.path.basename(fg_path)}", flush=True)
    fg = ad.read_h5ad(fg_path)
    pmap = {str(b): i for i, b in enumerate(fg.obs_names)}
    want = [str(b) for b in sub.obs_names]
    missing = [b for b in want if b not in pmap]
    if missing:
        raise RuntimeError(f"{len(missing)} barcodes absent from {fg_path} (e.g. {missing[:3]})")
    raw = ad.AnnData(X=sp.csr_matrix(fg.X)[[pmap[b] for b in want]],
                     var=pd.DataFrame(index=fg.var_names))
    del fg
    counts, cov = _align(raw, stem)
    return counts, y, cov


def load_nd(col, targets, refs):
    """neural_diff bundle: own 32,738-gene space, symbols as var_names, raw counts."""
    p = f"{ROOT}/ans_replication/bundles/neural_diff_a_raw.h5ad"
    if not os.path.exists(p):
        p = f"{ROOT}/ans_replication/bundles/neural_diff_a_log.h5ad"
        print(f"    !! raw bundle missing, falling back to {os.path.basename(p)}", flush=True)
    A = ad.read_h5ad(p, backed="r")
    lab = A.obs[col].astype(str).to_numpy()
    keep_t = np.isin(lab, targets)
    keep_r = np.isin(lab, refs) if refs is not None else ~keep_t
    idx = np.flatnonzero(keep_t | keep_r)
    print(f"    subset {len(idx)} cells ({keep_t.sum()} target / {keep_r.sum()} reference)", flush=True)
    sub = A[idx].to_memory()
    A.file.close()
    y = np.where(np.isin(sub.obs[col].astype(str).to_numpy(), targets), "target", "reference")
    raw = ad.AnnData(X=sp.csr_matrix(sub.X), var=pd.DataFrame(index=sub.var_names))
    raw.var_names_make_unique()
    return raw, y


def load_sym(path, col, targets, refs):
    """Local h5ad with symbol var_names and RAW COUNTS in .X (no .raw, no layers).

    The values are checked to be integer (see `_align`).
    """
    A = ad.read_h5ad(path, backed="r")
    lab = A.obs[col].astype(str).to_numpy()
    keep_t = np.isin(lab, targets)
    keep_r = np.isin(lab, refs) if refs is not None else ~keep_t
    idx = np.flatnonzero(keep_t | keep_r)
    print(f"    subset {len(idx)} cells ({keep_t.sum()} target / {keep_r.sum()} reference)", flush=True)
    sub = A[idx].to_memory()
    A.file.close()
    y = np.where(np.isin(sub.obs[col].astype(str).to_numpy(), targets), "target", "reference")
    X = sp.csr_matrix(sub.X)
    if not (X.nnz and bool(np.all(X.data == np.round(X.data)))):
        raise RuntimeError(f"{path}: .X is not integer counts")
    print("    .X validated as integer counts", flush=True)
    raw = ad.AnnData(X=X, var=pd.DataFrame(index=sub.var_names))
    raw.var_names_make_unique()
    return raw, y


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(CONTRASTS)
    rows = []
    for name in only:
        kind, path, col, targets, refs = CONTRASTS[name]
        print(f"\n=== {name}  [{kind}] ===", flush=True)
        try:
            if kind == "p2":
                counts, y, cov = load_p2(path, col, targets, refs)
            elif kind == "cx":
                raw, y = load_cx(path, col, targets, refs)
                counts, cov = _align(raw, name)
            elif kind == "nd":
                raw, y = load_nd(col, targets, refs)
                counts, cov = _align(raw, name)
            elif kind == "sym":
                raw, y = load_sym(path, col, targets, refs)
                counts, cov = _align(raw, name)
            else:
                raise ValueError(kind)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append(dict(contrast=name, status=f"ERROR {type(e).__name__}: {e}"))
            continue

        n_t = int((y == "target").sum())
        if n_t < MIN_TARGET:
            print(f"    !! only {n_t} target cells, skipping", flush=True)
            rows.append(dict(contrast=name, status=f"TOO_SMALL n_target={n_t}"))
            continue
        d = f"{OUT}/{name}"
        os.makedirs(d, exist_ok=True)
        sp.save_npz(f"{d}/counts.npz", sp.csr_matrix(counts))
        with open(f"{d}/labels.txt", "w") as fh:
            fh.write("\n".join(y) + "\n")
        meta = dict(contrast=name, source=path or "neural_diff bundle", kind=kind,
                    label_col=col, target_labels=targets, reference_labels=refs,
                    n_target=n_t, n_reference=int((y == "reference").sum()),
                    n_genes=int(counts.shape[1]), genes_measured_in_panel=int(cov))
        json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
        rows.append(dict(contrast=name, status="ok", **{k: meta[k] for k in
                    ("n_target", "n_reference", "genes_measured_in_panel")}))
        print(f"    wrote {d}", flush=True)

    df = pd.DataFrame(rows)
    f = f"{CR}/identity_contrasts_built.csv"
    if os.path.exists(f) and len(only) < len(CONTRASTS):
        old = pd.read_csv(f)
        df = pd.concat([old[~old.contrast.isin(df.contrast)], df], ignore_index=True)
    df.to_csv(f, index=False)
    print("\n=== BUILD SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
