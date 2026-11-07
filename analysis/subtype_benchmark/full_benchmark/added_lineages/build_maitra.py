#!/usr/bin/env python
"""Maitra 2023 dlPFC (GEO GSE213982 combined matrix) -> maitra2023_raw.h5ad, built as build_cellxgene_dataset.py
builds the CELLxGENE inputs: control subjects only (the paper's Supplementary Data 1), the broad class in the cell name
(donor.barcode.broad.cluster) mapped to benchmark subtypes, <= CAP cells per subtype (random_state=SEED), integer
counts as deposited.  M24 and M24_2 are one subject sequenced twice: both libraries are kept, under donor M24.  The
deposited genes are symbols only (no Ensembl ids) and already unique, so the symbol collapse changes nothing here.

Runs on the CPU server (env marker_methods).  Usage: python -I build_maitra.py <dl_dir> <out_dir>"""
import gzip
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import added_datasets as D  # noqa: E402

KEY = "maitra2023"
PFX = "GSE213982_combined_counts_matrix"


def select(dl):
    cells = pd.read_csv(f"{dl}/{PFX}_cells_columns.csv.gz")["x"].astype(str)
    assert cells.is_unique
    tok = cells.str.split(".", expand=True)
    assert tok.shape[1] == 4 and tok.notna().all().all(), "cell names are not donor.barcode.broad.cluster"
    o = pd.DataFrame({"cell": cells.values, "library": tok[0].values, "barcode": tok[1].values,
                      "broad": tok[2].values, "cluster": tok[3].values, "col": np.arange(len(cells))})
    missing = sorted(set(D.MAITRA["controls"]) - set(o["library"]))
    assert not missing, f"control libraries not in the matrix: {missing}"
    print(f"[{KEY}] {len(o)} cells from {o['library'].nunique()} libraries; keeping {len(D.MAITRA['controls'])} "
          f"control libraries", flush=True)
    o = o[o["library"].isin(D.MAITRA["controls"])].copy()
    o["donor_id"] = o["library"].replace(D.MAITRA["same_subject"])
    o["author_label"] = o["broad"]
    o["label"] = o["broad"].map(D.MAITRA["mapping"])
    print(f"[{KEY}] control cells {len(o)} ({o['donor_id'].nunique()} subjects); dropped broad classes "
          f"(not in mapping):\n{o.loc[o['label'].isna(), 'author_label'].value_counts().to_string()}", flush=True)
    o = o[o["label"].notna()]
    print(f"[{KEY}] pre-subsample per state x donor:\n"
          f"{o.groupby(['label', 'donor_id']).size().unstack(fill_value=0).to_string()}", flush=True)
    parts = []
    for lab, g in o.groupby("label", sort=False):
        if len(g) > D.CAP:
            g = g.sample(n=D.CAP, random_state=D.SEED)
        parts.append(g)
    return pd.concat(parts)


def main(dl, out_dir):
    sel = select(dl)
    genes = pd.read_csv(f"{dl}/{PFX}_genes_rows.csv.gz")["x"].astype(str)
    col_map = np.full(int(sel["col"].max()) + 1, -1, np.int64)
    col_map[sel["col"].to_numpy()] = np.arange(len(sel))
    rows, cols, vals = [], [], []
    with gzip.open(f"{dl}/{PFX}.mtx.gz", "rt") as fh:
        line = fh.readline()
        assert "coordinate integer" in line, line.strip()
        while line.startswith("%"):
            line = fh.readline()
        nr, nc, nnz = map(int, line.split())
        assert nr == len(genes), (nr, len(genes))
        n_read = 0
        for ch in pd.read_csv(fh, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64, chunksize=20_000_000):
            n_read += len(ch)
            c = ch["c"].to_numpy() - 1
            c = np.where(c < len(col_map), col_map[np.minimum(c, len(col_map) - 1)], -1)
            k = c >= 0
            rows.append(c[k]); cols.append(ch["r"].to_numpy()[k] - 1); vals.append(ch["v"].to_numpy()[k])
            print(f"[{KEY}] read {n_read}/{nnz} entries", flush=True)
    assert n_read == nnz, f"read {n_read} entries, header says {nnz}"
    X = sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(sel), nr))
    # symbols only; duplicate symbols summed (no-op when unique, kept for parity with build_cellxgene_dataset.py)
    codes, uniques = pd.factorize(genes.values)
    print(f"[{KEY}] genes {len(genes)}; unique symbols {len(uniques)}", flush=True)
    M = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
                      shape=(len(codes), len(uniques)))
    X = sp.csr_matrix(X @ M)
    o = pd.DataFrame({"label": sel["label"].values, "author_label": sel["author_label"].values,
                      "cluster": sel["cluster"].values, "donor_id": sel["donor_id"].values,
                      "library": sel["library"].values, "barcode": sel["barcode"].values},
                     index=pd.Index(sel["cell"].values, name=None))
    o["sample"] = o["donor_id"].values
    out = ad.AnnData(X=X, obs=o, var=pd.DataFrame(index=pd.Index(uniques, name=None)))
    assert np.allclose(out.X.data, np.round(out.X.data)) and out.X.min() >= 0, "not integer counts"
    tot = np.asarray(out.X.sum(1)).ravel()
    assert (tot > 0).all(), "cells with zero counts"
    print(f"[{KEY}] {out.shape}; integer counts; median per-cell total {np.median(tot):.0f}; max {out.X.max():.0f}", flush=True)
    print(f"[{KEY}] post-subsample per state x donor:\n"
          f"{out.obs.groupby(['label', 'donor_id'], observed=True).size().unstack(fill_value=0).to_string()}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    out.write_h5ad(f"{out_dir}/{KEY}_raw.h5ad")
    print(f"[{KEY}] SAVED {out_dir}/{KEY}_raw.h5ad", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
