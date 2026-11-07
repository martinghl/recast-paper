#!/usr/bin/env python
"""Hickey 2023 colon epithelium (HuBMAP snRNA-seq, Zenodo 7675308) -> hickey2023colon_raw.h5ad, built as
build_cellxgene_dataset.py builds the CELLxGENE inputs: the dataset's CellType mapped to benchmark subtypes, <= CAP
cells per subtype (random_state=SEED), raw Cell Ranger counts, genes without a symbol dropped and duplicate symbols
summed.

Only the Multiome libraries are used (6 donors).  The 3' v3 libraries (donors B001/B004/B005) used another chemistry
and mostly another gene reference (GRCh38-3.0.0, 33,538 genes vs 36,601 in GRCh38-2020-A), so mixing them would need a
gene intersection and add a library effect.  Every library used must have the same Gene Expression feature list
(asserted); the Peaks features of the Multiome matrices are dropped.

Runs on the CPU server (env marker_methods).  Usage: python -I build_hickey.py <dl_dir> <out_dir> [workers]"""
import gzip
import os
import sys
from multiprocessing import Pool

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import added_datasets as D  # noqa: E402

KEY = "hickey2023colon"
COLON = {"Sigmoid", "Descending", "Transverse", "Ascending"}


def select(dl):
    t = pd.read_csv(f"{dl}/{D.HICKEY['label_file']}", sep="\t")
    assert t["Cell ID"].is_unique
    t["library"] = t["Cell ID"].str.rsplit("_", n=1).str[0]
    t["barcode"] = t["Cell ID"].str.rsplit("_", n=1).str[1]
    m = pd.read_csv(f"{dl}/sample_location_metadata.csv").rename(columns={"SampleNameRNA": "library"})
    t = t.merge(m, on="library", how="left", validate="many_to_one")
    assert t["Donor"].notna().all() and t["Location"].isin(COLON).all()
    print(f"[{KEY}] colon epithelial cells {len(t)}; libraries by Multiome flag:\n"
          f"{t.groupby(['Multiome', 'Donor'])['library'].nunique().to_string()}", flush=True)
    t = t[t["Multiome"] == "Yes"].copy()
    t["author_label"] = t["CellType"].astype(str)
    t["label"] = t["author_label"].map(D.HICKEY["mapping"])
    print(f"[{KEY}] Multiome cells {len(t)}; dropped author labels (not in mapping):\n"
          f"{t.loc[t['label'].isna(), 'author_label'].value_counts().to_string()}", flush=True)
    t = t[t["label"].notna()]
    print(f"[{KEY}] pre-subsample per state x donor:\n"
          f"{t.groupby(['label', 'Donor']).size().unstack(fill_value=0).to_string()}", flush=True)
    parts = []
    for lab, g in t.groupby("label", sort=False):
        if len(g) > D.CAP:
            g = g.sample(n=D.CAP, random_state=D.SEED)
        parts.append(g)
    return pd.concat(parts)


def read_library(args):
    """One Cell Ranger MEX -> (cells x Gene Expression features) CSR for the requested barcodes, in order."""
    dl, lib, barcodes = args
    # Multiome features carry 6 columns (id, name, type, chr, start, end); keep the first three
    feat = pd.read_csv(f"{dl}/{lib}_features.tsv.gz", sep="\t", header=None).iloc[:, :3]
    feat.columns = ["id", "name", "type"]
    bc = pd.read_csv(f"{dl}/{lib}_barcodes.tsv.gz", header=None)[0].astype(str)
    assert bc.is_unique, lib
    want = pd.Series(np.arange(len(bc)), index=bc).reindex(barcodes)
    assert want.notna().all(), f"{lib}: {int(want.isna().sum())} labelled barcodes missing from the MEX"
    col_map = np.full(len(bc), -1, np.int64)
    col_map[want.to_numpy().astype(np.int64)] = np.arange(len(barcodes))
    gex = (feat["type"] == "Gene Expression").to_numpy()
    row_map = np.full(len(feat), -1, np.int64)
    row_map[gex] = np.arange(int(gex.sum()))
    rows, cols, vals = [], [], []
    with gzip.open(f"{dl}/{lib}_matrix.mtx.gz", "rt") as fh:
        line = fh.readline()
        assert "coordinate integer" in line, f"{lib}: {line.strip()}"
        while line.startswith("%"):
            line = fh.readline()
        nr, nc, nnz = map(int, line.split())
        assert nr == len(feat) and nc == len(bc), lib
        n_read = 0
        for ch in pd.read_csv(fh, sep=" ", header=None, names=["r", "c", "v"], dtype=np.int64, chunksize=20_000_000):
            n_read += len(ch)
            r = row_map[ch["r"].to_numpy() - 1]
            c = col_map[ch["c"].to_numpy() - 1]
            k = (r >= 0) & (c >= 0)
            rows.append(c[k]); cols.append(r[k]); vals.append(ch["v"].to_numpy()[k])
    assert n_read == nnz, f"{lib}: read {n_read} entries, header says {nnz}"
    X = sp.csr_matrix((np.concatenate(vals).astype(np.float32), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(barcodes), int(gex.sum())))
    return lib, X, feat.loc[gex, ["id", "name"]].reset_index(drop=True)


def main(dl, out_dir, workers):
    sel = select(dl)
    libs = sorted(sel["library"].unique())
    jobs = [(dl, lib, sel.loc[sel["library"] == lib, "barcode"].tolist()) for lib in libs]
    with Pool(min(workers, len(jobs))) as pool:
        res = {lib: (X, f) for lib, X, f in pool.map(read_library, jobs)}
    f0 = res[libs[0]][1]
    assert len(f0) > 30000, f"only {len(f0)} Gene Expression features"
    for lib in libs:
        assert res[lib][1].equals(f0), f"{lib}: Gene Expression features differ from {libs[0]}"
    print(f"[{KEY}] {len(libs)} libraries, identical Gene Expression features ({len(f0)})", flush=True)
    X = sp.vstack([res[lib][0] for lib in libs]).tocsr()
    obs = pd.concat([sel[sel["library"] == lib] for lib in libs])
    # genes with a symbol only; duplicate symbols summed (build_cellxgene_dataset.py / build_hlca.py step 5)
    fid = f0["id"].astype(str).values
    fname = f0["name"].astype(str).values
    has_symbol = fname != fid
    print(f"[{KEY}] genes total {len(fid)}; with symbol {int(has_symbol.sum())}", flush=True)
    X = X[:, has_symbol]
    codes, uniques = pd.factorize(fname[has_symbol])
    M = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
                      shape=(len(codes), len(uniques)))
    X = sp.csr_matrix(X @ M)
    ens = pd.Series(fid[has_symbol]).groupby(codes).apply(lambda s: ";".join(sorted(set(s)))).reindex(range(len(uniques))).values
    o = pd.DataFrame({"label": obs["label"].values, "author_label": obs["author_label"].values,
                      "donor_id": obs["Donor"].values, "library": obs["library"].values,
                      "location": obs["Location"].values, "barcode": obs["barcode"].values},
                     index=pd.Index(obs["Cell ID"].values, name=None))
    o["sample"] = o["donor_id"].values
    out = ad.AnnData(X=X, obs=o, var=pd.DataFrame({"ensembl_id": ens}, index=pd.Index(uniques, name=None)))
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
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
