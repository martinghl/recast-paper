#!/usr/bin/env python
"""One CELLxGENE dataset -> <key>_raw.h5ad (raw counts, HGNC symbols, obs['label']), built as the HLCA input was
(paper/datasets/scripts/build_hlca.py): Census 2025-11-08, primary cells only, <= CAP cells per subtype
(random_state=SEED), X_name='raw', genes with only an Ensembl id dropped and duplicate symbols summed.  The datasets'
labels and donors are read from the Census-hosted source h5ad obs (HTTP range reads) and joined to Census on
(dataset_id, observation_joinid).
Env census.  Usage: python -I build_cellxgene_dataset.py <key> [<key> ...]"""
import os
import sys

import anndata as ad
import cellxgene_census
import fsspec
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import added_datasets as D  # noqa: E402

FS = "/path/to/research/analysis/subtype_benchmark/full_benchmark"
URL = "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com/cell-census/{v}/h5ads/{d}.h5ad"
OBS_COLS = ["soma_joinid", "dataset_id", "observation_joinid", "cell_type", "donor_id", "disease", "assay",
            "tissue", "sex", "development_stage", "is_primary_data"]


def _col(f, c):
    n = f["obs"][c]
    if isinstance(n, h5py.Group):
        cats = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]], dtype=object)
        codes = n["codes"][:]
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
    v = n[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object) if v.dtype.kind in "SO" else v


def source_obs(did, cols):
    fs = fsspec.filesystem("http")
    with fs.open(URL.format(v=D.CENSUS_VERSION, d=did), "rb", block_size=1 << 22, cache_type="blockcache") as fh:
        f = h5py.File(fh, "r")
        o = pd.DataFrame({c: _col(f, c) for c in cols})
    o["dataset_id"] = did
    return o


def select(key):
    e = D.CXG[key]
    cols = sorted({e["label_col"], "donor_id", "observation_joinid", "is_primary_data", *e["where"]})
    parts = []
    for did in e["dataset_ids"]:
        o = source_obs(did, cols)
        assert o["observation_joinid"].is_unique, f"{did}: duplicate observation_joinid"
        print(f"[{key}] {did}: {len(o)} cells in source obs", flush=True)
        parts.append(o)
    o = pd.concat(parts, ignore_index=True)
    keep = o["is_primary_data"].astype(bool).to_numpy()
    if e["donors"] is not None:
        keep &= o["donor_id"].isin(e["donors"]).to_numpy()
    for c, vals in e["where"].items():
        keep &= o[c].astype(str).isin(vals).to_numpy()
    o = o[keep].copy()
    o["author_label"] = o[e["label_col"]].astype(str)
    o["label"] = o["author_label"].map(e["mapping"])
    unmapped = o.loc[o["label"].isna(), "author_label"].value_counts()
    print(f"[{key}] eligible cells {len(o)}; dropped author labels (not in mapping):\n{unmapped.to_string()}", flush=True)
    o = o[o["label"].notna()]
    print(f"[{key}] pre-subsample per state x donor:\n"
          f"{o.groupby(['label', 'donor_id']).size().unstack(fill_value=0).to_string()}", flush=True)
    parts = []
    for lab, g in o.groupby("label", sort=False):
        if len(g) > D.CAP:
            g = g.sample(n=D.CAP, random_state=D.SEED)
        parts.append(g)
    return pd.concat(parts)


def fetch(key):
    e = D.CXG[key]
    out_dir = f"{FS}/data/raw/{key}"
    os.makedirs(out_dir, exist_ok=True)
    sel = select(key)
    with cellxgene_census.open_soma(census_version=D.CENSUS_VERSION) as census:
        filt = " or ".join(f"dataset_id == '{d}'" for d in e["dataset_ids"])
        cen = cellxgene_census.get_obs(census, "homo_sapiens", value_filter=filt,
                                       column_names=["soma_joinid", "dataset_id", "observation_joinid"])
        sel = sel.merge(cen, on=["dataset_id", "observation_joinid"], how="left", validate="one_to_one")
        assert sel["soma_joinid"].notna().all(), f"{key}: cells missing from Census"
        sel = sel.sort_values("soma_joinid")
        adata = cellxgene_census.get_anndata(
            census, organism="homo_sapiens", measurement_name="RNA", X_name="raw",
            obs_coords=sel["soma_joinid"].astype(np.int64).to_numpy(), obs_column_names=OBS_COLS,
            var_column_names=["feature_id", "feature_name"])
    print(f"[{key}] raw fetched {adata.shape}", flush=True)
    m = sel.set_index(["dataset_id", "observation_joinid"])
    idx = pd.MultiIndex.from_arrays([adata.obs["dataset_id"].astype(str), adata.obs["observation_joinid"].astype(str)])
    adata.obs["label"] = m.loc[idx, "label"].to_numpy()
    adata.obs["author_label"] = m.loc[idx, "author_label"].to_numpy()
    assert adata.obs["label"].notna().all()
    assert (adata.obs["donor_id"].astype(str).to_numpy() == m.loc[idx, "donor_id"].astype(str).to_numpy()).all()

    # HGNC symbols only; duplicate symbols summed (build_hlca.py step 5)
    fid = adata.var["feature_id"].astype(str).values
    fname = adata.var["feature_name"].astype(str).values
    has_symbol = fname != fid
    print(f"[{key}] genes total {adata.n_vars}; with HGNC symbol {int(has_symbol.sum())}", flush=True)
    adata = adata[:, has_symbol].copy()
    sym = adata.var["feature_name"].astype(str).values
    codes, uniques = pd.factorize(sym)
    M = sp.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
                      shape=(len(codes), len(uniques)))
    X = sp.csr_matrix(adata.X.tocsr().astype(np.float32) @ M)
    ens = pd.Series(fid[has_symbol]).groupby(codes).apply(lambda s: ";".join(sorted(set(s)))).reindex(range(len(uniques))).values
    obs = adata.obs[["label", "author_label", "cell_type", "donor_id", "dataset_id", "observation_joinid",
                     "disease", "assay", "tissue", "sex", "development_stage"]].copy()
    obs["sample"] = obs["donor_id"].astype(str).values
    obs.index = [f"{d[:8]}_{j}" for d, j in zip(obs["dataset_id"], obs["observation_joinid"])]
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"ensembl_id": ens}, index=pd.Index(uniques, name=None)))
    assert np.allclose(out.X.data, np.round(out.X.data)) and out.X.min() >= 0, f"{key}: not integer counts"
    tot = np.asarray(out.X.sum(1)).ravel()
    print(f"[{key}] {out.shape}; integer counts; median per-cell total {np.median(tot):.0f}; max {out.X.max():.0f}", flush=True)
    print(f"[{key}] post-subsample per state x donor:\n"
          f"{out.obs.groupby(['label', 'donor_id'], observed=True).size().unstack(fill_value=0).to_string()}", flush=True)
    out.write_h5ad(f"{out_dir}/{key}_raw.h5ad")
    print(f"[{key}] SAVED {out_dir}/{key}_raw.h5ad", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        fetch(k)
