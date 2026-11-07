#!/usr/bin/env python
"""HLCA lung epithelium from CELLxGENE Census -> 6 subtypes (lineage fs_lung_hlca)."""
import numpy as np
import pandas as pd
import cellxgene_census
import anndata as ad

CENSUS_VERSION = "2025-11-08"
FULL = "9f222629-9e39-47d0-b83f-e08d610c7479"  # HLCA full/extended dataset
OUT = "/path/to/research/analysis/dataset_files/hlca_lung/hlca_lung_raw.h5ad"
SEED = 0
CAP = 3000

# exact CL cell_type string -> target label
CL2LABEL = {
    "respiratory basal cell": "basal",
    "club cell": "club",
    "multiciliated columnar cell of tracheobronchial tree": "ciliated",
    "multiciliated epithelial cell": "ciliated",
    "ionocyte": "ionocyte",
    "pulmonary alveolar type 1 cell": "AT1",
    "pulmonary alveolar type 2 cell": "AT2",
}
CL_STRINGS = list(CL2LABEL.keys())

with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    # 1) metadata-only pull of the epithelial cells we want (primary only)
    ct_filter = " or ".join([f"cell_type == '{c}'" for c in CL_STRINGS])
    vf = f"dataset_id == '{FULL}' and is_primary_data == True and ({ct_filter})"
    obs = cellxgene_census.get_obs(
        census, "homo_sapiens", value_filter=vf,
        column_names=["soma_joinid", "cell_type", "donor_id", "disease", "assay", "tissue", "sex"],
    )
    obs["label"] = obs["cell_type"].map(CL2LABEL)
    print("Pulled metadata rows:", len(obs))
    print("Pre-subsample per-label:\n", obs["label"].value_counts().to_string())

    # 2) subsample to <=CAP per label (keep all if fewer)
    parts = []
    for lab, g in obs.groupby("label", sort=False):
        if len(g) > CAP:
            g = g.sample(n=CAP, random_state=SEED)
        parts.append(g)
    sel = pd.concat(parts).sort_values("soma_joinid")
    print("\nPost-subsample per-label:\n", sel["label"].value_counts().to_string())
    print("Total selected cells:", len(sel))

    # 3) fetch raw-count expression for ONLY selected cells
    coords = sel["soma_joinid"].to_numpy()
    adata = cellxgene_census.get_anndata(
        census, organism="homo_sapiens", measurement_name="RNA", X_name="raw",
        obs_coords=coords,
        obs_column_names=["soma_joinid", "cell_type", "donor_id", "disease", "assay", "tissue", "sex"],
        var_column_names=["feature_id", "feature_name"],
    )

print("\nRaw fetched adata:", adata.shape)

# 4) label from cell_type (deterministic), verify alignment
adata.obs["label"] = adata.obs["cell_type"].astype(str).map(CL2LABEL)
assert adata.obs["label"].notna().all(), "unmapped cell_type present!"

# 5) clean gene panel -> HGNC symbols only, collapse duplicate symbols by summing counts
import scipy.sparse as sp
fid = adata.var["feature_id"].astype(str).values
fname = adata.var["feature_name"].astype(str).values
has_symbol = fname != fid                      # drop ENSG-only (no HGNC symbol)
print("genes total:", adata.n_vars, " with HGNC symbol:", int(has_symbol.sum()),
      " ENSG-only dropped:", int((~has_symbol).sum()))
adata = adata[:, has_symbol].copy()
sym = adata.var["feature_name"].astype(str).values
n_dup = pd.Series(sym).duplicated().sum()
print("duplicate symbols to collapse:", int(n_dup))
codes, uniques = pd.factorize(sym)             # first-appearance order preserved
if n_dup > 0:
    M = sp.csr_matrix((np.ones(len(codes), dtype=np.float32),
                       (np.arange(len(codes)), codes)), shape=(len(codes), len(uniques)))
    Xnew = adata.X.tocsr().astype(np.float32) @ M           # sum duplicate columns
    ens = pd.Series(fid[has_symbol]).groupby(codes).apply(lambda s: ";".join(sorted(set(s)))).reindex(range(len(uniques))).values
    adata = ad.AnnData(X=sp.csr_matrix(Xnew), obs=adata.obs.copy(),
                       var=pd.DataFrame({"ensembl_id": ens}, index=uniques))
else:
    adata.var = adata.var.rename(columns={"feature_id": "ensembl_id"})[["ensembl_id"]]
    adata.var_names = uniques
adata.var.index.name = None
print("genes after collapse (unique symbols):", adata.n_vars)

# 6) verify raw integer counts
X = adata.X
sub = X[:500].toarray() if hasattr(X, "toarray") else np.asarray(X[:500])
print("X dtype:", X.dtype, "min:", float(sub.min()), "max:", float(sub.max()),
      "all_integer(sample):", bool(np.allclose(sub, np.round(sub))))
# full integer check
if hasattr(X, "data"):
    allint = np.allclose(X.data, np.round(X.data))
else:
    allint = np.allclose(X, np.round(X))
print("all_integer(full):", bool(allint), "min(full):", float(X.min()))

# tidy obs
adata.obs["sample"] = adata.obs["donor_id"].astype(str).values
adata.obs = adata.obs[["label", "cell_type", "donor_id", "sample", "disease", "assay", "tissue", "sex"]]

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
adata.write_h5ad(OUT)
print("\nSAVED:", OUT)
print("final shape:", adata.shape)
print("label value_counts:\n", adata.obs["label"].value_counts().to_string())
print("var_names sample:", list(adata.var_names[:12]))
print("n donors:", adata.obs["donor_id"].nunique(), " diseases:", adata.obs["disease"].unique().tolist()[:8])
