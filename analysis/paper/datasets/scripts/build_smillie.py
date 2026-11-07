#!/usr/bin/env python
"""Smillie et al. 2019 colon epithelium (CELLxGENE Discover, healthy) -> 6 subtypes (lineage fs_gut_smillie)."""
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad

SRC = "/path/to/scratch/smillie_cxg.h5ad"
OUT = "/path/to/research/analysis/dataset_files/smillie_colon/smillie_colon_raw.h5ad"
SEED = 0
CAP = 3000

# CellType cluster of the source study -> subtype label (everything not listed is dropped: TA/Cycling TA/Secretory TA, M cells, immune, stromal)
CT2LABEL = {
    "Enterocytes": "enterocyte",
    "Immature Enterocytes 1": "enterocyte",
    "Immature Enterocytes 2": "enterocyte",
    "Enterocyte Progenitors": "enterocyte",
    "Goblet": "goblet",
    "Immature Goblet": "goblet",
    "Tuft": "tuft",
    "Enteroendocrine": "enteroendocrine",
    "Best4+ Enterocytes": "BEST4",
    "Stem": "stem",
}

a = ad.read_h5ad(SRC)
print("loaded:", a.shape, " disease values:", a.obs["disease"].unique().tolist())
assert set(a.obs["disease"].unique()) == {"normal"}, "expected healthy/normal only!"

# map source clusters -> label, keep only mapped cells (epithelial subtypes)
a.obs["label"] = a.obs["CellType"].astype(str).map(CT2LABEL)
keep = a.obs["label"].notna()
print("cells mapped to a target state:", int(keep.sum()), "of", a.n_obs)
a = a[keep].copy()
print("\nPre-subsample per-label:\n", a.obs["label"].value_counts().to_string())

# subsample <=CAP per label
rng_parts = []
for lab, idx in a.obs.groupby("label", sort=False).groups.items():
    idx = pd.Index(idx)
    if len(idx) > CAP:
        idx = pd.Index(pd.Series(idx).sample(n=CAP, random_state=SEED).values)
    rng_parts.append(idx)
sel_idx = pd.Index(np.concatenate([p.values for p in rng_parts]))
a = a[a.obs_names.isin(sel_idx)].copy()
print("\nPost-subsample per-label:\n", a.obs["label"].value_counts().to_string())
print("total cells:", a.n_obs)

# --- build RAW-count AnnData: use a.raw (raw counts), symbols from feature_name ---
raw = a.raw.to_adata()          # cells x genes, X = raw counts, var has feature_name
raw = raw[a.obs_names].copy()   # align order
X = raw.X
assert X.min() >= 0
Xsub = X[:400].toarray() if hasattr(X, "toarray") else np.asarray(X[:400])
print("\nraw.X integer?", np.allclose(Xsub, np.round(Xsub)), " max:", float(Xsub.max()))

fid = raw.var_names.astype(str).values                       # Ensembl IDs
fname = raw.var["feature_name"].astype(str).values           # HGNC symbols
has_symbol = fname != fid
print("genes total:", len(fid), " with HGNC symbol:", int(has_symbol.sum()),
      " ENSG-only dropped:", int((~has_symbol).sum()))

Xc = X.tocsr()[:, has_symbol]
sym = fname[has_symbol]
ensg = fid[has_symbol]
codes, uniques = pd.factorize(sym)
n_dup = len(codes) - len(uniques)
print("duplicate symbols to collapse:", int(n_dup))
if n_dup > 0:
    M = sp.csr_matrix((np.ones(len(codes), dtype=np.float32),
                       (np.arange(len(codes)), codes)), shape=(len(codes), len(uniques)))
    Xnew = Xc.astype(np.float32) @ M
    ens = pd.Series(ensg).groupby(codes).apply(lambda s: ";".join(sorted(set(s)))).reindex(range(len(uniques))).values
else:
    Xnew = Xc.astype(np.float32)
    ens = ensg

obs = a.obs[["label", "CellType", "donor_id", "Source", "assay", "tissue", "sex", "disease", "Smoking"]].copy()
obs["sample"] = obs["donor_id"].astype(str).values
out = ad.AnnData(X=sp.csr_matrix(Xnew), obs=obs,
                 var=pd.DataFrame({"ensembl_id": ens}, index=pd.Index(uniques, name=None)))

# verify integer counts (full)
allint = np.allclose(out.X.data, np.round(out.X.data))
print("genes after collapse:", out.n_vars, " all_integer(full):", bool(allint), " min:", float(out.X.min()), " max:", float(out.X.max()))

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.write_h5ad(OUT)
print("\nSAVED:", OUT)
print("final shape:", out.shape)
print("label value_counts:\n", out.obs["label"].value_counts().to_string())
print("var_names sample:", list(out.var_names[:12]))
print("n donors:", out.obs["donor_id"].nunique(), " donors:", sorted(out.obs["donor_id"].unique().tolist()))
