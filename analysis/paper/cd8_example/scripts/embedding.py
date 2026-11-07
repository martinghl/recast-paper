"""SCimilarity embedding and UMAP of the four CD8 T subtypes of the Zhao data (Fig. 2a).

GPU (SCimilarity encoder).  Encodes the harmonized log-normalized expression stored in
atlases/zhao_attribution_scimilarity.h5ad (var_names = SCimilarity's 28,231-gene order) and runs UMAP on the 128-d
embedding with fixed parameters, which are saved with the coordinates.
Output: paper/cd8_example/data/cd8_embedding.csv (umap1, umap2, state) + cd8_embedding_params.json
"""
import sys, os, json
sys.path.insert(0, "/path/to/research/external/Signature/src")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
import attributions as sig_attr

H5 = "atlases/zhao_attribution_scimilarity.h5ad"
CD8 = ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"]   # CRCR3 = source typo for CXCR3
DISP = {"CRCR3+ CD8": "CXCR3+ CD8"}
UMAP_PARAMS = dict(n_neighbors=25, min_dist=0.3, n_components=2, metric="euclidean", random_state=0)

a = ad.read_h5ad(H5)
w = sig_attr.get_wrapper(use_gpu=True)
assert list(a.var_names) == list(w.gene_order), "h5ad var_names must equal SCimilarity gene_order"
mask = a.obs["label"].astype(str).isin(CD8).values
sub = a[mask]
lab = sub.obs["label"].astype(str).map(lambda s: DISP.get(s, s)).values
X = sub.X
X = X.toarray() if sp.issparse(X) else np.asarray(X)
print(f"encoding {X.shape[0]} CD8 cells through SCimilarity encoder ...", flush=True)

dev = next(w.model.parameters()).device
embs = []
with torch.no_grad():
    for i in range(0, X.shape[0], 512):
        xb = torch.tensor(X[i:i + 512], dtype=torch.float32, device=dev)
        embs.append(w.model(xb).detach().cpu().numpy())
Z = np.concatenate(embs, 0)
print("representation:", Z.shape, flush=True)

import umap
reducer = umap.UMAP(**UMAP_PARAMS)
U = reducer.fit_transform(Z)
out = pd.DataFrame({"umap1": U[:, 0], "umap2": U[:, 1], "state": lab})
out.to_csv("paper/cd8_example/data/cd8_embedding.csv", index=False)
json.dump({"umap_params": UMAP_PARAMS, "n_cells": int(X.shape[0]), "rep_dim": int(Z.shape[1]),
           "source_h5ad": H5, "states": {DISP.get(s, s): int((lab == DISP.get(s, s)).sum()) for s in CD8}},
          open("paper/cd8_example/data/cd8_embedding_params.json", "w"), indent=2)
print("saved cd8_embedding.csv + params;", {s: int((lab == s).sum()) for s in np.unique(lab)}, flush=True)
