"""Raw-counts .h5ad (cells x genes, HGNC symbols, obs['label']) for three datasets added to the analyses;
the input of dataset_pipeline.py.

  python load_datasets.py baron       -> dataset_files/baron_pancreas/baron_pancreas_raw.h5ad
  python load_datasets.py macparland  -> dataset_files/macparland_liver/macparland_liver_raw.h5ad
  python load_datasets.py pelka       -> dataset_files/pelka_crc_epi/pelka_crc_epi_raw.h5ad (epithelial cells)"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

PD = "/path/to/research/analysis/dataset_files"


def baron():
    d = f"{PD}/baron_pancreas_gse84133"
    frames = []
    for f in sorted(glob.glob(f"{d}/GSM*human*_umifm_counts.csv.gz")):
        x = pd.read_csv(f, index_col=0)
        x["donor"] = os.path.basename(f).split("_")[1]
        frames.append(x)
    df = pd.concat(frames)
    obs = df[["barcode", "assigned_cluster", "donor"]].reset_index(drop=True)
    obs.columns = ["barcode", "label", "donor"]
    X = df.drop(columns=["barcode", "assigned_cluster", "donor"])
    A = ad.AnnData(sp.csr_matrix(X.values.astype("float32")),
                   obs=obs, var=pd.DataFrame(index=X.columns.astype(str)))
    out = f"{PD}/baron_pancreas/baron_pancreas_raw.h5ad"
    _write(A, out, "baron pancreas")


def macparland():
    d = f"{PD}/macparland_liver_gse115469"
    df = pd.read_csv(f"{d}/GSE115469_Data.csv.gz", index_col=0)          # genes x cells
    A = ad.AnnData(sp.csr_matrix(df.values.T.astype("float32")),
                   obs=pd.DataFrame(index=df.columns.astype(str)),
                   var=pd.DataFrame(index=df.index.astype(str)))
    meta = pd.read_csv(f"{d}/GSE115469_CellClusterType.txt.gz", sep="\t").set_index("CellName")
    A.obs["label"] = meta["CellType"].reindex(A.obs_names).values
    A.obs["sample"] = meta["Sample"].reindex(A.obs_names).values
    A = A[~A.obs["label"].isna()].copy()
    out = f"{PD}/macparland_liver/macparland_liver_raw.h5ad"
    _write(A, out, "macparland liver")


def pelka():
    import scanpy as sc
    d = f"{PD}/pelka_crc_gse178341"
    A = sc.read_10x_h5(f"{d}/GSE178341_crc10x_full_c295v4_submit.h5")
    A.var_names_make_unique()
    lab = pd.read_csv(f"{d}/GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz").set_index("sampleID")
    for c in ["clTopLevel", "cl295v11SubFull"]:
        A.obs[c] = lab[c].reindex(A.obs_names).values
    meta = pd.read_csv(f"{d}/GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz")
    idcol = "cellID" if "cellID" in meta.columns else meta.columns[0]
    meta = meta.set_index(idcol)
    if "SPECIMEN_TYPE" in meta.columns:
        A.obs["specimen"] = meta["SPECIMEN_TYPE"].reindex(A.obs_names).values
    # focus: epithelial compartment (malignant + normal epithelial fine states) — keeps it tractable
    A = A[A.obs["clTopLevel"] == "Epi"].copy()
    A.obs["label"] = A.obs["cl295v11SubFull"].astype(str)
    A = A[~A.obs["label"].isin(["nan", "None", ""])].copy()
    print(f"[pelka] Epi cells={A.n_obs}, fine states={A.obs['label'].nunique()}", flush=True)
    # cap at ~40k cells, stratified by fine label, for tractable attribution
    CAP = 40000
    if A.n_obs > CAP:
        rng = np.random.default_rng(0)
        idx = (pd.Series(np.arange(A.n_obs)).groupby(A.obs["label"].values, group_keys=False)
               .apply(lambda s: s.sample(min(len(s), max(30, int(CAP * len(s) / A.n_obs))), random_state=0)))
        A = A[np.sort(idx.values)].copy()
        print(f"[pelka] capped to {A.n_obs} cells (stratified)", flush=True)
    out = f"{PD}/pelka_crc_epi/pelka_crc_epi_raw.h5ad"
    _write(A, out, "pelka CRC (Epi subset)")


def _write(A, out, name):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    A.obs["label"] = A.obs["label"].astype(str)
    vc = A.obs["label"].value_counts()
    A.write_h5ad(out)
    print(f"[{name}] {A.n_obs} cells x {A.n_vars} genes -> {out}")
    print(f"  labels ({len(vc)}): {vc.head(20).to_dict()}", flush=True)


if __name__ == "__main__":
    {"baron": baron, "macparland": macparland, "pelka": pelka}[sys.argv[1]]()
