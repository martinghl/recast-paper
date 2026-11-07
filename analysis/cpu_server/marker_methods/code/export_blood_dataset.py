#!/usr/bin/env python3
"""Write one study of the CellTypist "Blood organ atlas" (Xu et al. 2023;
https://cog.sanger.ac.uk/celltypist/Resources/Organ_atlas/Blood/Blood.h5ad) as plain files.

Blood.h5ad holds four studies (Ren et al. 2021, Stephenson et al. 2021, Yoshida et al. 2021, Dominguez Conde et al.
2022; 335,916 cells x 47,332 genes): .X normalized, .raw.X raw integer counts, obs['Dataset'] the study, and the
annotation columns (Curated_annotation, Original_annotation, CellHint_harmonised_group, cell_type, donor_id, ...)
that the benchmark's code/prep-{ren,stephenson,yoshida,dominguez_conde}.R read.  The files are read back by the
full_gene_sce_<id>.R scripts with base R and Matrix::readMM().

Output, next to <out_prefix>:
  <out_prefix>_counts.mtx    -- genes (rows) x cells (columns), raw integer counts
  <out_prefix>_genes.txt     -- one gene symbol per line, in the order of the matrix rows
  <out_prefix>_barcodes.txt  -- one cell barcode per line, in the order of the matrix columns
  <out_prefix>_obs.csv       -- per-cell metadata, in barcode order

Usage: python3 export_blood_dataset.py "<Dataset value>" <out_prefix>"""
import sys
import os
import numpy as np
import anndata as ad
from scipy.io import mmwrite
from scipy.sparse import csr_matrix

BLOOD_H5AD = "/path/to/cpu_server/analysis/marker_methods/data/downloaded_data/Blood.h5ad"
BLOOD_URL = "https://cog.sanger.ac.uk/celltypist/Resources/Organ_atlas/Blood/Blood.h5ad"


def main():
    study, out_prefix = sys.argv[1], sys.argv[2]

    if not os.path.exists(BLOOD_H5AD):
        import urllib.request
        os.makedirs(os.path.dirname(BLOOD_H5AD), exist_ok=True)
        urllib.request.urlretrieve(BLOOD_URL, BLOOD_H5AD)

    adata = ad.read_h5ad(BLOOD_H5AD, backed="r")
    mask = (adata.obs["Dataset"] == study).values
    n_hit = int(mask.sum())
    if n_hit == 0:
        raise SystemExit(f"no cells found for Dataset == {study!r}; available: "
                          f"{sorted(adata.obs['Dataset'].unique().tolist())}")

    sub = adata[mask].to_memory()
    raw_X = sub.raw.X  # cells x genes, raw counts (materialized, in-memory)

    Xchk = raw_X[:200, :200]
    if hasattr(Xchk, "toarray"):
        Xchk = Xchk.toarray()
    is_int = bool(np.allclose(Xchk, np.round(Xchk)))
    print(f"{study}: {sub.n_obs} cells x {sub.n_vars} genes; "
          f"raw-counts sample min={Xchk.min()} max={Xchk.max()} all_int={is_int}")
    if not is_int:
        raise SystemExit("raw counts sample is not integer-valued -- aborting export")

    # genes (rows) x cells (cols), matching SingleCellExperiment convention
    counts_gxc = csr_matrix(raw_X).T.tocoo()

    mmwrite(out_prefix + "_counts.mtx", counts_gxc)
    with open(out_prefix + "_genes.txt", "w") as f:
        f.write("\n".join(sub.raw.var_names.astype(str)) + "\n")
    with open(out_prefix + "_barcodes.txt", "w") as f:
        f.write("\n".join(sub.obs_names.astype(str)) + "\n")
    sub.obs.to_csv(out_prefix + "_obs.csv", index=True)

    print("wrote", out_prefix + "_{counts.mtx,genes.txt,barcodes.txt,obs.csv}",
          f"({sub.n_obs} cells x {sub.n_vars} genes)")


if __name__ == "__main__":
    main()
