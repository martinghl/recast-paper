# Export the all-gene pbmc3k SingleCellExperiment (data/real_data_fullgene/pbmc3k.rds, from
# full_gene_sce_pbmc3k.R) to .h5ad for the SCimilarity attributions:
#   - X          = raw integer counts, cells x genes
#   - var_names  = gene symbols, unique
#   - obs_names  = cell barcodes in the benchmark's order
#   - obs$label  = the 9 cell-type labels
#   - counts only (no logcounts / scaledata)
#
# full_gene_sce_pbmc3k.R replaced the Ensembl rownames by scuttle::uniquifyFeatureNames(ENSEMBL_ID, Symbol_TENx), so
# rownames(sce) are symbols; the script checks this and stops otherwise.

suppressMessages({
  library(SingleCellExperiment)
  library(reticulate)
})

# Bare `Rscript /path/to/envs/marker_methods/bin/Rscript ...` does NOT activate
# the marker_methods conda env's PATH, so reticulate's auto-discovery can silently
# fall back to provisioning its own bare ephemeral Python (confirmed: it grabs
# a fresh cpython 3.12 with only numpy, no scipy/anndata, and import() fails
# downstream). Pin explicitly so this script is correct regardless of how it
# is invoked.
reticulate::use_python(
  "/path/to/envs/marker_methods/bin/python",
  required = TRUE
)

sce <- readRDS(here::here("data", "real_data_fullgene", "pbmc3k.rds"))

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(sce),
  !is.null(colLabels(sce))
)

gene_ids <- rownames(sce)

# --- Verify gene identifiers are usable symbols, not bare Ensembl IDs -----
n_bare_ensembl <- sum(grepl("^ENSG[0-9]+$", gene_ids))
n_hybrid <- sum(grepl("-ENSG[0-9]+$", gene_ids))
n_dup <- sum(duplicated(gene_ids))

cat(sprintf(
  "gene id check: %d total | %d bare-Ensembl | %d symbol-ENSG hybrid (dup-symbol disambiguation suffix) | %d duplicated\n",
  length(gene_ids), n_bare_ensembl, n_hybrid, n_dup
))

if (n_bare_ensembl > 0) {
  stop(
    n_bare_ensembl, " / ", length(gene_ids), " rownames(sce) are bare ",
    "Ensembl IDs (regex ^ENSG[0-9]+$). rowData(sce) no longer carries the ",
    "original Symbol_TENx/ENSEMBL_ID columns (the upstream prep script ",
    "overwrote rowData with the rownames themselves), so this script ",
    "cannot silently remap them to symbols. Investigate data/real_data_fullgene/",
    "pbmc3k.rds's provenance before proceeding -- do not export Ensembl IDs ",
    "as var_names; SCimilarity matches its panel by symbol."
  )
}
stopifnot("gene_ids (var_names) must be unique for anndata" = n_dup == 0)

markers_present <- c("MS4A1", "CD14", "NKG7") %in% gene_ids
cat("sanity marker genes present (MS4A1, CD14, NKG7):",
    paste(markers_present, collapse = ","), "\n")

# --- Verify raw counts are integer-valued before export -------------------
count_vals <- counts(sce)@x
stopifnot("counts assay must be all integer-valued (raw counts)" =
            all(count_vals %% 1 == 0))
cat("counts integer check OK (", length(count_vals), "nonzero entries, max =",
    max(count_vals), ")\n")

# --- Build AnnData: X = raw counts ONLY (cells x genes) --------------------
sp <- import("scipy.sparse")
ad <- import("anndata")

X <- sp$csr_matrix(r_to_py(Matrix::t(counts(sce))))  # cells x genes, RAW counts

obs_df <- data.frame(
  label = as.character(colLabels(sce)),
  row.names = colnames(sce),
  stringsAsFactors = FALSE
)
var_df <- data.frame(row.names = gene_ids)

adata <- ad$AnnData(
  X = X,
  obs = r_to_py(obs_df),
  var = r_to_py(var_df)
)

dir.create(here::here("atlases"), showWarnings = FALSE, recursive = TRUE)
out_path <- here::here("atlases", "pbmc3k_fullgene.h5ad")
adata$write_h5ad(out_path)

cat("wrote h5ad:", adata$shape[[1]], "cells x", adata$shape[[2]], "genes ->", out_path, "\n")
cat("obs label counts:\n")
print(table(obs_df$label))
cat("n unique labels:", length(unique(obs_df$label)), "\n")
cat("first 5 var_names:", paste(head(gene_ids, 5), collapse = ", "), "\n")
cat("first 3 / last 3 obs_names:",
    paste(head(rownames(obs_df), 3), collapse = ","), " / ",
    paste(tail(rownames(obs_df), 3), collapse = ","), "\n")
