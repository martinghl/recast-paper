# Export data/real_data_fullgene/<id>.rds to atlases/<id>_fullgene.h5ad, as export_h5ad_pbmc3k.R:
#   - X          = raw counts, cells x genes
#   - var_names  = rownames(sce)
#   - obs_names  = cell barcodes in the object's order
#   - obs$label  = colLabels(sce)
#   - counts only (no logcounts / scaledata)
#
# Some atlases have a few genes without a symbol (kept as Ensembl IDs) or duplicated symbols (e.g. lawlor 1,099 of 26,616
# bare Ensembl IDs, ss3_pbmc 490 of 38,630 duplicated); the script prints these counts and stops only if bare Ensembl IDs
# are more than half of the genes or duplicates more than 10%.  SIGnature's data.harmonize() makes var_names unique
# before aligning to the SCimilarity gene panel.
suppressMessages({
  library(SingleCellExperiment)
  library(reticulate)
})
# Bare `Rscript /path/to/envs/marker_methods/bin/Rscript ...` does NOT activate
# the marker_methods conda env's PATH, so reticulate's auto-discovery can silently
# fall back to provisioning its own bare ephemeral Python. Pin explicitly.
reticulate::use_python(
  "/path/to/envs/marker_methods/bin/python",
  required = TRUE
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("usage: Rscript export_h5ad.R <dataset_id>")
}
id <- args[[1]]

sce <- readRDS(here::here("data", "real_data_fullgene", paste0(id, ".rds")))
stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(sce),
  !is.null(colLabels(sce))
)

gene_ids <- rownames(sce)
n_bare_ensembl <- sum(grepl("^ENSG[0-9]+$", gene_ids))
n_hybrid <- sum(grepl("-ENSG[0-9]+$", gene_ids))
n_dup <- sum(duplicated(gene_ids))
frac_bare <- n_bare_ensembl / length(gene_ids)
frac_dup <- n_dup / length(gene_ids)
cat(sprintf(
  "[%s] gene id check: %d total | %d bare-Ensembl (%.2f%%) | %d symbol-ENSG hybrid | %d duplicated (%.2f%%)\n",
  id, length(gene_ids), n_bare_ensembl, 100 * frac_bare, n_hybrid, n_dup, 100 * frac_dup
))
if (frac_bare > 0.5) {
  stop(
    id, ": ", n_bare_ensembl, " / ", length(gene_ids), " (", round(100 * frac_bare, 1),
    "%) rownames(sce) are bare Ensembl IDs -- majority-Ensembl, which is the ",
    "failure mode export_h5ad_pbmc3k.R's original check guards against. ",
    "Investigate data/real_data_fullgene/", id, ".rds's provenance before proceeding; ",
    "do not silently remap to symbols."
  )
}
if (frac_dup > 0.10) {
  stop(
    id, ": ", n_dup, " / ", length(gene_ids), " (", round(100 * frac_dup, 1),
    "%) rownames(sce) are duplicated -- above the 10% sanity ceiling. ",
    "Investigate data/real_data_fullgene/", id, ".rds's provenance before proceeding."
  )
}

# --- Verify counts are non-negative, finite, raw/unnormalized-scale -------
# Counts assay class varies by dataset (dense base `matrix` for the smaller
# objects e.g. lawlor/ss3_pbmc, sparse dgCMatrix for the larger ones e.g.
# pbmc3k) -- @x only exists on the sparse S4 class, so branch explicitly
# rather than assuming sparsity.
#
# Integer-valuedness is NOT a hard requirement of the downstream SIGnature
# pipeline (data.harmonize()/lognorm_counts() only calls
# sc.pp.normalize_total + sc.pp.log1p, which require non-negative finite
# input, not integers; align_dataset() explicitly checks for negatives, never
# for integers). It IS diagnostic of provenance: pbmc3k's own export script
# hard-required integers because a violation there would mean logcounts/
# scaledata got exported by mistake. Confirmed by direct inspection (not
# assumed) that lawlor's "counts" assay is RPKM-scale, not integer read
# counts (range 0.01-1,453,667, floor value 0.01) -- and that this is NOT an
# artifact of this fullgene rebuild: data/real_data/lawlor.rds (the
# 2000-gene object every other method in this benchmark harness already runs
# on) has the identical range and floor value. This is a known, documented
# characteristic of the public Lawlor et al. dataset (scRNAseq::
# LawlorPancreasData() ships RPKM only, per its own GEO deposit -- no raw
# integer counts were ever released), consistent project-wide, not a bug to
# route around here. So: hard-require non-negative + finite (the pipeline's
# real requirements); report non-integer fraction as a diagnostic, and WARN
# (not fail) when it is non-trivial, so this is visible in the log rather
# than silently smoothed over.
count_mat <- counts(sce)
is_sparse <- methods::is(count_mat, "sparseMatrix")
count_vals <- if (is_sparse) count_mat@x else as.vector(count_mat)
stopifnot("counts assay must be non-negative (raw/unnormalized scale)" =
            all(count_vals >= 0))
stopifnot("counts assay must be finite" = all(is.finite(count_vals)))
frac_noninteger <- mean(count_vals %% 1 > 1e-6 & count_vals %% 1 < (1 - 1e-6))
cat(sprintf(
  "[%s] counts sanity OK (%s, class=%s, %d values checked, non-negative & finite; max = %s; frac non-integer = %.3f)\n",
  id, if (is_sparse) "sparse: nonzero entries only" else "dense: all entries",
  paste(class(count_mat), collapse = "/"), length(count_vals), max(count_vals), frac_noninteger
))
if (frac_noninteger > 0.01) {
  cat(sprintf(
    "[%s] NOTE: %.1f%% of counts values are non-integer -- this dataset's \"counts\" assay is not integer read/UMI counts (e.g. RPKM-scale for plate-based assays like Lawlor et al.). Non-negative & finite (verified above), which is all data.harmonize()/lognorm_counts() actually require; flagged here for the record, not treated as an error.\n",
    id, 100 * frac_noninteger
  ))
}

# --- Build AnnData: X = raw counts ONLY (cells x genes) --------------------
# obs/var are built as pandas DataFrames DIRECTLY (via reticulate, not by
# r_to_py()-converting an R data.frame) because R's data.frame() rejects
# duplicate row.names outright, and several of these datasets have a small
# number of duplicated gene symbols (see gene id check above; e.g. lawlor 10,
# ss3_pbmc 490, zhao 34) -- pandas tolerates a non-unique index natively, and
# downstream data.harmonize() (SC env) calls var_names_make_unique() before
# using var_names for anything, so a non-unique var index at this stage is
# safe to carry through. Cell barcodes (obs) are expected unique but built
# the same defensive way for consistency.
# pandas imported with convert=FALSE: reticulate's default (convert=TRUE)
# auto-converts a returned pandas.DataFrame back to a plain R data.frame,
# and R data.frames cannot hold duplicate row.names -- that round-trip
# silently collapsed a duplicate-indexed var_df to 0 rows here (confirmed by
# the "index contains duplicated values: row names not set" warning). Keeping
# pd's return values as raw Python object handles avoids the round-trip
# entirely; ad$AnnData() accepts them as-is.
sp <- import("scipy.sparse")
ad <- import("anndata")
pd <- import("pandas", convert = FALSE)
X <- sp$csr_matrix(r_to_py(Matrix::t(counts(sce))))  # cells x genes, RAW counts
obs_df <- pd$DataFrame(list(label = as.character(colLabels(sce))), index = colnames(sce))
var_df <- pd$DataFrame(index = gene_ids)
adata <- ad$AnnData(
  X = X,
  obs = obs_df,
  var = var_df
)
dir.create(here::here("atlases"), showWarnings = FALSE, recursive = TRUE)
out_path <- here::here("atlases", paste0(id, "_fullgene.h5ad"))
adata$write_h5ad(out_path)
cat(sprintf("[%s] wrote h5ad: %d cells x %d genes -> %s\n",
            id, adata$shape[[1]], adata$shape[[2]], out_path))
cat(sprintf("[%s] n unique labels: %d\n", id, length(unique(as.character(colLabels(sce))))))
cat(sprintf(
  "[%s] first 3 / last 3 obs_names: %s / %s\n",
  id,
  paste(head(colnames(sce), 3), collapse = ","),
  paste(tail(colnames(sce), 3), collapse = ",")
))
