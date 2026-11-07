# All-gene SingleCellExperiment of Dominguez Conde et al. 2022, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-dominguez_conde.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/dominguez_conde.rds while all genes and the raw counts are kept.
#
# The raw data are the study's part of the CellTypist "Blood organ atlas" (Xu et al. 2023;
# https://cog.sanger.ac.uk/celltypist/Resources/Organ_atlas/Blood/Blood.h5ad), whose obs columns and Curated_annotation
# levels match data/real_data/dominguez_conde.rds and whose .raw.X holds the raw integer counts (.X is normalized).
# export_blood_dataset.py writes the study's counts and metadata as Matrix-Market / CSV files, read back here with
# Matrix::readMM().
#
# Also kept from code/prep-dominguez_conde.R: removal of clusters with < 200 cells, quickCluster / computeSumFactors /
# logNormCounts.
#
# Changes from code/prep-dominguez_conde.R: the raw data are read as described above; the output path is fixed (no command-line
# arguments); colLabels() is set before the gene subset (it only reads a colData column, so the labels are the same);
# runPCA / runTSNE / runUMAP are not run.

suppressMessages({
  library(scran)
  library(scater)
  library(Matrix)
  library(SingleCellExperiment)
})

set.seed(406023)

dir.create(here::here("data", "downloaded_data"), showWarnings = FALSE, recursive = TRUE)

prefix <- here::here("data", "downloaded_data", "blood_raw_dominguez_conde")
counts_path <- paste0(prefix, "_counts.mtx")
if (!file.exists(counts_path)) {
  status <- system2(
    "/path/to/envs/marker_methods/bin/python3",
    c(shQuote(here::here("code", "export_blood_dataset.py")),
      shQuote("Dominguez Conde et al. 2022"),
      shQuote(prefix))
  )
  stopifnot(status == 0)
}

counts <- as(Matrix::readMM(counts_path), "CsparseMatrix")
genes <- readLines(paste0(prefix, "_genes.txt"))
barcodes <- readLines(paste0(prefix, "_barcodes.txt"))
obs <- read.csv(paste0(prefix, "_obs.csv"), check.names = FALSE, stringsAsFactors = TRUE)

stopifnot(nrow(counts) == length(genes), ncol(counts) == length(barcodes), nrow(obs) == length(barcodes))
# obs.csv's first column is the exported pandas index (cell barcode); it and
# barcodes.txt are both derived from the same in-memory object/order in
# code/export_blood_dataset.py, so they must match exactly.
stopifnot(identical(as.character(obs[[1]]), barcodes))
obs <- obs[, -1, drop = FALSE]

rownames(counts) <- genes
colnames(counts) <- barcodes

dominguez_conde <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = DataFrame(obs, row.names = barcodes)
)

# --- code/prep-dominguez_conde.R, from args$raw_data, up to (not --------
# --- including) the gene-subset line -------------------------------------

small_clusters <- names(which(table(dominguez_conde$Curated_annotation) < 200))
dominguez_conde <- dominguez_conde[, !(dominguez_conde$Curated_annotation %in% small_clusters)]

clusters <- quickCluster(dominguez_conde)
dominguez_conde <- computeSumFactors(dominguez_conde, clusters = clusters)
dominguez_conde <- logNormCounts(dominguez_conde)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #3 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colLabels(dominguez_conde) <- factor(dominguez_conde$Curated_annotation)

# Save HERE: immediately before the original script's final gene-subset line
#   dominguez_conde <- dominguez_conde[union(top_dominguez_conde, dominguez_conde_expert_mgs), ]
# which (along with everything after it) is intentionally NOT reproduced.

if (!is(dominguez_conde, "SingleCellExperiment")) {
  dominguez_conde <- as.SingleCellExperiment(dominguez_conde)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(dominguez_conde),
  "logcounts" %in% SummarizedExperiment::assayNames(dominguez_conde),
  !is.null(colLabels(dominguez_conde))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(dominguez_conde, here::here("data", "real_data_fullgene", "dominguez_conde.rds"))

cat("full-gene SCE:", nrow(dominguez_conde), "genes x", ncol(dominguez_conde), "cells; assays:",
    paste(SummarizedExperiment::assayNames(dominguez_conde), collapse = ","), "\n")
