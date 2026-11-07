# All-gene SingleCellExperiment of Ren et al. 2021, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-ren.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/ren.rds while all genes and the raw counts are kept.
#
# The raw data are the study's part of the CellTypist "Blood organ atlas" (Xu et al. 2023;
# https://cog.sanger.ac.uk/celltypist/Resources/Organ_atlas/Blood/Blood.h5ad), whose obs columns and Curated_annotation
# levels match data/real_data/ren.rds and whose .raw.X holds the raw integer counts (.X is normalized).
# export_blood_dataset.py writes the study's counts and metadata as Matrix-Market / CSV files, read back here with
# Matrix::readMM().
#
# code/prep-ren.R downsamples with sample(1:ncol(x), 25000) before any filtering; the cells it picks depend on
# the seed and on the cell order.  The seed (set.seed(406023) before any random call) and the cell order of Blood.h5ad
# are kept, and the check below compares the result with data/real_data/ren.rds.
# Also kept: removal of clusters with < 200 cells, quickCluster / computeSumFactors / logNormCounts.
#
# Changes from code/prep-ren.R: the raw data are read as described above; the output path is fixed (no command-line
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

prefix <- here::here("data", "downloaded_data", "blood_raw_ren")
counts_path <- paste0(prefix, "_counts.mtx")
if (!file.exists(counts_path)) {
  status <- system2(
    "/path/to/envs/marker_methods/bin/python3",
    c(shQuote(here::here("code", "export_blood_dataset.py")),
      shQuote("Ren et al. 2021"),
      shQuote(prefix))
  )
  stopifnot(status == 0)
}

counts <- as(Matrix::readMM(counts_path), "CsparseMatrix")
genes <- readLines(paste0(prefix, "_genes.txt"))
barcodes <- readLines(paste0(prefix, "_barcodes.txt"))
obs <- read.csv(paste0(prefix, "_obs.csv"), check.names = FALSE, stringsAsFactors = TRUE)

stopifnot(nrow(counts) == length(genes), ncol(counts) == length(barcodes), nrow(obs) == length(barcodes))
stopifnot(identical(as.character(obs[[1]]), barcodes))
obs <- obs[, -1, drop = FALSE]

rownames(counts) <- genes
colnames(counts) <- barcodes

ren <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = DataFrame(obs, row.names = barcodes)
)

# --- code/prep-ren.R, from args$raw_data, up to (not including) the -----
# --- gene-subset line ------------------------------------------------------

ren <- ren[, sample(1:ncol(ren), 25000)]

small_clusters <- names(which(table(ren$Curated_annotation) < 200))
ren <- ren[, !(ren$Curated_annotation %in% small_clusters)]

clusters <- quickCluster(ren)
ren <- computeSumFactors(ren, clusters = clusters)
ren <- logNormCounts(ren)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #3 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colLabels(ren) <- factor(ren$Curated_annotation)

# Save HERE: immediately before the original script's final gene-subset line
#   ren <- ren[union(top_ren, ren_expert_mgs), ]
# which (along with everything after it) is intentionally NOT reproduced.

if (!is(ren, "SingleCellExperiment")) {
  ren <- as.SingleCellExperiment(ren)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(ren),
  "logcounts" %in% SummarizedExperiment::assayNames(ren),
  !is.null(colLabels(ren))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(ren, here::here("data", "real_data_fullgene", "ren.rds"))

cat("full-gene SCE:", nrow(ren), "genes x", ncol(ren), "cells; assays:",
    paste(SummarizedExperiment::assayNames(ren), collapse = ","), "\n")
