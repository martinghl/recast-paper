# All-gene SingleCellExperiment of Yoshida et al. 2021, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-yoshida.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/yoshida.rds while all genes and the raw counts are kept.
#
# The raw data are the study's part of the CellTypist "Blood organ atlas" (Xu et al. 2023;
# https://cog.sanger.ac.uk/celltypist/Resources/Organ_atlas/Blood/Blood.h5ad), whose obs columns and Curated_annotation
# levels match data/real_data/yoshida.rds and whose .raw.X holds the raw integer counts (.X is normalized).
# export_blood_dataset.py writes the study's counts and metadata as Matrix-Market / CSV files, read back here with
# Matrix::readMM().
#
# Also kept from code/prep-yoshida.R: removal of clusters with < 200 cells, quickCluster / computeSumFactors /
# logNormCounts.
#
# Changes from code/prep-yoshida.R: the raw data are read as described above; the output path is fixed (no command-line
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

prefix <- here::here("data", "downloaded_data", "blood_raw_yoshida")
counts_path <- paste0(prefix, "_counts.mtx")
if (!file.exists(counts_path)) {
  status <- system2(
    "/path/to/envs/marker_methods/bin/python3",
    c(shQuote(here::here("code", "export_blood_dataset.py")),
      shQuote("Yoshida et al. 2021"),
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

yoshida <- SingleCellExperiment(
  assays = list(counts = counts),
  colData = DataFrame(obs, row.names = barcodes)
)

# --- code/prep-yoshida.R, from args$raw_data, up to (not including) the --
# --- gene-subset line -----------------------------------------------------

small_clusters <- names(which(table(yoshida$Curated_annotation) < 200))
yoshida <- yoshida[, !(yoshida$Curated_annotation %in% small_clusters)]

clusters <- quickCluster(yoshida)
yoshida <- computeSumFactors(yoshida, clusters = clusters)
yoshida <- logNormCounts(yoshida)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #3 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colLabels(yoshida) <- factor(yoshida$Curated_annotation)

# Save HERE: immediately before the original script's final gene-subset line
#   yoshida <- yoshida[union(top_yoshida, yoshida_expert_mgs), ]
# which (along with everything after it) is intentionally NOT reproduced.

if (!is(yoshida, "SingleCellExperiment")) {
  yoshida <- as.SingleCellExperiment(yoshida)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(yoshida),
  "logcounts" %in% SummarizedExperiment::assayNames(yoshida),
  !is.null(colLabels(yoshida))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(yoshida, here::here("data", "real_data_fullgene", "yoshida.rds"))

cat("full-gene SCE:", nrow(yoshida), "genes x", ncol(yoshida), "cells; assays:",
    paste(SummarizedExperiment::assayNames(yoshida), collapse = ","), "\n")
