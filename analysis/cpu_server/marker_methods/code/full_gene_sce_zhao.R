# All-gene SingleCellExperiment of the Zhao et al. liver immune cells, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-zhao.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/zhao.rds while all genes and the raw counts are kept.
#
# The raw data come from ZhaoImmuneLiverData(), the call the benchmark's code/download-raw-data.R uses.
#
# Also kept from code/prep-zhao.R: dropping cells without a `fine` annotation, rownames from rowData()$Symbol,
# quickCluster / computeSumFactors / logNormCounts.
#
# Changes from code/prep-zhao.R: the raw data are read as described above; the output path is fixed (no command-line
# arguments); colLabels() is set before the gene subset (it only reads a colData column, so the labels are the same);
# runPCA / runTSNE / runUMAP are not run.

suppressMessages({
  library(scRNAseq)
  library(scater)
  library(scran)
})

set.seed(07122021)

# Same source as code/download-raw-data.R's:
#   raw_zhao <- ZhaoImmuneLiverData()
#   saveRDS(raw_zhao, here::here("data", "raw_data", "zhao.rds"))
zhao <- ZhaoImmuneLiverData()

# Remove cells without annotations.
zhao <- zhao[, !is.na(zhao$fine)]

rownames(zhao) <- rowData(zhao)$Symbol

clusters <- quickCluster(zhao)
zhao <- computeSumFactors(zhao, clusters = clusters)
zhao <- logNormCounts(zhao)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #3 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colLabels(zhao) <- factor(zhao$fine)

# Save HERE: immediately before the original script's final gene-subset line
#   zhao <- zhao[top_zhao, ]
# which (along with everything after it) is intentionally NOT reproduced.
# Everything above mirrors code/prep-zhao.R (see deviations above).

if (!is(zhao, "SingleCellExperiment")) {
  zhao <- as.SingleCellExperiment(zhao)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(zhao),
  "logcounts" %in% SummarizedExperiment::assayNames(zhao),
  !is.null(colLabels(zhao))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(zhao, here::here("data", "real_data_fullgene", "zhao.rds"))

cat("full-gene SCE:", nrow(zhao), "genes x", ncol(zhao), "cells; assays:",
    paste(SummarizedExperiment::assayNames(zhao), collapse = ","), "\n")
