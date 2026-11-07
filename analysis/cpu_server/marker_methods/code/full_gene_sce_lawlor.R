# All-gene SingleCellExperiment of the Lawlor et al. pancreatic islets, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-lawlor.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/lawlor.rds while all genes and the raw counts are kept.
#
# The raw data come from LawlorPancreasData(), the call the benchmark's code/download-raw-data.R uses.
#
# Also kept from code/prep-lawlor.R: EnsDb.Hsapiens.v86 SYMBOL / SEQNAME mapping with rownames for missing symbols,
# perCellQCMetrics + quickPerCellQC (batched by `islet unos id`), quickCluster / computeSumFactors / logNormCounts.
#
# Changes from code/prep-lawlor.R: the raw data are read as described above; the output path is fixed (no command-line
# arguments); colLabels() is set before the gene subset (it only reads a colData column, so the labels are the same);
# runPCA / runTSNE / runUMAP are not run.

suppressMessages({
  library(scRNAseq)
  library(AnnotationHub)
  library(AnnotationDbi)
  library(EnsDb.Hsapiens.v86)
  library(scater)
  library(scran)
})

set.seed(24072021)

# Same source as code/download-raw-data.R's:
#   raw_lawlor <- LawlorPancreasData()
#   saveRDS(raw_lawlor, here::here("data", "raw_data", "lawlor.rds"))
# The raw data are pulled directly instead of
# reading args$raw_data.
lawlor <- LawlorPancreasData()

anno <- select(
  EnsDb.Hsapiens.v86,
  keys = rownames(lawlor),
  keytype = "GENEID",
  columns = c("SYMBOL", "SEQNAME")
)
rowData(lawlor) <- anno[match(rownames(lawlor), anno[, 1]), -1]

old_rownames <- rownames(lawlor)
rownames(lawlor) <- rowData(lawlor)$SYMBOL
is_na_rownames <- which(is.na(rownames(lawlor)))
rownames(lawlor)[is_na_rownames] <- old_rownames[is_na_rownames]

stats <- perCellQCMetrics(
  lawlor,
  subsets = list(Mito = which(rowData(lawlor)$SEQNAME == "MT"))
)

qc <- quickPerCellQC(
  stats,
  percent_subsets = "subsets_Mito_percent",
  batch = lawlor$`islet unos id`
)

lawlor <- lawlor[, !qc$discard]

clusters <- quickCluster(lawlor)
lawlor <- computeSumFactors(lawlor, clusters = clusters)
lawlor <- logNormCounts(lawlor)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #3 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colData(lawlor)$`cell type` <- factor(colData(lawlor)$`cell type`)
colLabels(lawlor) <- colData(lawlor)$`cell type`

# Save HERE: immediately before the original script's final gene-subset line
#   lawlor <- lawlor[union(top_lawlor, lawlor_expert_mgs), ]
# which (along with everything after it) is intentionally NOT reproduced.
# Everything above mirrors code/prep-lawlor.R (see deviations above).

if (!is(lawlor, "SingleCellExperiment")) {
  lawlor <- as.SingleCellExperiment(lawlor)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(lawlor),
  "logcounts" %in% SummarizedExperiment::assayNames(lawlor),
  !is.null(colLabels(lawlor))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(lawlor, here::here("data", "real_data_fullgene", "lawlor.rds"))

cat("full-gene SCE:", nrow(lawlor), "genes x", ncol(lawlor), "cells; assays:",
    paste(SummarizedExperiment::assayNames(lawlor), collapse = ","), "\n")
