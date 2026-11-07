# All-gene SingleCellExperiment of pbmc3k, for the attributions.
#
# Runs the marker-gene benchmark's code/prep-pbmc3k.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/pbmc3k.rds while all genes and the raw counts are kept.
#
# The raw data come from TENxPBMCData(dataset = "pbmc3k"), the call the benchmark's code/download-raw-data.R uses.
#
# Also kept from code/prep-pbmc3k.R: unique rownames, the Seurat QC thresholds, LogNormalize, FindVariableFeatures,
# ScaleData, RunPCA, FindNeighbors, FindClusters(resolution = 0.5), RunUMAP / RunTSNE, RenameIdents, the conversion
# to SingleCellExperiment and the labels.
#
# Changes from code/prep-pbmc3k.R: the raw data are read as described above; the output path is fixed (no
# command-line arguments).

.libPaths()

suppressMessages({
  library(TENxPBMCData)
  library(scater)
  library(Seurat)
  library(scran)
})

set.seed(2572021)

# Same source as code/download-raw-data.R's:
#   raw_pbmc3k <- TENxPBMCData(dataset = "pbmc3k")
#   saveRDS(raw_pbmc3k, here::here("data", "raw_data", "pbmc3k.rds"))
# The raw data are pulled directly instead of
# reading args$raw_data.
pbmc3k <- TENxPBMCData(dataset = "pbmc3k")

rownames(pbmc3k) <- uniquifyFeatureNames(
  rowData(pbmc3k)$ENSEMBL_ID,
  rowData(pbmc3k)$Symbol_TENx
)

old_rowdata <- rowData(pbmc3k)

counts <- as(counts(pbmc3k), "dgCMatrix")
colnames(counts) <- paste0("Cell", 1:ncol(counts))

pbmc3k <- CreateSeuratObject(
  counts = counts,
  project = "pbmc3k",
  min.cells = 3,
  min.features = 200
)

pbmc3k[["percent.mt"]] <- PercentageFeatureSet(pbmc3k, pattern = "^MT-")

pbmc3k <- subset(
  pbmc3k,
  subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5
)

pbmc3k <- NormalizeData(
  pbmc3k,
  normalization.method = "LogNormalize",
  scale.factor = 10000
)

pbmc3k <- FindVariableFeatures(
  pbmc3k,
  selection.method = "vst",
  nfeatures = 2000
)

all_genes <- rownames(pbmc3k)
pbmc3k <- ScaleData(pbmc3k, features = all_genes)

pbmc3k <- RunPCA(pbmc3k, features = VariableFeatures(object = pbmc3k))

pbmc3k <- FindNeighbors(pbmc3k, dims = 1:10)
pbmc3k <- FindClusters(pbmc3k, resolution = 0.5)

pbmc3k <- RunUMAP(pbmc3k, dims = 1:10)
pbmc3k <- RunTSNE(pbmc3k, dims = 1:10)

new_cluster_ids <- c(
  "Naive CD4 T",
  "CD14+ Mono",
  "Memory CD4 T",
  "B",
  "CD8 T",
  "FCGR3A+ Mono",
  "NK",
  "DC",
  "Platelet"
)
names(new_cluster_ids) <- levels(pbmc3k)
pbmc3k <- RenameIdents(pbmc3k, new_cluster_ids)

# Convert back to a SingleCellExperiment
pbmc3k <- as.SingleCellExperiment(pbmc3k)
colLabels(pbmc3k) <- colData(pbmc3k)$ident

# Add back some rowData to prevent downstream (Python) errors.
rowData(pbmc3k) <- rownames(pbmc3k)

# --- stop before the gene subset ------------------------------------------
# Save HERE: immediately before the original script's final gene-subset line
#   pbmc3k <- pbmc3k[union(top_pbmc3k, pbmc3k_expert_mgs), ]
# which (along with everything after it -- computeSumFactors, per-cluster
# subsets, the final saveRDS(pbmc3k, args$data)) is intentionally NOT
# reproduced. Everything above this comment mirrors code/prep-pbmc3k.R.

if (!is(pbmc3k, "SingleCellExperiment")) {
  pbmc3k <- as.SingleCellExperiment(pbmc3k)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(pbmc3k),
  "logcounts" %in% SummarizedExperiment::assayNames(pbmc3k),
  !is.null(colLabels(pbmc3k))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(pbmc3k, here::here("data", "real_data_fullgene", "pbmc3k.rds"))

cat("full-gene SCE:", nrow(pbmc3k), "genes x", ncol(pbmc3k), "cells; assays:",
    paste(SummarizedExperiment::assayNames(pbmc3k), collapse = ","), "\n")
