# All-gene SingleCellExperiment of the Smart-seq3 PBMCs of Hagemann-Jensen et al., for the attributions.
#
# Runs the marker-gene benchmark's code/prep-ss3_pbmc.R up to, and not including, its final gene subset, so the cells,
# their order and their labels are those of data/real_data/ss3_pbmc.rds while all genes and the raw counts are kept.
#
# The raw data are built as in the benchmark's code/download-raw-data.R, from two files of ArrayExpress E-MTAB-8735
# (HCA.UMIcounts.PBMC.txt, Smart3.PBMC.annotated.txt), downloaded into data/downloaded_data/.  The header line of
# HCA.UMIcounts.PBMC.txt has no name for the gene column (3,129 fields against 3,130 in the data rows); as the harness
# did by hand, "genes\t" is prepended to it, in a copy of the file.
#
# Also kept from code/prep-ss3_pbmc.R: EnsDb.Hsapiens.v86 SYMBOL mapping with rownames for missing symbols,
# quickCluster / computeSumFactors / logNormCounts.
#
# Changes from code/prep-ss3_pbmc.R: the raw data are read as described above; the output path is fixed (no command-line
# arguments); colLabels() is set before the gene subset (it only reads a colData column, so the labels are the same);
# runPCA / runTSNE / runUMAP are not run.

suppressMessages({
  library(scRNAseq)
  library(scran)
  library(scater)
  library(AnnotationHub)
  library(AnnotationDbi)
  library(EnsDb.Hsapiens.v86)
  library(readr)
})

set.seed("21122021")

dir.create(here::here("data", "downloaded_data"), showWarnings = FALSE, recursive = TRUE)

umi_path <- here::here("data", "downloaded_data", "HCA.UMIcounts.PBMC.txt")
anno_path <- here::here("data", "downloaded_data", "Smart3.PBMC.annotated.txt")

if (!file.exists(umi_path)) {
  download.file(
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/735/E-MTAB-8735/Files/HCA.UMIcounts.PBMC.txt",
    umi_path, quiet = TRUE
  )
}
if (!file.exists(anno_path)) {
  download.file(
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/735/E-MTAB-8735/Files/Smart3.PBMC.annotated.txt",
    anno_path, quiet = TRUE
  )
}

# --- reproduce the documented manual header fix, without mutating the ----
# --- pristine download ----------------------------------------------------
umi_fixed_path <- here::here("data", "downloaded_data", "HCA.UMIcounts.PBMC.fixed.txt")
if (!file.exists(umi_fixed_path)) {
  con_in <- file(umi_path, "r")
  first_line <- readLines(con_in, n = 1)
  rest <- readLines(con_in)
  close(con_in)
  con_out <- file(umi_fixed_path, "w")
  writeLines(paste0("genes\t", first_line), con_out)
  writeLines(rest, con_out)
  close(con_out)
}

# --- the Smart-seq3 PBMC block of the benchmark's code/download-raw-data.R (paths changed) --

# (the harness's note: the string "genes  " was added by hand to the first line of
# HCA.UMIcounts.PBMC.txt so that the file parses; done above in a copy)
umi_counts <- read_table(
  umi_fixed_path,
)

pbmc_anno <- read_tsv(
  anno_path,
  show_col_types = FALSE
)

counts_mat <- as.matrix(umi_counts[, -1])
rownames(counts_mat) <- umi_counts[, 1][[1]]

ss3_pbmc <- SingleCellExperiment(assays = list(counts = counts_mat))
colData(ss3_pbmc) <- DataFrame(pbmc_anno)
colnames(ss3_pbmc) <- pbmc_anno$cellID

# --- code/prep-ss3_pbmc.R, up to (not including) the gene-subset line ----

anno <- select(
  EnsDb.Hsapiens.v86,
  keys = rownames(ss3_pbmc),
  keytype = "GENEID",
  columns = c("SYMBOL", "SEQNAME")
)
rowData(ss3_pbmc) <- anno[match(rownames(ss3_pbmc), anno[, 1]), -1]

gene_name <- rowData(ss3_pbmc)$SYMBOL
no_symbol_ind <- which(is.na(gene_name))
gene_name[no_symbol_ind] <- rownames(ss3_pbmc)[no_symbol_ind]
rownames(ss3_pbmc) <- gene_name

clusters <- quickCluster(ss3_pbmc)
ss3_pbmc <- computeSumFactors(ss3_pbmc, clusters = clusters)
ss3_pbmc <- logNormCounts(ss3_pbmc)

# --- stop before the gene subset ------------------------------------------
# Hoisted from after the gene-subset line (see deviation #4 above): this is
# an untouched colData column, so the result is identical regardless of when
# it's assigned.
colLabels(ss3_pbmc) <- factor(ss3_pbmc$clusterName)

# Save HERE: immediately before the original script's final gene-subset line
#   ss3_pbmc <- ss3_pbmc[union(top_ss3_pbmc, ss3_pbmc_expert_mgs), ]
# which (along with everything after it) is intentionally NOT reproduced.

if (!is(ss3_pbmc, "SingleCellExperiment")) {
  ss3_pbmc <- as.SingleCellExperiment(ss3_pbmc)
}

stopifnot(
  "counts" %in% SummarizedExperiment::assayNames(ss3_pbmc),
  "logcounts" %in% SummarizedExperiment::assayNames(ss3_pbmc),
  !is.null(colLabels(ss3_pbmc))
)

dir.create(here::here("data", "real_data_fullgene"),
           showWarnings = FALSE, recursive = TRUE)
saveRDS(ss3_pbmc, here::here("data", "real_data_fullgene", "ss3_pbmc.rds"))

cat("full-gene SCE:", nrow(ss3_pbmc), "genes x", ncol(ss3_pbmc), "cells; assays:",
    paste(SummarizedExperiment::assayNames(ss3_pbmc), collapse = ","), "\n")
