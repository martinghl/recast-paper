#!/usr/bin/env Rscript
# Three benchmark SingleCellExperiments of the Hao PBMC subset (pp_pbmc_b_mono_nk_scored.h5ad) with the
# same expression and three label levels (celltype.l1 / .l2 / .l3).  layers/counts holds integer counts (min 1, max 607),
# so the usual preparation applies: genes detected in >= 3 cells, scran normalization, 2,000 variable genes with
# modelGeneVarByPoisson.
suppressPackageStartupMessages({
  library(rhdf5); library(Matrix); library(SingleCellExperiment); library(scran); library(scuttle)
})
H5 <- "/path/to/cpu_server/analysis/pp_pbmc_b_mono_nk_scored.h5ad"
OUT <- "/path/to/cpu_server/analysis/marker_methods"
msg <- function(...) cat(sprintf(...), file = stderr())
rd_csr <- function(path, n_obs, n_var) {
  sparseMatrix(j = as.integer(h5read(H5, paste0(path, "/indices"))) + 1L,
               p = as.integer(h5read(H5, paste0(path, "/indptr"))),
               x = as.numeric(h5read(H5, paste0(path, "/data"))),
               dims = c(n_obs, n_var), index1 = TRUE)
}
cat_col <- function(k) {
  cats <- as.character(h5read(H5, paste0("obs/", k, "/categories")))
  cod  <- as.integer(h5read(H5, paste0("obs/", k, "/codes")))
  factor(ifelse(cod >= 0, cats[cod + 1L], NA), levels = cats)
}
genes <- as.character(h5read(H5, "var/_index"))
cells <- as.character(h5read(H5, "obs/_index"))
msg("h5ad: %d cells x %d genes\n", length(cells), length(genes))
cts <- t(rd_csr("layers/counts", length(cells), length(genes)))
rownames(cts) <- genes; colnames(cts) <- cells
stopifnot("counts are not integral" = all(cts@x == round(cts@x)), !anyDuplicated(rownames(cts)))
keep <- Matrix::rowSums(cts > 0) >= 3
msg("genes detected in >=3 cells: %d / %d\n", sum(keep), nrow(cts))
sce0 <- SingleCellExperiment(list(counts = cts[keep, , drop = FALSE]))
set.seed(20220502)
sce0 <- computeSumFactors(sce0, clusters = quickCluster(sce0))
sce0 <- logNormCounts(sce0)
hvg <- getTopHVGs(modelGeneVarByPoisson(sce0), n = 2000)
sce0 <- sce0[hvg, ]
msg("gene space after HVG-2000: %d\n", nrow(sce0))
for (lv in c("l1","l2","l3")) {
  lab <- cat_col(paste0("celltype.", lv))
  stopifnot(!anyNA(lab), length(lab) == ncol(sce0))
  sce <- sce0; colLabels(sce) <- droplevels(lab)
  id <- paste0("pbmcbmn_", lv)
  saveRDS(sce, file.path(OUT, "data/real_data", paste0(id, ".rds")))
  writeLines(rownames(sce), file.path(OUT, "atlases", paste0("bench_genes_", id, ".txt")))
  writeLines(colnames(sce), file.path(OUT, "atlases", paste0("bench_cells_", id, ".txt")))
  msg("[%s] %d genes x %d cells, %d classes (min %d)\n", id, nrow(sce), ncol(sce),
      nlevels(colLabels(sce)), min(table(colLabels(sce))))
}
