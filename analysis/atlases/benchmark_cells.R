# The cells and genes of each of the nine atlases as the marker-gene benchmark hands them to every method
# (data/real_data/<id>.rds): rows = colnames, columns = rownames.  data/real_data_fullgene/<id>.rds (all genes, used
# for the attributions) has the same cells in the same order; the Python side checks this against these lists.
#
# Reads the .rds files directly (no Snakemake).
#
# Usage: Rscript benchmark_cells.R
suppressMessages(library(SingleCellExperiment))

setwd("/path/to/cpu_server/analysis/marker_methods")
OUT_DIR <- "atlases"

ids <- c("pbmc3k", "lawlor", "ss3_pbmc", "citeseq", "ren",
         "stephenson", "dominguez_conde", "yoshida", "zhao")

summary_rows <- list()

for (id in ids) {
  path <- file.path("data", "real_data", paste0(id, ".rds"))
  cat(sprintf("\n=== %s (%s) ===\n", id, path))
  sce <- readRDS(path)
  cat(sprintf("  class: %s\n", paste(class(sce), collapse = "/")))
  stopifnot(is(sce, "SingleCellExperiment"))

  genes <- rownames(sce)
  cells <- colnames(sce)

  stopifnot("rownames(sce) is NULL -- no gene identity to extract" = !is.null(genes))
  stopifnot("colnames(sce) is NULL -- no cell barcode identity to extract" = !is.null(cells))

  n_genes <- length(genes)
  n_cells <- length(cells)
  n_dup_genes <- sum(duplicated(genes))
  n_dup_cells <- sum(duplicated(cells))

  labels <- tryCatch(as.character(colLabels(sce)), error = function(e) NA)
  has_labels <- !(length(labels) == 1 && is.na(labels))
  n_labels <- if (has_labels) length(unique(labels)) else NA

  cat(sprintf("  dim: %d genes x %d cells\n", n_genes, n_cells))
  cat(sprintf("  duplicated gene names: %d | duplicated cell barcodes: %d\n",
              n_dup_genes, n_dup_cells))
  cat(sprintf("  colLabels present: %s%s\n", has_labels,
              if (has_labels) sprintf(" (%d unique clusters)", n_labels) else ""))
  cat(sprintf("  first 3 genes: %s\n", paste(head(genes, 3), collapse = ", ")))
  cat(sprintf("  first 3 cells: %s\n", paste(head(cells, 3), collapse = ", ")))
  cat(sprintf("  last 3 cells:  %s\n", paste(tail(cells, 3), collapse = ", ")))

  writeLines(genes, file.path(OUT_DIR, sprintf("bench_genes_%s.txt", id)))
  writeLines(cells, file.path(OUT_DIR, sprintf("bench_cells_%s.txt", id)))
  if (has_labels) {
    writeLines(labels, file.path(OUT_DIR, sprintf("bench_labels_%s.txt", id)))
  } else {
    cat("  WARNING: no colLabels -- bench_labels file NOT written\n")
  }

  summary_rows[[id]] <- data.frame(
    id = id, n_genes = n_genes, n_cells = n_cells,
    n_dup_genes = n_dup_genes, n_dup_cells = n_dup_cells,
    has_labels = has_labels, n_clusters = n_labels
  )
  rm(sce); gc(verbose = FALSE)
}

cat("\n\n=== SUMMARY ===\n")
summary_df <- do.call(rbind, summary_rows)
print(summary_df, row.names = FALSE)
cat("\nDONE\n")
