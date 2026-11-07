#!/usr/bin/env Rscript
# SingleCellExperiment .rds files of the cell-cycle comparisons, from the exports in
# marker_methods/data/cc_export (cell_cycle/export_*_contrasts.py).  One object per comparison, labels "target" and
# "reference".  As in build_lineage_sce.R, the counts are checked to be integer.
#
# On the CPU server (marker_methods R environment):
#   Rscript build_cell_cycle_sce.R $(cat /path/to/cpu_server/analysis/marker_methods/data/cc_export/CC_IDS.txt)
suppressMessages({library(Matrix); library(SingleCellExperiment); library(scater)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) > 0)
stg <- "/path/to/cpu_server/analysis/marker_methods/data/cc_export"
outdir <- "/path/to/cpu_server/analysis/marker_methods/data/real_data"
for (id in args) {
  out <- file.path(outdir, paste0(id, "_cc.rds"))
  if (file.exists(out)) { cat("[", id, "] exists, skip\n"); next }
  cat("[", id, "] reading mtx\n")
  m <- readMM(gzfile(file.path(stg, paste0(id, "_counts.mtx.gz"))))
  m <- as(m, "CsparseMatrix")
  genes <- readLines(file.path(stg, paste0(id, "_genes.txt")))
  labels <- readLines(file.path(stg, paste0(id, "_labels.txt")))
  stopifnot(nrow(m) == length(genes), ncol(m) == length(labels))
  stopifnot(all(m@x == round(m@x)))
  if (max(m@x) < 50) stop(sprintf("%s: max count %g -- looks like truncated lognorm, not counts", id, max(m@x)))
  stopifnot(setequal(unique(labels), c("target", "reference")))
  rownames(m) <- make.unique(genes); colnames(m) <- paste0("c", seq_len(ncol(m)))
  # "reference" first so it is the factor baseline; the battery only needs the two levels to exist.
  lab <- factor(as.character(labels), levels = c("reference", "target"))
  sce <- SingleCellExperiment(assays = list(counts = m), colData = DataFrame(label = lab))
  colLabels(sce) <- lab
  sce <- logNormCounts(sce)
  saveRDS(sce, out)
  cat(sprintf("[ %s ] wrote %d genes x %d cells, target=%d reference=%d, max=%g, median_total=%g\n",
              id, nrow(sce), ncol(sce), sum(lab == "target"), sum(lab == "reference"),
              max(m@x), median(colSums(m))))
}
cat("[cc sce build done]\n")
