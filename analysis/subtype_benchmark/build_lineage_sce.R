#!/usr/bin/env Rscript
# Build the SingleCellExperiment .rds files of the subtype lineages from the integer-count export
# (export_lineage_counts.py: sce_export_v2 -> fine_state_export_v2), as build_extra_sce.R does.  Writes
# "<id>_v2.rds".  Run on the CPU server in the marker_methods env.
suppressMessages({library(Matrix); library(SingleCellExperiment); library(scater)})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) > 0)
stg <- "/path/to/cpu_server/analysis/marker_methods/data/fine_state_export_v2"
outdir <- "/path/to/cpu_server/analysis/marker_methods/data/real_data"
for (id in args) {
  out <- file.path(outdir, paste0(id, "_v2.rds"))
  if (file.exists(out)) { cat("[", id, "] exists, skip\n"); next }
  cat("[", id, "] reading mtx\n")
  m <- readMM(gzfile(file.path(stg, paste0(id, "_counts.mtx.gz"))))
  m <- as(m, "CsparseMatrix")
  genes <- readLines(file.path(stg, paste0(id, "_genes.txt")))
  labels <- readLines(file.path(stg, paste0(id, "_labels.txt")))
  stopifnot(nrow(m) == length(genes), ncol(m) == length(labels))
  # guard against ever shipping a truncated-lognorm matrix again
  stopifnot(all(m@x == round(m@x)))
  if (max(m@x) < 50) stop(sprintf("%s: max count %g -- looks like truncated lognorm, not counts", id, max(m@x)))
  rownames(m) <- make.unique(genes); colnames(m) <- paste0("c", seq_len(ncol(m)))
  lab <- factor(as.character(labels))
  sce <- SingleCellExperiment(assays = list(counts = m), colData = DataFrame(label = lab))
  colLabels(sce) <- lab
  sce <- logNormCounts(sce)
  saveRDS(sce, out)
  cat(sprintf("[ %s ] wrote %d genes x %d cells, %d states, max=%g, median_total=%g\n",
              id, nrow(sce), ncol(sce), nlevels(lab), max(m@x), median(colSums(m))))
}
cat("[sce build done]\n")
