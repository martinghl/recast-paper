#!/usr/bin/env Rscript
# Top-200 gene lists of the comparator methods for each phase-defined cell-cycle comparison, one CSV
# per comparison.  Each comparison has two clusters ("target", "reference"); only the target list is kept.
#
# CPU server (marker_methods R).  Usage: Rscript export_comparator_genes_phase.R
# marker_methods_grid/data is a link to ../marker_methods/data, but marker_methods_grid/results is its own folder
# (Snakemake runs in marker_methods_grid), so the results are read from there; the script stops if it finds none.
RD <- "/path/to/cpu_server/analysis/marker_methods_grid/results/real_data"
OUT <- "/path/to/cpu_server/analysis/cell_cycle/phase_genesets"
IDS <- readLines("/path/to/cpu_server/analysis/marker_methods/data/cc_export/CC_FAIR_IDS.txt")
TOPN <- 200
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Fail on a wrong/empty results root rather than reporting it as twelve empty datasets.
if (!dir.exists(RD)) stop("results dir does not exist: ", RD)
n_all <- length(Sys.glob(file.path(RD, "*_cc-*.rds")))
if (n_all == 0) stop("no *_cc-*.rds under ", RD, " -- wrong results root, or the battery wrote nothing")
cat(sprintf("[cc_extract] %d result files under %s\n", n_all, RD))

for (id in IDS) {
  pref <- paste0(id, "_cc-")
  files <- Sys.glob(file.path(RD, paste0(pref, "*.rds")))
  rows <- list()
  for (f in files) {
    meth <- sub("\\.rds$", "", sub(paste0("^", pref), "", basename(f)))
    x <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(x)) next
    res <- if (is.list(x) && !is.data.frame(x) && "result" %in% names(x)) x[["result"]] else x
    res <- as.data.frame(res)
    if (!all(c("cluster", "gene") %in% colnames(res))) next
    g <- head(as.character(res$gene[as.character(res$cluster) == "target"]), TOPN)
    if (length(g) == 0) next
    rows[[length(rows) + 1]] <- data.frame(method = meth, rank = seq_along(g), gene = g,
                                           stringsAsFactors = FALSE)
  }
  if (length(rows) == 0) { cat(sprintf("[%s] NO genesets\n", id)); next }
  df <- do.call(rbind, rows)
  write.csv(df, file.path(OUT, paste0(id, ".csv")), row.names = FALSE, quote = TRUE)
  cat(sprintf("[%s] %d methods, %d rows\n", id, length(unique(df$method)), nrow(df)))
}
cat("CC_EXTRACT_FAIR_DONE\n")
