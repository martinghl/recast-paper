#!/usr/bin/env Rscript
# Depth-50 gene lists of the comparator methods from the integer-count runs: reads
# marker_methods/results/real_data/<id>_v2-<method>.rds and writes fs_genesets_v2/<id>.csv (the "_v2" is dropped from
# the output name).
# CPU server, marker_methods R.  Usage: Rscript export_comparator_top50.R [id ...]   (default: all in V2_IDS.txt)
RD  <- "/path/to/cpu_server/analysis/marker_methods/results/real_data"
OUT <- "/path/to/cpu_server/analysis/subtype_benchmark/fs_genesets_v2"
TOPN <- 50
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
args <- commandArgs(trailingOnly = TRUE)
IDS <- if (length(args) > 0) args else
  readLines("/path/to/cpu_server/analysis/marker_methods/data/fine_state_export_v2/V2_IDS.txt")
IDS <- IDS[nzchar(IDS)]

for (id in IDS) {
  files <- Sys.glob(file.path(RD, paste0(id, "_v2-*.rds")))
  files <- files[!grepl("recast", files)]
  rows <- list()
  for (f in files) {
    meth <- sub("\\.rds$", "", sub(paste0("^", id, "_v2-"), "", basename(f)))
    x <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(x)) next
    res <- if (is.list(x) && !is.data.frame(x) && "result" %in% names(x)) x[["result"]] else x
    res <- as.data.frame(res)
    if (!all(c("cluster", "gene") %in% colnames(res))) next
    for (cl in unique(res$cluster)) {
      g <- head(as.character(res$gene[res$cluster == cl]), TOPN)
      if (length(g) == 0) next
      rows[[length(rows) + 1]] <- data.frame(method = meth, cluster = as.character(cl),
                                             rank = seq_along(g), gene = g, stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0) { cat(sprintf("[%s] NO genesets\n", id)); next }
  df <- do.call(rbind, rows)
  write.csv(df, file.path(OUT, paste0(id, ".csv")), row.names = FALSE, quote = TRUE)
  cat(sprintf("[%s] %d methods, %d rows\n", id, length(unique(df$method)), nrow(df)))
}
cat("EXTRACT_V2_DONE\n")
