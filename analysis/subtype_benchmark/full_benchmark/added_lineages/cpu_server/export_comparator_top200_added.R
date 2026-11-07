#!/usr/bin/env Rscript
# Depth-200 gene lists of the comparator methods on the added lineages, for the marker-rank@200 and
# contamination@200 endpoints (as subtype_benchmark/export_comparator_top200.R).  Writes fs_genesets200_v2/<id>.csv.
# CPU server (marker_methods R).  Usage: Rscript export_comparator_top200_added.R
RD <- "/path/to/cpu_server/added_lineages/marker_methods/results/real_data"
BASE <- "/path/to/cpu_server/added_lineages"
TOPN <- 200
IDS <- readLines("/path/to/cpu_server/added_lineages/marker_methods/data/fine_state_export_v2/V2_IDS.txt"); IDS <- IDS[nzchar(IDS)]

out <- file.path(BASE, "fs_genesets200_v2")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
for (id in IDS) {
    pref <- paste0(id, "_v2-")
    files <- Sys.glob(file.path(RD, paste0(pref, "*.rds")))
    files <- files[!grepl("recast", files)]
    rows <- list()
    for (f in files) {
      meth <- sub("\\.rds$", "", sub(paste0("^", pref), "", basename(f)))
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
    write.csv(df, file.path(out, paste0(id, ".csv")), row.names = FALSE, quote = TRUE)
    cat(sprintf("[%s] %d methods, %d rows\n", id, length(unique(df$method)), nrow(df)))
}
cat("export done\n")
