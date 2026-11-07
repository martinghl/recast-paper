#!/usr/bin/env Rscript
# Depth-200 gene lists of the comparator methods for the marker-rank@200 and contamination@200 endpoints
# (export_comparator_top50.R stops at 50).  geneset_from_rds keeps each file's row order per cluster, so head(..., 200)
# here gives the same lists.  Writes fs_genesets200_v2/<id>.csv.
# CPU server (marker_methods R).  Usage: Rscript export_comparator_top200.R
RD <- "/path/to/cpu_server/analysis/marker_methods/results/real_data"
BASE <- "/path/to/cpu_server/analysis/subtype_benchmark"
TOPN <- 200
LIN20 <- c("fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
           "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
           "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
           "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh")

out <- file.path(BASE, "fs_genesets200_v2")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
for (id in LIN20) {
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
