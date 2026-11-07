# Selection time of every comparator run (the `time` element each run-<method>.R stores: system.time() around
# the method call alone, without R start-up, loading or writing), for the 20 subtype lineages, from the separate timing
# run of timing_run.sh.
# -> /path/to/cpu_server/analysis/marker_methods/field_selection_times_inprocess_battery.csv
suppressMessages(library(parallel))
ids <- c("fs_asc_zhao","fs_gut_smillie","fs_lung_hlca","fs_nk_pbmcbmn","fs_nk_zhao",
         "tf_b_ren","tf_b_steph","tf_b_yosh","tf_cd4_dom","tf_cd4_ren","tf_cd4_steph","tf_cd4_yosh",
         "tf_cd8_dom","tf_cd8_ren","tf_cd8_steph","tf_cd8_yosh","tf_mono_dom","tf_mono_ren","tf_mono_steph","tf_mono_yosh",
         "pbmc3k_allgenes","lawlor_allgenes","dominguez_conde_allgenes","ren_allgenes","stephenson_allgenes","yoshida_allgenes")
files <- list.files("/path/to/cpu_server/analysis/marker_methods/results/real_data", pattern = "\\.rds$", full.names = TRUE)
keep <- sapply(files, function(f) sub("-.*$", "", basename(f)) %in% ids)
files <- files[keep]
cat("files:", length(files), "\n")
rows <- mclapply(files, function(f) {
  b <- sub("\\.rds$", "", basename(f)); d <- sub("-.*$", "", b); m <- sub("^[^-]*-", "", b)
  x <- tryCatch(readRDS(f), error = function(e) NULL)
  t <- if (is.list(x) && !is.null(x$time)) as.numeric(x$time)[1] else NA_real_
  ng <- if (is.list(x) && !is.null(x$result) && is.data.frame(x$result)) nrow(x$result) else NA_integer_
  data.frame(data_id = d, method_id = m, time_s = t, n_result_rows = ng, file_mtime = format(file.info(f)$mtime, "%Y-%m-%d %H:%M"), stringsAsFactors = FALSE)
}, mc.cores = 32)
df <- do.call(rbind, rows)
write.csv(df, "/path/to/cpu_server/analysis/marker_methods/field_selection_times_inprocess_battery.csv", row.names = FALSE)
cat("wrote", nrow(df), "rows; NA times:", sum(is.na(df$time_s)), "\n")
print(aggregate(time_s ~ data_id, df, function(v) round(median(v), 2)))
