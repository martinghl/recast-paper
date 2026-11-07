# Supplementary Fig. 4b: RECAST's AUROC (recast_auroc.py: per population, AUROC of the mean expression of the
# selected genes; populations >= 10 cells; unweighted mean over populations) and the comparators' AUROC
# (spec_metrics of the harness: per selected gene, Mann-Whitney AUROC; unweighted mean over all population x gene
# pairs) are computed for both sides under both definitions on the same SCE.  Reads only.
suppressPackageStartupMessages({library(dplyr); library(purrr); library(tibble); library(SingleCellExperiment)})
setwd("/path/to/cpu_server/analysis/marker_methods_grid")
source("/path/to/cpu_server/analysis/marker_methods/code/analysis-utils.R")
HERE <- "/path/to/cpu_server/analysis/broad_auroc"
PAN <- rbind(read.csv(file.path(HERE, "recast_broad_panels.csv"), colClasses = c(cluster = "character")),
                 read.csv(file.path(HERE, "recast_broad_panels_citeseq.csv"), colClasses = c(cluster = "character")))
PAN$cluster <- as.character(PAN$cluster)
ATL <- c("pbmc3k","ren","stephenson","citeseq","dominguez_conde","yoshida","ss3_pbmc","lawlor")
K <- 20L
topk <- function(res, k) res$result %>% group_by(cluster) %>% group_split() %>%
  setNames(map_chr(., ~as.character(.x$cluster[[1]]))) %>% map(~ get_top_sel_mgs(.x, n=k)$gene)
one_atlas <- function(a) {
  ID <- paste0(a, "_allgenes"); sce <- readRDS(sprintf("data/real_data/%s.rds", ID))
  E <- t(as.matrix(logcounts(sce))); y <- as.character(colLabels(sce))
  big <- names(which(table(y) >= 10))
  rk_cache <- new.env()
  rnk <- function(g) { if (is.null(rk_cache[[g]])) rk_cache[[g]] <- rank(E[, g]); rk_cache[[g]] }
  auc_from_rank <- function(r, inc) { nc <- sum(inc); nr <- sum(!inc); (sum(r[inc]) - nc*(nc+1)/2) / (nc*nr) }
  per_gene <- function(gl) {               # = spec_metrics()[["auroc"]], same logic
    aur <- c(); np <- 0L
    for (cl in names(gl)) { g <- gl[[cl]]; g <- g[g %in% colnames(E)]; if (!length(g)) next
      inc <- y == cl; if (sum(inc) == 0 || sum(!inc) == 0) next
      np <- np + 1L; aur <- c(aur, vapply(g, function(gg) auc_from_rank(rnk(gg), inc), 0)) }
    c(auroc = mean(aur), n_pop = np, n_pairs = length(aur)) }
  signature <- function(gl) {               # = recast_auroc.py auroc(): >= 10-cell populations only
    a <- c()
    for (cl in intersect(sort(big), names(gl))) { g <- gl[[cl]]; g <- g[g %in% colnames(E)]; if (!length(g)) next
      inc <- y == cl; s <- rowMeans(E[, g, drop = FALSE]); a <- c(a, auc_from_rank(rank(s), inc)) }
    c(auroc = mean(a), n_pop = length(a), n_pairs = NA) }
  present <- list.files("results/real_data", pattern = sprintf("^%s-.*\\.rds$", ID))
  present <- sub(sprintf("^%s-", ID), "", sub("\\.rds$", "", present))
  BASE <- present                                  # all procedures the grid wrote
  out <- list()
  p <- PAN[PAN$dataset == a, ]; p <- p[order(p$cluster, p$rank), ]
  glR <- lapply(split(p$gene, p$cluster), function(v) head(as.character(v), K))
  for (m in c("RECAST", BASE)) {
    gl <- if (m == "RECAST") glR else topk(readRDS(sprintf("results/real_data/%s-%s.rds", ID, m)), K)
    pg <- per_gene(gl); sg <- signature(gl)
    out[[m]] <- tibble(atlas = a, method = m, auroc_per_gene = pg[["auroc"]], auroc_signature = sg[["auroc"]],
                       n_pop_per_gene = pg[["n_pop"]], n_pairs_per_gene = pg[["n_pairs"]], n_pop_signature = sg[["n_pop"]],
                       n_labels = length(unique(y)), n_labels_ge10 = length(big),
                       recast_clusters_matched = if (m == "RECAST") sum(names(glR) %in% unique(y)) else NA_integer_,
                       recast_clusters_total = if (m == "RECAST") length(glR) else NA_integer_)
  }
  cat(sprintf("[%s] %d methods done\n", a, length(out))); bind_rows(out)
}
res <- bind_rows(parallel::mclapply(ATL, function(a) tryCatch(one_atlas(a), error = function(e) {cat(a, conditionMessage(e), "\n"); tibble()}),
                                    mc.cores = 8, mc.preschedule = FALSE))
readr::write_csv(res, file.path(HERE, "auroc_two_definitions_all60.csv"))
cat("DONE rows", nrow(res), "\n")
