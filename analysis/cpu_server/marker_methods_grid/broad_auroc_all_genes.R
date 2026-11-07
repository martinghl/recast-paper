#!/usr/bin/env Rscript
# Marker-independent metrics of the comparator methods' panels on the all-gene atlases: sum-max specificity,
# k-NN, AUROC and on-target recovery ("fast"), and the SVM ("svm"), with the evaluator used for every method:
#   SCE     = data/real_data/<ID>.rds            (ID ends in _allgenes, e.g. pbmc3k_allgenes)
#   results = marker_methods_grid/results/real_data/<ID>-<method>.rds
#   methods = every <ID>-<method>.rds in results/real_data (51 methods)
# Usage: Rscript broad_auroc_all_genes.R <ID> <fast|svm>   e.g. pbmc3k_allgenes fast
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(purrr); library(tibble); library(SingleCellExperiment)})
setwd("/path/to/cpu_server/analysis/marker_methods_grid")            # results/ here; data/ symlinks to ../marker_methods/data
source("/path/to/cpu_server/analysis/marker_methods/code/analysis-utils.R")
args <- commandArgs(trailingOnly=TRUE)
ID <- args[[1]]; METRICSET <- if(length(args)>=2) args[[2]] else "fast"
KS <- c(3L,5L,10L,20L)
OUT <- "/path/to/cpu_server/analysis/marker_methods_grid/arm3_full"; dir.create(OUT, showWarnings=FALSE, recursive=TRUE)

sce <- readRDS(sprintf("data/real_data/%s.rds", ID))
E <- t(as.matrix(logcounts(sce))); y <- colLabels(sce); LEV <- levels(y)

set.seed(20220502)
fi <- (tibble(n=seq_len(ncol(sce)), label=y) %>% group_by(label) %>% mutate(fold=sample(1:5,size=n(),replace=TRUE)))$fold

# methods = every <ID>-<method>.rds present (the 51 allgenes baselines, plain names)
present <- list.files("results/real_data", pattern=sprintf("^%s-.*\\.rds$", ID))
present <- sub(sprintf("^%s-", ID), "", sub("\\.rds$", "", present))
ALLM <- present

topk <- function(res, k) res$result %>% group_by(cluster) %>% group_split() %>%
  setNames(map_chr(., ~as.character(.x$cluster[[1]]))) %>% map(~ get_top_sel_mgs(.x, n=k)$gene)
f1_folds <- function(gl, feats, clf) {
  genes <- unique(flatten_chr(gl)); genes <- genes[genes %in% colnames(feats)]
  glf <- map(gl, ~ .x[.x %in% colnames(feats)]); X <- feats[, genes, drop=FALSE]
  median(map_dbl(1:5, function(f){ tr <- fi!=f
    p <- clf(X, tr, glf)
    cm <- as.matrix(table(true=factor(y[!tr],levels=LEV), prediction=factor(p,levels=LEV)))
    rc<-diag(cm)/rowSums(cm); rc[is.nan(rc)]<-0; pr<-diag(cm)/colSums(cm); pr[is.nan(pr)]<-0
    ff<-2*pr*rc/(pr+rc); ff[is.nan(ff)]<-0; median(ff) }))
}
clf_summax <- function(X, tr, gl){ te<-scale(X[!tr,,drop=FALSE])
  sm<-sapply(names(gl), function(k) rowSums(te[, gl[[k]], drop=FALSE], na.rm=TRUE))
  colnames(sm)[apply(sm,1,which.max)] }
clf_knn <- function(X, tr, gl){ as.character(FNN::knn(X[tr,,drop=FALSE], X[!tr,,drop=FALSE], y[tr], k=3)) }
clf_svm <- function(X, tr, gl){ m<-e1071::svm(X[tr,,drop=FALSE], y[tr]); as.character(predict(m, X[!tr,,drop=FALSE])) }
spec_metrics <- function(gl) {
  aur <- c(); ont <- c()
  for (cl in names(gl)) {
    g <- gl[[cl]]; g <- g[g %in% colnames(E)]; if(length(g)==0) next
    inc <- as.character(y)==cl; nc <- sum(inc); nr <- sum(!inc); if(nc==0||nr==0) next
    Xg <- E[, g, drop=FALSE]; R <- apply(Xg, 2, rank)
    U <- colSums(R[inc,,drop=FALSE]) - as.double(nc)*(nc+1)/2; aur <- c(aur, U/(as.double(nc)*nr))
    on <- colMeans(Xg[inc,,drop=FALSE]); off <- colMeans(Xg[!inc,,drop=FALSE])
    ont <- c(ont, on/(on+off+1e-12)) }
  c(auroc=mean(aur), ontarget=mean(ont)) }
one_method <- function(m) {
  f <- sprintf("results/real_data/%s-%s.rds", ID, m); if(!file.exists(f)) return(tibble())
  r <- readRDS(f); rows <- list()
  for (k in KS) {
    gl <- topk(r, k)
    if (METRICSET=="fast") {
      sm_e <- f1_folds(gl, E, clf_summax); kn_e <- f1_folds(gl, E, clf_knn); sp <- spec_metrics(gl)
      rows[[length(rows)+1]] <- tibble(method=m, k=k, currency="expression",
        metric=c("sum_max","knn","auroc","ontarget"), value=c(sm_e,kn_e,sp[["auroc"]],sp[["ontarget"]]))
    } else if (METRICSET=="svm") {
      rows[[length(rows)+1]] <- tibble(method=m, k=k, currency="expression", metric="svm", value=f1_folds(gl,E,clf_svm))
    }
  }
  bind_rows(rows)
}
NCORES <- min(24L, parallel::detectCores()-2L)
res <- bind_rows(parallel::mclapply(ALLM, function(m) tryCatch(one_method(m), error=function(e) tibble()),
                                    mc.cores=NCORES, mc.preschedule=FALSE))
res$level <- ID
readr::write_csv(res, file.path(OUT, sprintf("arm3_%s_%s.csv", ID, METRICSET)))
cat(sprintf("wrote %s : %d rows over %d methods\n",
    file.path(OUT, sprintf("arm3_%s_%s.csv", ID, METRICSET)), nrow(res), length(unique(res$method))))
