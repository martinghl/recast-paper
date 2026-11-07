# R scorers without labels on every panel variant of a task (CPU server).
#   Rscript score_variants_r.R <task> <method> <variants.json> <out.csv.gz>
#   method in AUCell | ssGSEA | GSVA | zscore | VAM | singscore
# Settings: AUCell seed 0, default aucMaxRank; ssGSEA and GSVA on the sparse dgCMatrix (GSVA's single-cell default);
# z-score on the dense matrix (Lee 2008); VAM vamForCollection(center = FALSE, gamma = TRUE) with set.seed(0);
# singscore rankGenes(ties "min") + simpleScore.
# All variants' gene sets are scored in one call (sets named "<variant>::<candidate>"): each of these scorers scores
# each set independently given the cell rankings or gene statistics (ssGSEA's normalize = TRUE divides by one global
# constant, which changes neither the argmax, the AUROC nor a per-candidate z).  Empty panels are not passed -> NaN.
# Output: cells x sets CSV (column names "<variant>::<candidate>"), 7 significant digits.
suppressPackageStartupMessages({ library(Matrix); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
task <- args[1]; method <- args[2]; vfile <- args[3]; out <- args[4]
root <- "/path/to/cpu_server/program_scoring"
X <- as(readMM(gzfile(file.path(root, "rin", paste0(task, ".mtx.gz")))), "CsparseMatrix")
rownames(X) <- readLines(file.path(root, "rin", paste0(task, ".genes.txt")))
colnames(X) <- paste0("c", seq_len(ncol(X)) - 1L)
V <- fromJSON(vfile, simplifyVector = FALSE)
sets <- list()
for (v in names(V)) for (c in names(V[[v]])) {
  g <- unique(unlist(V[[v]][[c]])); g <- g[g %in% rownames(X)]
  if (length(g) > 0) sets[[paste0(v, "::", c)]] <- g
}
t0 <- Sys.time()
S <- NULL
if (method == "AUCell") {
  suppressPackageStartupMessages(library(AUCell))
  set.seed(0)
  rk <- AUCell_buildRankings(as.matrix(X), plotStats = FALSE, verbose = FALSE)
  S <- t(getAUC(AUCell_calcAUC(sets, rk, verbose = FALSE)))
} else if (method %in% c("ssGSEA", "GSVA")) {
  suppressPackageStartupMessages({ library(GSVA); library(BiocParallel) })
  p <- if (method == "ssGSEA") ssgseaParam(X, sets) else gsvaParam(X, sets)
  S <- t(as.matrix(gsva(p, verbose = FALSE, BPPARAM = SerialParam())))
} else if (method == "zscore") {
  suppressPackageStartupMessages({ library(GSVA); library(BiocParallel) })
  S <- t(as.matrix(gsva(zscoreParam(as.matrix(X), sets), verbose = FALSE, BPPARAM = SerialParam())))
} else if (method == "VAM") {
  suppressPackageStartupMessages(library(VAM))
  set.seed(0)
  gs <- lapply(sets, function(g) match(g, rownames(X)))
  r <- vamForCollection(gene.expr = t(as.matrix(X)), gene.set.collection = gs, center = FALSE, gamma = TRUE)
  S <- r$cdf.value; colnames(S) <- names(sets)
} else if (method == "singscore") {
  suppressPackageStartupMessages(library(singscore))
  rd <- rankGenes(as.matrix(X), tiesMethod = "min")
  S <- sapply(names(sets), function(s) simpleScore(rd, upSet = sets[[s]], centerScore = TRUE, knownDirection = TRUE)$TotalScore)
  rownames(S) <- colnames(X)
} else stop("unknown method ", method)
S <- S[colnames(X), , drop = FALSE]
missing <- setdiff(names(sets), colnames(S))
if (length(missing)) cat("WARNING", length(missing), "sets returned no score (NaN):", head(missing), "\n")
o <- gzfile(out, "w"); write.csv(signif(S, 7), o, row.names = FALSE); close(o)
cat("done", task, method, length(sets), "sets", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
