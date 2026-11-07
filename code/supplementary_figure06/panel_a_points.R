# Supplementary Fig. 6a: positions of the points (one per unexpected program x population pair, in "All pairs" and in
# its compartment).  The horizontal jitter is drawn with R's runif and seeds 1 and 2, in this row order, and
# 01_draw.py draws the points at these positions.  Nothing is drawn here.
#
#   Rscript panel_a_points.R <enrichment_pairs.csv> <enrichment_summary.csv> <out.csv>
# writes <out.csv>: one row per point (grp, gi, u, xj, dk, yp, off, sgn) and the identifiers of its pair

suppressMessages({ library(readr); library(dplyr) })
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
P  <- read_csv(args[1], show_col_types = FALSE)
SU <- read_csv(args[2], show_col_types = FALSE) %>% filter(kind == "capture")
COMP <- c("CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas")
GRP  <- c("All pairs", COMP)
SETS <- c(E = "expected", U = "unexpected")
ST <- "unexpected"
stopifnot(sum(P$cls == "E") == 99, sum(P$cls == "U") == 1164, setequal(unique(P$comp), COMP), nrow(SU) == 2)
su <- filter(SU, pairs == ST)

# jitter: seed 1 for the "All pairs" rows (and expected pairs), seed 2 for the compartment rows
BOX_LO <- -3.6; BOX_HI <- 3.6; BOX_FLOOR <- -3.6
JIT <- 0.2
BX <- bind_rows(P %>% mutate(grp = "All pairs"), P %>% mutate(grp = comp)) %>%
  mutate(set = unname(SETS[cls]), gi = match(grp, GRP))
first <- BX$set == "expected" | BX$grp == "All pairs"
BX$u <- NA_real_
set.seed(1); BX$u[first]  <- runif(sum(first), -JIT, JIT)
set.seed(2); BX$u[!first] <- runif(sum(!first), -JIT, JIT)
BX <- BX %>% filter(set == ST) %>%
  mutate(xj = gi + u, yp = pmax(dk, BOX_FLOOR), off = dk < BOX_FLOOR, sgn = sign(dk))
stopifnot(!any(BX$off), max(BX$dk) < BOX_HI, abs(mean(BX$dk[BX$grp == "All pairs"]) - su$diff) < 1e-9)

write_csv(BX %>% select(grp, gi, u, xj, dk, yp, off, sgn, program, collection, term, lineage, state, comp, subtype,
                        k_recast, k_cosg), args[3])
cat(sprintf("Supplementary Fig. 6a points: %d rows (%d pairs x 2 groupings)\n", nrow(BX), sum(BX$grp == "All pairs")))
