# Fig. 5b and c, for the 99 expected program-subtype pairs.
#   b  per pair, the number of the program's genes in RECAST's top 30 minus the number in COSG's top 30: a box per
#      group (all pairs, then each compartment), the pairs as jittered points, the mean as a dashed line
#   c  per group, the percentage of pairs enriched by the genes only RECAST's top 30 holds and by those only COSG's
#      holds, with counts above the bars and 95% intervals for all pairs
# Both panels have the same plot width and height, set in the gtable, so that their nine columns share one pitch; the
# width is solved so that the row fills 4.4-179.6 mm with GAP mm between b's ink and c's.
#
#   Rscript panels_b_c.R
# reads  ../shared_data/enrichment_pairs.csv, enrichment_summary.csv, program_genes_method_specific.csv,
#        program_genes_by_comparison.csv
# writes out/panels/panel_b.{svg,pdf,png}, out/panels/panel_c.{svg,pdf,png}

suppressMessages({
  library(readr); library(dplyr); library(ggplot2); library(cowplot); library(grid); library(gtable)
  library(systemfonts); library(svglite); library(ragg)
})

here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
ST <- "expected"

FS     <- 7.0
FAMILY <- "Liberation Sans"
LW     <- 0.40 / (0.75 * .pt)                   # ggplot linewidth giving 0.4 pt
RECAST_BLUE <- "#4E86B0"; COSG_ORANGE <- "#D94801"; REF_COL <- "#8C8C8C"
MINUS  <- function(s) gsub("-", "−", s, fixed = TRUE)
fmt_p  <- function(p) if (p < 0.001) "P < 0.001" else if (p < 0.1) sprintf("P = %.3f", p) else sprintf("P = %.2f", p)
PT2MM  <- 25.4 / 72
LH     <- FS * PT2MM                             # one text line at lineheight 1.0, mm

ROW_X0 <- 4.4; ROW_X1 <- 179.6                   # Fig. 5's panel row (letters at 0.4 and 4 mm left of c's ink)
GAP    <- 6.0                                    # b's ink to c's ink, mm (c's letter sits in it)
PH     <- 30.0                                   # plot height, mm, both panels
YLAB_BOX <- "Program genes,\nRECAST − COSG"
YLAB_BAR <- "Pairs enriched by\nspecific genes (%)"

# ---- data ---------------------------------------------------------------------------------------------------------
D  <- file.path(here, "..", "shared_data")
P  <- read_csv(file.path(D, "enrichment_pairs.csv"), show_col_types = FALSE)
SU <- read_csv(file.path(D, "enrichment_summary.csv"), show_col_types = FALSE) %>% filter(kind == "capture")
SP <- read_csv(file.path(D, "program_genes_method_specific.csv"), show_col_types = FALSE) %>% filter(set == "specific")
CC <- read_csv(file.path(D, "program_genes_by_comparison.csv"), show_col_types = FALSE)
COMP <- c("CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas")   # a's column groups, left to right
GRP  <- c("All pairs", COMP)
SETS <- c(E = "expected", U = "unexpected")
stopifnot(sum(P$cls == "E") == 99, sum(P$cls == "U") == 1164, setequal(unique(P$comp), COMP), nrow(SU) == 2,
          nrow(SP) == 2, identical(CC$comp, COMP))
su <- filter(SU, pairs == ST)
sp <- filter(SP, metric == c(expected = "recovery", unexpected = "unexpected")[ST])
stopifnot(nrow(su) == 1, nrow(sp) == 1)

# ---- b: the per-pair difference ------------------------------------------------------------------------------------
BOX_LO <- -7.6; BOX_HI <- 6.6; BOX_FLOOR <- -7    # y range; pairs below the floor are drawn on it
BOX_BRK <- c(-6, -3, 0, 3, 6)
BOX_W <- 0.6; JIT <- 0.2
BX <- bind_rows(P %>% mutate(grp = "All pairs"), P %>% mutate(grp = comp)) %>%
  mutate(set = unname(SETS[cls]), gi = match(grp, GRP))
# horizontal jitter: one runif draw (seed 1) over the "All pairs" rows and the expected pairs' compartment rows, in
# this order; Supplementary Fig. 6a draws its "All pairs" points from the same draw
first <- BX$set == "expected" | BX$grp == "All pairs"
BX$u <- NA_real_
set.seed(1); BX$u[first] <- runif(sum(first), -JIT, JIT)
BX <- BX %>% filter(set == ST) %>%
  mutate(xj = gi + u, yp = pmax(dk, BOX_FLOOR), off = dk < BOX_FLOOR, sgn = sign(dk))
stopifnot(sum(BX$off) == 2, all(BX$dk[BX$off] == -12))   # one pair (cycling ASC, E2F targets): All pairs, B / ASC
stopifnot(max(BX$dk) < BOX_HI, abs(mean(BX$dk[BX$grp == "All pairs"]) - su$diff) < 1e-9)
NB <- BX %>% count(gi, grp) %>% arrange(gi) %>% mutate(lab = sprintf("%s (%s)", grp, formatC(n, big.mark = ",")))
stopifnot(nrow(NB) == 9, NB$n[1] == su$n_pairs, sum(NB$n[-1]) == su$n_pairs)
stopifnot(all(NB$n[-1] == CC$n))

# ---- c: enrichment by the genes only one top 30 holds -------------------------------------------------------------
BYC <- P %>% filter(cls == names(SETS)[SETS == ST]) %>% group_by(comp) %>%
  summarise(n = n(), k_r = sum(hit_recast_specific), k_c = sum(hit_cosg_specific), .groups = "drop") %>%
  slice(match(COMP, comp))
stopifnot(identical(BYC$comp, COMP), sum(BYC$n) == sp$n, sum(BYC$k_r) == sp$recast_n, sum(BYC$k_c) == sp$cosg_n,
          all(BYC$n == NB$n[-1]))
stopifnot(all(BYC$k_r == CC$recast_specific), all(BYC$k_c == CC$cosg_specific))
G <- bind_rows(tibble(grp = "All pairs", n = sp$n, k_r = sp$recast_n, k_c = sp$cosg_n,
                      lo_r = sp$r_lo, hi_r = sp$r_hi, lo_c = sp$c_lo, hi_c = sp$c_hi),
               BYC %>% transmute(grp = comp, n, k_r, k_c, lo_r = NA, hi_r = NA, lo_c = NA, hi_c = NA)) %>%
  mutate(gi = match(grp, GRP))
B <- bind_rows(
  G %>% transmute(gi, grp, n, method = "RECAST-specific", k = k_r, lo = lo_r, hi = hi_r, dx = -0.15),
  G %>% transmute(gi, grp, n, method = "COSG-specific",   k = k_c, lo = lo_c, hi = hi_c, dx = +0.15)) %>%
  mutate(pct = 100 * k / n, x = gi + dx, method = factor(method, levels = c("RECAST-specific", "COSG-specific")))
stopifnot(abs(B$pct[B$gi == 1 & B$method == "RECAST-specific"] - sp$recast) < 1e-9,
          abs(B$pct[B$gi == 1 & B$method == "COSG-specific"] - sp$cosg) < 1e-9,
          with(filter(B, !is.na(lo)), all(lo <= pct & pct <= hi)))
YMAX <- 50
YBRK <- seq(0, 50, 10)
mm2y <- function(v) v * YMAX / PH
B <- B %>% mutate(ylab = ifelse(is.na(hi), pct, hi) + mm2y(0.8))
stopifnot(max(B$ylab) + mm2y(LH) < YMAX)          # every count below the panel top

# ---- the two panels -----------------------------------------------------------------------------------------------
TITLE <- "Expected pairs"
SUB_BOX <- MINUS(sprintf("Mean difference %+.2f genes (95%% CI %.2f to %.2f); %s", su$diff, su$lo, su$hi, fmt_p(su$p)))
SUB_BAR <- MINUS(sprintf("Difference %+.1f points (95%% CI %.1f to %.1f); %s", sp$diff, sp$d_lo, sp$d_hi,
                         fmt_p(sp$p_perm)))
theme_fig <- function() theme_cowplot(font_size = FS, font_family = FAMILY, line_size = LW) +
  theme(plot.title = element_text(size = FS, face = "bold", hjust = 0, margin = margin(0, 0, 0.6, 0, "mm")),
        plot.subtitle = element_text(size = FS, hjust = 0, lineheight = 1.0, margin = margin(0, 0, 1.2, 0, "mm")),
        plot.title.position = "plot", axis.ticks.length = unit(1, "mm"),
        axis.text = element_text(colour = "black"),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, margin = margin(0.6, 0, 0, 0, "mm")),
        axis.title.y = element_text(lineheight = 1.0, margin = margin(0, 1.5, 0, 0, "mm")),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, "mm"))
XL <- c(0.4, length(GRP) + 0.6)

box_plot <- function() {
  ggplot(BX, aes(x = gi, y = dk, group = gi)) +
    geom_hline(yintercept = 0, linetype = "22", linewidth = LW, colour = REF_COL) +
    geom_boxplot(width = BOX_W, outlier.shape = NA, fill = "#EFEFEF", colour = NA) +          # fill, under the points
    geom_point(aes(x = xj, y = yp, colour = factor(sgn), fill = factor(sgn), shape = off, size = off),
               alpha = 0.7, stroke = 0.2) +
    geom_boxplot(width = BOX_W, outlier.shape = NA, fill = NA, colour = "black", linewidth = LW) +   # outlines, over
    stat_summary(fun = mean, geom = "errorbar", aes(ymax = after_stat(y), ymin = after_stat(y)), width = BOX_W,
                 linetype = "44", linewidth = LW, colour = "black") +
    geom_text(data = filter(BX, off), aes(x = xj + 0.13, y = yp, label = MINUS(as.character(dk))), hjust = 0,
              vjust = 0.5, size = FS / .pt, family = FAMILY, inherit.aes = FALSE) +
    scale_colour_manual(values = c(`-1` = COSG_ORANGE, `0` = "#9E9E9E", `1` = RECAST_BLUE), guide = "none") +
    scale_fill_manual(values = c(`-1` = COSG_ORANGE, `0` = "#9E9E9E", `1` = RECAST_BLUE), guide = "none") +
    scale_shape_manual(values = c(`FALSE` = 21, `TRUE` = 25), guide = "none") +
    scale_size_manual(values = c(`FALSE` = 0.7, `TRUE` = 1.1), guide = "none") +
    scale_x_continuous(breaks = NB$gi, labels = NB$lab, expand = c(0, 0)) +
    scale_y_continuous(breaks = BOX_BRK, labels = MINUS(as.character(BOX_BRK)), expand = c(0, 0)) +
    coord_cartesian(xlim = XL, ylim = c(BOX_LO, BOX_HI), clip = "off") +
    labs(title = TITLE, subtitle = SUB_BOX, x = NULL, y = YLAB_BOX) +
    theme_fig()
}
bar_plot <- function() {
  ggplot(B, aes(x = x, y = pct)) +
    geom_col(aes(fill = method), width = 0.3) +
    geom_errorbar(data = filter(B, !is.na(lo)), aes(ymin = lo, ymax = hi), width = 0.1, linewidth = LW,
                  colour = "black") +
    geom_text(aes(y = ylab, label = k), vjust = 0, size = FS / .pt, family = FAMILY) +
    scale_fill_manual(values = c(`RECAST-specific` = RECAST_BLUE, `COSG-specific` = COSG_ORANGE), name = NULL) +
    scale_x_continuous(breaks = NB$gi, labels = NB$lab, expand = c(0, 0)) +
    scale_y_continuous(breaks = YBRK, expand = c(0, 0)) +
    coord_cartesian(xlim = XL, ylim = c(0, YMAX), clip = "off") +
    labs(title = TITLE, subtitle = SUB_BAR, x = NULL, y = YLAB_BAR) +
    theme_fig() +
    theme(legend.position = "inside", legend.position.inside = c(1, 1), legend.justification = c(1, 1),
          legend.key.size = unit(2.4, "mm"), legend.key.spacing.y = unit(0.6, "mm"),
          legend.text = element_text(size = FS, margin = margin(0, 0, 0, 1.0, "mm")),
          legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(0, "mm"))
}

# ---- exact sizes: one plot width W for both panels, the row filled ------------------------------------------------
svglite::svglite(tempfile(fileext = ".svg"), width = 7.2, height = 3)   # text metrics as the SVGs will have them
gb <- ggplotGrob(box_plot()); gc <- ggplotGrob(bar_plot())
panel_of <- function(g) g$layout[g$layout$name == "panel", ]
mm_w <- function(g, i) sum(convertWidth(g$widths[i], "mm", valueOnly = TRUE))
mm_h <- function(g, i) sum(convertHeight(g$heights[i], "mm", valueOnly = TRUE))
pb <- panel_of(gb); pc <- panel_of(gc)
stopifnot(pb$t == pc$t, nrow(gb) == nrow(gc))
for (i in seq_len(nrow(gb))) if (i != pb$t) {                # every row above and below the panel: the larger of the two
  h <- max(mm_h(gb, i), mm_h(gc, i)); gb$heights[i] <- unit(h, "mm"); gc$heights[i] <- unit(h, "mm")
}
M_L <- 0.5                                                    # plot.margin left (whitespace, not ink)
Lb <- mm_w(gb, seq_len(pb$l - 1)) - M_L; Lc <- mm_w(gc, seq_len(pc$l - 1)) - M_L
W  <- ((ROW_X1 - ROW_X0) - GAP - Lb - Lc) / 2
for (g in c("gb", "gc")) {
  x <- get(g); p <- panel_of(x)
  x$widths[p$l] <- unit(W, "mm"); x$heights[p$t] <- unit(PH, "mm")
  assign(g, x)
}
# the subtitles must end inside their panels' width (their ink then ends at the x axis's end)
sw <- function(s) as.numeric(systemfonts::string_width(s, family = FAMILY, size = FS, res = 72)) * PT2MM
stopifnot(sw(SUB_BOX) < Lb + W - 1, sw(SUB_BAR) < Lc + W - 1)
TW <- c(b = mm_w(gb, seq_along(gb$widths)), c = mm_w(gc, seq_along(gc$widths)))
TH <- c(b = mm_h(gb, seq_along(gb$heights)), c = mm_h(gc, seq_along(gc$heights)))
invisible(dev.off())
stopifnot(abs(TH[["b"]] - TH[["c"]]) < 1e-9)

OUT <- file.path(here, "out", "panels")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
save_grob <- function(g, key, w, h) {
  out <- file.path(OUT, key)
  svglite::svglite(paste0(out, ".svg"), width = w / 25.4, height = h / 25.4); grid.draw(g); invisible(dev.off())
  cairo_pdf(paste0(out, ".pdf"), width = w / 25.4, height = h / 25.4); grid.draw(g); invisible(dev.off())
  ragg::agg_png(paste0(out, ".png"), width = w / 25.4, height = h / 25.4, units = "in", res = 600, background = "white")
  grid.draw(g); invisible(dev.off())
}
save_grob(gb, "panel_b", TW[["b"]], TH[["b"]])
save_grob(gc, "panel_c", TW[["c"]], TH[["c"]])

cat(sprintf(paste0("[Fig. 5b,c] %s pairs: plot %.2f x %.2f mm each (column pitch %.2f mm); left decorations b %.2f, c %.2f mm; ",
                   "canvases b %.2f x %.2f, c %.2f x %.2f mm\n  b: %s | %s\n  c: %s | %s | RECAST/COSG by compartment: %s\n"),
            ST, W, PH, W / diff(XL), Lb, Lc, TW[["b"]], TH[["b"]], TW[["c"]], TH[["c"]],
            paste(NB$lab, collapse = ", "), SUB_BOX, SUB_BAR,
            sprintf("all %d/%d", sp$recast_n, sp$cosg_n),
            paste(sprintf("%s %d/%d", BYC$comp, BYC$k_r, BYC$k_c), collapse = ", ")))
