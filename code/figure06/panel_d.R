# Fig. 6d: change in balanced accuracy when genes are added to each curated program panel (0, 5, 10, 20, 40 genes),
# for RECAST, ANS and the reference-free method side by side at each dose, over the 28 tasks; stars mark the doses at
# which RECAST's change is significantly larger than a comparator's (scoring_added_genes_stats.csv).
#   Rscript panel_d.R
# reads  ../shared_data/scoring_added_genes.csv, scoring_added_genes_stats.csv
# writes out/panels/panel_d.svg (71.2 x 58 mm, no background)
suppressMessages({ library(dplyr); library(readr); library(ggplot2); library(cowplot); library(svglite); library(systemfonts); library(grid) })
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
W_MM <- 71.2; H_MM <- 58; FS <- 6.2; K <- FS / 14; FAMILY <- "Liberation Sans"
PANEL_TOP <- 7.91      # pt below the canvas top: top edge of the plot panel
DOSES <- c("0", "5", "10", "20", "40")
TOOLS <- c("RECAST", "ANS", "SIGnature")                              # the methods' keys in the data
LABELS <- c("RECAST", "ANS", "Reference-free")
COL <- c(RECAST = "#4E86B0", ANS = "#4D4D4D", SIGnature = "#7E9E77")
DW <- 0.84                                                             # the three columns inside one dose

my_theme = function() {
  background_grid(major = "y", minor = "none", size.major = 0.4 * K) +
    theme(title = element_text(size = FS), axis.text = element_text(size = FS),
          strip.background = element_rect(colour = "white", fill = "white"), strip.text.x = element_text(size = FS))
}
theme_set(theme_cowplot(font_size = FS, font_family = FAMILY, line_size = 0.5 * K))

D <- file.path(here, "..", "shared_data")
raw <- read_csv(file.path(D, "scoring_added_genes.csv"), show_col_types = FALSE) %>% filter(metric == "balanced_acc", tool %in% TOOLS)
stopifnot(all(raw$value[raw$dose == 0] == 0), nrow(raw) == 3 * 28 * 5)
stopifnot(min(raw$value) > -0.35, max(raw$value) < 0.05)                               # every point inside the limits
d  <- raw %>% filter(as.character(dose) %in% DOSES) %>%
  mutate(tool = factor(tool, levels = TOOLS), dropin = factor(as.character(dose), levels = DOSES), strip = " ")
d0 <- d %>% filter(dose == 0) %>% distinct(tool, dropin, value, strip); d1 <- d %>% filter(dose != 0)
sig <- read_csv(file.path(D, "scoring_added_genes_stats.csv"), show_col_types = FALSE) %>%
  filter(metric == "balanced_acc", comparison == "RECAST - comparator", comparator %in% TOOLS) %>%
  mutate(tool = factor(comparator, levels = TOOLS), dropin = factor(as.character(dose), levels = DOSES),
         x = as.numeric(dropin) + (as.numeric(tool) - 2) * DW / 3, strip = " ")
stopifnot(nrow(sig) == 8, all(sig$mean > 0))
YLIM <- c(-0.35, 0.05); BRK <- scales::breaks_extended()(YLIM); BRK <- BRK[BRK >= YLIM[1] - 1e-9 & BRK <= YLIM[2] + 1e-9]
STAR_Y <- YLIM[2] + 0.05 * diff(YLIM); YTOP <- YLIM[2] + 0.10 * diff(YLIM)

p <- ggplot(d1, aes(x = dropin, y = value, colour = tool)) +
  geom_point(position = position_jitterdodge(jitter.width = 0.22, jitter.height = 0, dodge.width = DW, seed = 1), alpha = .6, size = 2.2 * K, stroke = 0.3 * K) +
  geom_violin(fill = NA, linewidth = 0.5 * K, position = position_dodge(width = DW), width = DW * 0.92, scale = "width") +
  geom_point(data = d0, position = position_dodge(width = DW), alpha = .6, size = 2.2 * K, stroke = 0.3 * K) +
  geom_hline(yintercept = 0, lty = 2, linewidth = 0.5 * K) +
  geom_text(data = sig, aes(x = x, y = STAR_Y, label = stars), inherit.aes = FALSE, size = FS / .pt, family = FAMILY) +
  scale_colour_manual(values = COL, limits = TOOLS) + scale_x_discrete(limits = DOSES) +
  scale_y_continuous(limits = c(YLIM[1], YTOP), breaks = BRK, labels = function(x) sub("^-", "−", format(x, trim = TRUE))) +
  labs(x = "Genes added per curated panel", y = "Change in balanced accuracy") + facet_wrap(~strip) + my_theme() +
  theme(legend.position = "none",
        plot.margin  = margin(0, FS / 2, 1.98, FS / 2),
        strip.text.x = element_text(margin = margin(0.83, FS / 4, FS / 4, FS / 4)),
        axis.text.x  = element_text(margin = margin(t = 1.97)), axis.title.x = element_text(margin = margin(t = 2.36)))

# key: one row above the plot, its left end KEY_X0 pt from the centre of the plot, drawn inside the plot (no ggdraw:
# cowplot lays the plot out under pdf(NULL), which does not know the font, loses the text descent and moves the axis)
PW <- 1.1 * FS; PH <- 0.7 * FS; PAD <- 0.35 * FS; GAP <- 1.0 * FS; DY <- PANEL_TOP - 5.82
KEY_X0 <- -51.185
tw <- string_width(LABELS, family = FAMILY, size = FS, res = 72)
x0 <- KEY_X0
key <- list()
for (i in seq_along(TOOLS)) {
  key[[length(key) + 1]] <- rectGrob(x = unit(.5, "npc") + unit(x0, "pt"), y = unit(1, "npc") + unit(DY, "pt"), width = unit(PW, "pt"), height = unit(PH, "pt"), just = c("left", "bottom"),
                                     gp = gpar(fill = COL[[TOOLS[i]]], col = "black", lwd = 0.3 / 0.75))
  key[[length(key) + 1]] <- textGrob(LABELS[i], x = unit(.5, "npc") + unit(x0 + PW + PAD, "pt"), y = unit(1, "npc") + unit(DY, "pt"), just = c("left", "bottom"),
                                     gp = gpar(fontsize = FS, fontfamily = FAMILY, col = "black"))
  x0 <- x0 + PW + PAD + tw[i] + GAP
}
q <- p + annotation_custom(do.call(grobTree, key)) + coord_cartesian(clip = "off") + theme(strip.background = element_blank())
OUT <- file.path(here, "out", "panels")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
# no background, so nothing covers the panel letter beside it
ggsave(file.path(OUT, "panel_d.svg"), q, width = W_MM, height = H_MM, units = "mm", device = svglite, bg = "transparent", fix_text_size = FALSE)
