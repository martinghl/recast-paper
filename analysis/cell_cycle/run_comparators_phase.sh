#!/bin/bash
# Comparator methods on the phase-defined cell-cycle comparisons, on the CPU server:
#   nohup ./run_comparators_phase.sh 96 > logs/phase_grid.log 2>&1 &
# Same methods and harness as for the identity-defined comparisons.  Reads CC_FAIR_IDS.txt; the ids carry the "_cc"
# suffix that build_cell_cycle_sce.R writes.
set -uo pipefail
cd /path/to/cpu_server/analysis/marker_methods_grid
export PATH=/path/to/envs/marker_methods/bin:$PATH
export RETICULATE_PYTHON=/path/to/envs/marker_methods/bin/python
mkdir -p logs
DS="$(sed 's/$/_cc/' /path/to/cpu_server/analysis/marker_methods/data/cc_export/CC_FAIR_IDS.txt)"
METHODS="cepo cosg edger_glm edger_ql glmgampoi limma_trend limma_voom lm_two_sample logfc_abs logfc_raw presto random rankcorr_2 rankcorr_5 rankcorr_10 rankcorr_20 scanpy_t_rankby_abs scanpy_t_rankby_raw scanpy_tover_rankby_abs scanpy_tover_rankby_raw scanpy_wilcoxon_rankby_abs scanpy_wilcoxon_rankby_raw scanpy_wilcoxontiecorrect_rankby_abs scanpy_wilcoxontiecorrect_rankby_raw scran_findMarkers_binom_all scran_findMarkers_binom_any scran_findMarkers_binom_some scran_findMarkers_t_all scran_findMarkers_t_any scran_findMarkers_t_some scran_findMarkers_wilcox_all scran_findMarkers_wilcox_any scran_findMarkers_wilcox_some scran_scoreMarkers_max_AUC scran_scoreMarkers_max_logFC_cohen scran_scoreMarkers_max_logFC_detected scran_scoreMarkers_mean_AUC scran_scoreMarkers_mean_logFC_cohen scran_scoreMarkers_mean_logFC_detected scran_scoreMarkers_median_AUC scran_scoreMarkers_median_logFC_cohen scran_scoreMarkers_median_logFC_detected scran_scoreMarkers_min_AUC scran_scoreMarkers_min_logFC_cohen scran_scoreMarkers_min_logFC_detected scran_scoreMarkers_rank_AUC scran_scoreMarkers_rank_logFC_cohen scran_scoreMarkers_rank_logFC_detected seurat_LR seurat_MAST seurat_bimod seurat_negbinom seurat_poisson seurat_roc seurat_t seurat_wilcox smash venice"
CORES="${1:-96}"
for d in $DS; do [ -f "data/real_data/${d}.rds" ] || { echo "FATAL missing SCE ${d}.rds"; exit 2; }; done
T=""; for d in $DS; do for m in $METHODS; do T="$T results/real_data/${d}-${m}.rds"; done; done
echo "[phase_grid] $(echo $DS|wc -w) ids x $(echo $METHODS|wc -w) methods = $(echo $T|wc -w) targets, cores=$CORES"; date
snakemake -s Snakefile.py --cores "$CORES" --latency-wait 90 --rerun-incomplete --keep-going --nolock $T
echo "[phase_grid exit $?]"; date
for d in $DS; do echo "  $d: $(ls results/real_data/${d}-*.rds 2>/dev/null|wc -l)/58"; done
