#!/bin/bash
# Run the comparator methods again on the 20 subtype lineages (their SCEs are in data/real_data/) to record each
# method's wall time with Snakemake's benchmark: directive (benchmarks/{id}-{method}_benchmark.txt).  The ids are added
# to config.yaml if missing.
set -uo pipefail
cd /path/to/cpu_server/analysis/marker_methods
export PATH=/path/to/envs/marker_methods/bin:$PATH
export RETICULATE_PYTHON=/path/to/envs/marker_methods/bin/python

IDS="fs_asc_zhao fs_gut_smillie fs_lung_hlca fs_nk_pbmcbmn fs_nk_zhao tf_b_ren tf_b_steph tf_b_yosh tf_cd4_dom tf_cd4_ren tf_cd4_steph tf_cd4_yosh tf_cd8_dom tf_cd8_ren tf_cd8_steph tf_cd8_yosh tf_mono_dom tf_mono_ren tf_mono_steph tf_mono_yosh"
METHODS="cepo cosg edger_glm edger_ql glmgampoi limma_trend limma_voom lm_two_sample logfc_abs logfc_raw random scanpy_t_rankby_abs scanpy_t_rankby_raw scanpy_tover_rankby_abs scanpy_tover_rankby_raw scanpy_wilcoxon_rankby_abs scanpy_wilcoxon_rankby_raw scanpy_wilcoxontiecorrect_rankby_abs scanpy_wilcoxontiecorrect_rankby_raw scran_findMarkers_binom_all scran_findMarkers_binom_any scran_findMarkers_binom_some scran_findMarkers_t_all scran_findMarkers_t_any scran_findMarkers_t_some scran_findMarkers_wilcox_all scran_findMarkers_wilcox_any scran_findMarkers_wilcox_some scran_scoreMarkers_max_AUC scran_scoreMarkers_max_logFC_cohen scran_scoreMarkers_max_logFC_detected scran_scoreMarkers_mean_AUC scran_scoreMarkers_mean_logFC_cohen scran_scoreMarkers_mean_logFC_detected scran_scoreMarkers_median_AUC scran_scoreMarkers_median_logFC_cohen scran_scoreMarkers_median_logFC_detected scran_scoreMarkers_min_AUC scran_scoreMarkers_min_logFC_cohen scran_scoreMarkers_min_logFC_detected scran_scoreMarkers_rank_AUC scran_scoreMarkers_rank_logFC_cohen scran_scoreMarkers_rank_logFC_detected seurat_LR seurat_MAST seurat_bimod seurat_negbinom seurat_poisson seurat_roc seurat_t seurat_wilcox"
CORES="${1:-120}"

# idempotent id registration into all_data_ids:
python3 - "$IDS" <<'PY'
import sys
ids=sys.argv[1].split()
cfg="config.yaml"; s=open(cfg).read()
missing=[i for i in ids if f'"{i}"' not in s]
if missing:
    lines=s.splitlines(); out=[]; inblk=False; done=False
    for i,l in enumerate(lines):
        out.append(l)
        if l.strip().startswith("all_data_ids:"): inblk=True; continue
        if inblk and not done:
            nxt=lines[i+1] if i+1<len(lines) else ""
            if not (nxt.strip().startswith("- ") or nxt.strip()=="" or nxt.strip().startswith("#")):
                for x in missing: out.append(f'  - "{x}"')
                done=True; inblk=False
    open(cfg,"w").write("\n".join(out)+"\n"); print("registered:", missing)
else:
    print("all 20 ids already registered")
PY

TARGETS=""
for d in $IDS; do for m in $METHODS; do TARGETS="$TARGETS results/real_data/${d}-${m}.rds"; done; done
NT=$(echo $TARGETS | wc -w)
echo "launching timing run: 20 lineages x $(echo $METHODS|wc -w) methods = $NT targets, cores=$CORES"
date
snakemake -s Snakefile.py --cores "$CORES" --latency-wait 90 --rerun-incomplete --keep-going --nolock $TARGETS
echo "[timing run exit $?]"; date
echo "benchmark files for 20 ids: $(for d in $IDS; do ls benchmarks/${d}-*_benchmark.txt 2>/dev/null; done | wc -l)"
echo "[timing run done]"
