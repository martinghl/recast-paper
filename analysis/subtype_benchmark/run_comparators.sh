#!/bin/bash
# The 51 comparator methods of the marker_methods harness on the integer-count lineage exports (*_v2 SCEs).
# Run on the CPU server:  nohup ./run_comparators.sh 96 > logs/comparators.log 2>&1 < /dev/null &
#
# 51 methods x 41 datasets.  The dataset ids carry a "_v2" suffix (results/real_data/<id>_v2-<method>.rds).  Phase A runs
# the 21 lineages of the marker benchmark (PHASE_A.txt, smallest first), phase B the other 20.
set -uo pipefail
cd /path/to/cpu_server/analysis/marker_methods
export PATH=/path/to/envs/marker_methods/bin:$PATH
export RETICULATE_PYTHON=/path/to/envs/marker_methods/bin/python
mkdir -p logs

STG=data/fine_state_export_v2
A="$(cat $STG/PHASE_A.txt)"          # the 21 lineages of the marker benchmark, smallest first
B="$(cat $STG/PHASE_B.txt)"          # the other 20
METHODS="cepo cosg edger_glm edger_ql glmgampoi limma_trend limma_voom lm_two_sample logfc_abs logfc_raw random scanpy_t_rankby_abs scanpy_t_rankby_raw scanpy_tover_rankby_abs scanpy_tover_rankby_raw scanpy_wilcoxon_rankby_abs scanpy_wilcoxon_rankby_raw scanpy_wilcoxontiecorrect_rankby_abs scanpy_wilcoxontiecorrect_rankby_raw scran_findMarkers_binom_all scran_findMarkers_binom_any scran_findMarkers_binom_some scran_findMarkers_t_all scran_findMarkers_t_any scran_findMarkers_t_some scran_findMarkers_wilcox_all scran_findMarkers_wilcox_any scran_findMarkers_wilcox_some scran_scoreMarkers_max_AUC scran_scoreMarkers_max_logFC_cohen scran_scoreMarkers_max_logFC_detected scran_scoreMarkers_mean_AUC scran_scoreMarkers_mean_logFC_cohen scran_scoreMarkers_mean_logFC_detected scran_scoreMarkers_median_AUC scran_scoreMarkers_median_logFC_cohen scran_scoreMarkers_median_logFC_detected scran_scoreMarkers_min_AUC scran_scoreMarkers_min_logFC_cohen scran_scoreMarkers_min_logFC_detected scran_scoreMarkers_rank_AUC scran_scoreMarkers_rank_logFC_cohen scran_scoreMarkers_rank_logFC_detected seurat_LR seurat_MAST seurat_bimod seurat_negbinom seurat_poisson seurat_roc seurat_t seurat_wilcox"

CORES="${1:-96}"

for d in $A $B; do
  [ -f "data/real_data/${d}_v2.rds" ] || { echo "FATAL missing SCE ${d}_v2.rds"; exit 2; }
done

# --- register the _v2 ids in config.yaml (idempotent) ---
python3 - <<'PY'
cfg = "config.yaml"
ids = [l.strip() + "_v2" for l in open("data/fine_state_export_v2/V2_IDS.txt") if l.strip()]
s = open(cfg).read()
new = [i for i in ids if f'"{i}"' not in s]
if not new:
    print("all _v2 ids already registered"); raise SystemExit
lines = s.splitlines(); out = []; inblk = False; done = False
for i, l in enumerate(lines):
    out.append(l)
    if l.strip().startswith("all_data_ids:"):
        inblk = True; continue
    if inblk and not done:
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if not (nxt.strip().startswith("- ") or nxt.strip() == "" or nxt.strip().startswith("#")):
            out += [f'  - "{x}"' for x in new]
            done = True; inblk = False
assert done, "could not find end of all_data_ids block"
open(cfg, "w").write("\n".join(out) + "\n")
print(f"registered {len(new)} _v2 ids in config.yaml")
PY

run_phase () {
  local name="$1"; shift
  local T=""
  for d in "$@"; do for m in $METHODS; do T="$T results/real_data/${d}_v2-${m}.rds"; done; done
  echo "[$name] $# ids x $(echo $METHODS | wc -w) methods = $(echo $T | wc -w) targets, cores=$CORES"
  date
  snakemake -s Snakefile.py --cores "$CORES" --latency-wait 90 --rerun-incomplete --keep-going --nolock $T
  echo "[$name exit $?]"
  date
  for d in "$@"; do echo "  $d: $(ls results/real_data/${d}_v2-*.rds 2>/dev/null | wc -l)/51"; done
}

run_phase A $A
echo "phase A (marker benchmark lineages) done"
run_phase B $B
echo "all comparators done"
