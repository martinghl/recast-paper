#!/bin/bash
# Every scorer without labels on every panel variant of every task for one experiment, RECAST and SIGnature from the
# exported weights, then the evaluation (CPU server).
#   usage: [SKIP_W=1] nohup ./run_variants.sh B 120 8 > logs/run_B.log 2>&1 &
#   args: experiment, parallel jobs, number of variant chunks for the 8 signaturescoring methods (slow).
EXP=$1; P=${2:-120}; NC=${3:-8}
cd /path/to/cpu_server/program_scoring/added_genes
PY=/path/to/envs/scoring_python/bin/python; PYB=/path/to/envs/decoupler/bin/python
RS=/path/to/envs/scoring_r/bin/Rscript
SSC="ANS Seurat Seurat_AG Seurat_LVG Scanpy Jasmine_LH Jasmine_OR UCell"
mkdir -p scores_$EXP logs/$EXP
TASKS=$(ls variants_$EXP/*.json | xargs -n1 basename | sed 's/.json$//')
{
for t in $TASKS; do      # slow chunked jobs first so they do not straggle at the end
  for m in $SSC; do for i in $(seq 0 $((NC-1))); do echo "C $t $m $i"; done; done
done
for t in $TASKS; do
  for m in AUCell ssGSEA GSVA zscore VAM singscore; do echo "R $t $m 0"; done
  for m in expr_mean SCSE; do echo "P $t $m 0"; done
  echo "U $t decoupleR_ULM 0"; [ -z "$SKIP_W" ] && echo "W $t RECAST_SIGnature 0"
done
} > logs/$EXP/jobs.txt
run_one() {
  kind=$1; t=$2; m=$3; i=$4; V=variants_$EXP/$t.json; O=scores_$EXP/${t}__$m
  export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  case $kind in
    C) [ -s $O.npz ] || [ -s ${O}__c${i}of$NC.npz ] || $PY score_variants_python.py $t $m $V ${O}__c${i}of$NC.npz $i/$NC > logs/$EXP/${t}__${m}__c$i.log 2>&1 ;;
    R) [ -s $O.csv.gz ] || $RS score_variants_r.R $t $m $V $O.csv.gz > logs/$EXP/${t}__$m.log 2>&1 ;;
    P) [ -s $O.npz ] || $PY score_variants_python.py $t $m $V $O.npz > logs/$EXP/${t}__$m.log 2>&1 ;;
    U) [ -s $O.npz ] || $PYB score_variants_python.py $t $m $V $O.npz > logs/$EXP/${t}__$m.log 2>&1 ;;
    W) [ -s $O.npz ] || $PY score_variants_recast.py $t $V $O.npz > logs/$EXP/${t}__$m.log 2>&1 ;;
  esac
  echo "$(date +%T) finished $kind $t $m $i rc=$?"
}
export -f run_one; export EXP NC PY PYB RS
xargs -P $P -L 1 bash -c 'run_one $0 $1 $2 $3' < logs/$EXP/jobs.txt
echo "$(date +%T) scoring jobs returned; merging chunks"
for t in $TASKS; do for m in $SSC; do
  [ -s scores_$EXP/${t}__$m.npz ] || $PY merge_chunks.py variants_$EXP/$t.json scores_$EXP/${t}__$m $NC || echo "MERGE FAILED $t $m"
done; done
[ -n "$SKIP_W" ] && { echo "$(date +%T) SKIP_W set: stop before RECAST/SIGnature and evaluation"; exit 0; }
echo "$(date +%T) evaluating"
for t in $TASKS; do echo $t; done | xargs -P 28 -I{} bash -c "NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 $PY evaluate.py {} $EXP > logs/$EXP/eval_{}.log 2>&1; echo eval {} rc=\$?"
echo "$(date +%T) ALL DONE"
