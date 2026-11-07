#!/bin/bash
# Inputs of the benchmark scoring (full_benchmark/scoring) from the four dedicated marker-selection tools on the
# 20 original lineages: gene lists, selection times and the per-state AUROC.
#   1. assemble the per-subtype scGeneFit parts on the CPU server and copy gene lists and timing back
#   2. merge the tools' gene lists into fs_genesets_v2_plus / fs_genesets200_v2_plus (merge_gene_lists.py)
#   3. timing rows (add_timing.py -> score_out/selection_time_per_cell.csv)
#   4. per-state AUROC on the CPU server, where the attribution h5ad files are (subtype_auroc.py), copied back to
#      score_out/fs_cm_arm3_perstate_<lineage>_v3x.csv
# STALL: elapsed single-thread seconds of the two scGeneFit subtypes whose linear program did not finish (a lower
# bound on their cost); full_benchmark/scoring/fill_unfinished.py scores those subtypes as the worst value.
set -e
cd /path/to/research/analysis/subtype_benchmark/dedicated_tools
SB=/path/to/research/analysis
FSB=$SB/subtype_benchmark
RB=/path/to/cpu_server/analysis/subtype_benchmark
R=cpu_server
PY=/path/to/envs/analysis/bin/python
export NUMPY_MADVISE_HUGEPAGE=0
mkdir -p score_out
STALL=${STALL:-0}

echo "### 1. assemble scGeneFit from its per-subtype parts, then copy gene lists and timing back"
ssh $R "cd /path/to/cpu_server/dedicated_tools && /path/to/envs/dedicated_tools/bin/python assemble_scgenefit.py \
  --allow-partial --stall-seconds $STALL"
rsync -a $R:/path/to/cpu_server/dedicated_tools/genesets200/ genesets200/
rsync -a $R:/path/to/cpu_server/dedicated_tools/timing/ timing/
echo "    gene lists: $(ls genesets200/*.csv | wc -l)   timing: $(ls timing/*.json | wc -l)"

echo "### 2. merge gene-list directories"
$PY merge_gene_lists.py

echo "### 3. timing rows"
$PY add_timing.py

echo "### 4. per-state AUROC on the CPU server"
rsync -a --delete $FSB/fs_genesets_v2_plus/ $R:$RB/fs_genesets_v2_plus/
rsync -a $FSB/subtype_auroc.py $R:$RB/
ssh $R "cd $RB && GENESETS_DIR=$RB/fs_genesets_v2_plus RECAST_RESULT_TAG=_v3x OMP_NUM_THREADS=8 \
  /path/to/envs/marker_methods/bin/python subtype_auroc.py" | tail -3
rsync -a --include='fs_cm_arm3_perstate_*_v3x.csv' --exclude='*' $R:$RB/results_remote/ score_out/
echo "    per-lineage AUROC files: $(ls score_out/fs_cm_arm3_perstate_*_v3x.csv | wc -l)"
