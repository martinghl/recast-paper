#!/bin/bash
# Inputs of the benchmark scoring as one folder per kind (scoring/in/panels, genesets50, genesets200, auroc): links
# to the files of the original lineages (subtype_benchmark/) and of the added lineages (full_benchmark/data/).
# Re-runnable; links only, nothing is copied or edited.
set -e
FSB=/path/to/research/analysis/subtype_benchmark
S=$FSB/full_benchmark/scoring/in
D=$FSB/full_benchmark/data
ORIGINAL=$(ls $FSB/results/fs_cm_recovery_uniq_*_v3.csv | sed 's/.*uniq_//; s/_v3.csv//')
ADDED=$(cat $D/sce_export_v2/V2_IDS.txt)
mkdir -p $S/panels $S/genesets50 $S/genesets200 $S/auroc
for lin in $ORIGINAL; do
  ln -sfn $FSB/panels/topN_${lin}_recast_M0.csv $S/panels/
  [ -f $FSB/panels/topN200_${lin}_recast_M0.csv ] && ln -sfn $FSB/panels/topN200_${lin}_recast_M0.csv $S/panels/
  ln -sfn $FSB/fs_genesets_v2_plus/${lin}.csv $S/genesets50/
  [ -f $FSB/fs_genesets200_v2_plus/${lin}.csv ] && ln -sfn $FSB/fs_genesets200_v2_plus/${lin}.csv $S/genesets200/
  [ -f $FSB/dedicated_tools/score_out/fs_cm_arm3_perstate_${lin}_v3x.csv ] && \
    ln -sfn $FSB/dedicated_tools/score_out/fs_cm_arm3_perstate_${lin}_v3x.csv $S/auroc/
done
for lin in $ADDED; do
  for k in topN topN200; do ln -sfn $D/panels/${k}_${lin}_recast_M0.csv $S/panels/; done
  [ -f $D/genesets50_plus/${lin}.csv ] && ln -sfn $D/genesets50_plus/${lin}.csv $S/genesets50/
  [ -f $D/genesets200_plus/${lin}.csv ] && ln -sfn $D/genesets200_plus/${lin}.csv $S/genesets200/
  [ -f $D/auroc/fs_cm_arm3_perstate_${lin}_v3x.csv ] && ln -sfn $D/auroc/fs_cm_arm3_perstate_${lin}_v3x.csv $S/auroc/
done
for k in panels genesets50 genesets200 auroc; do
  printf "%-12s %3d links, %3d dangling\n" $k "$(ls $S/$k | wc -l)" "$(find -L $S/$k -type l | wc -l)"
done
