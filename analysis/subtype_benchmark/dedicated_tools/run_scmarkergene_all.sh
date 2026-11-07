#!/bin/bash
# scMarkerGene over the 20 subtype lineages, smallest first.
# The GPU stage runs locally (CUDA_VISIBLE_DEVICES=1); the per-lineage attribution arrays are tens of GB and are deleted
# after stage 003 reads them, so the selection stage runs locally too (single-core minutes).  The CPU-heavy work (the
# four other tools on 20 lineages, and the AUROC) runs on the CPU server.
cd /path/to/research/analysis/subtype_benchmark/dedicated_tools
PY=/path/to/envs/analysis/bin/python
for lin in tf_b_ren tf_mono_steph tf_cd8_dom tf_mono_ren tf_b_yosh fs_nk_zhao tf_cd8_steph \
           tf_mono_dom tf_cd4_ren tf_cd8_ren tf_mono_yosh fs_gut_smillie tf_cd4_steph \
           tf_cd8_yosh fs_asc_zhao tf_cd4_dom fs_nk_pbmcbmn tf_cd4_yosh fs_lung_hlca; do
  if [ -f "genesets200/${lin}__scmarkergene.csv" ]; then echo "[skip] $lin"; continue; fi
  echo "=== $lin  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 $PY run_scmarkergene.py --lineage "$lin" --python $PY \
    || echo "[FAIL] $lin"
done
echo "ALL DONE $(date '+%F %T')"
