#!/bin/bash
# scMarkerGene on the added lineages (local GPU), with dedicated_tools/run_scmarkergene.py and the added-lineage
# inputs and outputs.
# Usage: run_scmarkergene_added.sh <lineage> ...   (give each worker a disjoint list: the skip test only sees
# finished lineages, and two workers on one lineage would share one scmg_work/<lin> tree).
TOOLS=/path/to/research/analysis/subtype_benchmark/dedicated_tools
D=/path/to/research/analysis/subtype_benchmark/full_benchmark/data
PY=/path/to/envs/analysis/bin/python
cd $TOOLS
for lin in "$@"; do
  [ -f "$D/dedicated_tools/genesets200/${lin}__scmarkergene.csv" ] && { echo "[skip] $lin"; continue; }
  echo "=== $lin $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 $PY run_scmarkergene.py --lineage "$lin" --python $PY \
      --sce $D/sce_export_v2 --work $D/dedicated_tools/scmg_work --out $D/dedicated_tools/genesets200 --timing $D/dedicated_tools/timing \
      || echo "[FAIL] $lin"
done
echo "WORKER DONE $(date '+%F %T')"
