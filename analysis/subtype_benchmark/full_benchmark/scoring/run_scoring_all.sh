#!/bin/bash
# Scoring chain of the subtype marker benchmark (Fig. 3, Supplementary Fig. 3, Supplementary Data sheet 15):
# recall@20 -> marker rank@200 + contamination@200 -> benchmark table -> scGeneFit's unfinished subtypes -> summary.
# Inputs come from scoring/in/ (link_inputs.sh); the marker panels from ground_truth/panels/marker_panels.json.
# Usage: run_scoring_all.sh <out_dir>
set -e
S=/path/to/research/analysis/subtype_benchmark/full_benchmark/scoring
G=/path/to/research/analysis/subtype_benchmark/full_benchmark/ground_truth
PY=/path/to/envs/analysis/bin/python
OUTD=$1
mkdir -p "$OUTD"
export NUMPY_MADVISE_HUGEPAGE=0 PANELS_DIR=$S/in/panels UNIT_PANELS=$G/panels/marker_panels.json
cd $S
echo "### recall@20"
GENESETS_DIR=$S/in/genesets50 RECALL_OUT_DIR="$OUTD" $PY -I marker_recall_all.py > "$OUTD/recall.log" 2>&1
grep -A4 "FINE tier" "$OUTD/recall.log" | head -5
echo "### marker rank@200, contamination@200"
for s in marker_rank contamination; do
  GENESETS200_DIR=$S/in/genesets200 SCORE_OUT_DIR="$OUTD" RECALL_DIR="$OUTD" $PY -I $s.py > "$OUTD/$s.log" 2>&1
  head -1 "$OUTD/$s.log"
done
echo "### benchmark table"
RESULTS_DIR="$OUTD" AUROC_DIR=$S/in/auroc TABLES_DIR="$OUTD" $PY -I tables.py > "$OUTD/tables.log" 2>&1
tail -n 1 "$OUTD/tables.log"
echo "### scGeneFit's unfinished subtypes"
$PY -I fill_unfinished.py --score "$OUTD" > "$OUTD/fill.log" 2>&1
grep -E "inserted|covers" "$OUTD/fill.log"
echo "### summary"
$PY -I summary.py --score "$OUTD" | tee "$OUTD/summary.txt"
