#!/bin/bash
# The four CPU marker-selection tools on the added lineages, on the CPU server (env dedicated_tools): NS-Forest /
# scMAGS / RankCorr per lineage (run_tools.py) and scGeneFit per subtype (run_scgenefit.py), one thread each, all
# lineages in parallel; then the scGeneFit parts are assembled per lineage (assemble_scgenefit.py).  The code is the
# copy of subtype_benchmark/dedicated_tools on the CPU server; inputs and outputs are in the added-lineage workspace.
# Usage: run_tools_added.sh <lineage> ...
#   RP_SGF_STALL   wall-time cap per scGeneFit subtype in seconds (default 24053, the cap of the original lineages;
#                  full_benchmark/scoring/fill_unfinished.py scores subtypes stopped at the cap)
set -uo pipefail
CODE=/path/to/cpu_server/dedicated_tools
WORK=/path/to/cpu_server/added_lineages/dedicated_tools
PY=/path/to/envs/dedicated_tools/bin/python
LINS="$*"
[ -n "$LINS" ] || { echo "FATAL no lineage given"; exit 2; }
STALL="${RP_SGF_STALL:-24053}"
cd "$WORK" || exit 2
mkdir -p genesets200 timing sgf_parts logs sgf_logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "[tools] $(date '+%F %T') lineages: $LINS | stall cap ${STALL}s"

for lin in $LINS; do
  [ -f "sce_export_v2/${lin}_labels.txt" ] || { echo "FATAL missing sce_export_v2/${lin}_labels.txt"; exit 2; }
  $PY $CODE/run_tools.py --sce "$WORK/sce_export_v2" --out "$WORK/genesets200" --timing "$WORK/timing" \
      --methods nsforest,scmags,rankcorr --lineages "$lin" > "logs/cpu_${lin}.log" 2>&1 &
  n=$(sort -u "sce_export_v2/${lin}_labels.txt" | wc -l)
  for ((i = 0; i < n; i++)); do
    timeout "${STALL}s" $PY $CODE/run_scgenefit.py --sce "$WORK/sce_export_v2" --part "$WORK/sgf_parts" \
        --lineage "$lin" --cluster-index "$i" > "sgf_logs/${lin}_${i}.log" 2>&1 &
  done
done
wait
echo "[tools] $(date '+%F %T') selections finished; assembling scGeneFit"
# assemble_scgenefit.py loops over its own lineage list (LIN20); here the list is set to the lineages given.
$PY - "$CODE" "$WORK" "$STALL" $LINS <<'PYEOF'
import sys
code, dedicated_tools, stall, lins = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path.insert(0, code)
import assemble_scgenefit as m
m.LIN20 = list(lins)
sys.argv = ["assemble_scgenefit.py", "--part", f"{dedicated_tools}/sgf_parts", "--sce", f"{dedicated_tools}/sce_export_v2",
            "--out", f"{dedicated_tools}/genesets200", "--timing", f"{dedicated_tools}/timing", "--allow-partial", "--stall-seconds", stall]
m.main()
PYEOF

rc=0
for lin in $LINS; do
  for m in nsforest scmags rankcorr scgenefit; do
    [ -s "genesets200/${lin}__${m}.csv" ] || { echo "MISSING genesets200/${lin}__${m}.csv"; rc=1; }
    [ -s "timing/${lin}__${m}.json" ] || { echo "MISSING timing/${lin}__${m}.json"; rc=1; }
  done
done
echo "[tools done rc=$rc] $(date '+%F %T')"
exit $rc
