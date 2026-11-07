#!/bin/bash
# Score the subtype marker benchmark into scoring/out/ and run the significance tests.
# Usage: bash run.sh   (after link_inputs.sh)
set -euo pipefail
S=/path/to/research/analysis/subtype_benchmark/full_benchmark/scoring
PY=/path/to/envs/analysis/bin/python
cd "$S"
./run_scoring_all.sh "$S/out"
$PY -I tests.py --score "$S/out"
