#!/bin/bash
# Download the prepared datasets of the marker-gene benchmark (Pullin and McCarthy; Zenodo record 10447426)
# into data/real_data/.
cd /path/to/cpu_server/analysis/marker_methods/data/real_data
for d in lawlor ss3_pbmc citeseq mesenchymal endothelial dominguez_conde ren stephenson yoshida zhao; do
  if [ -s "$d.rds" ]; then echo "$d already present"; continue; fi
  echo "fetching $d ..."
  curl -fSL -o "$d.rds" "https://zenodo.org/records/10447426/files/$d.rds?download=1" || echo "FAILED: $d"
done
echo "--- final inventory ---"; ls -la *.rds | awk "{print \$9, \$5}"
