#!/bin/bash
# Download Hickey 2023 colon MEX (Zenodo 7675308, md5-checked) and Maitra 2023 combined matrix (GEO GSE213982).
FS=/path/to/research/analysis/subtype_benchmark/full_benchmark
cd $FS/data/dl/hickey2023
while read url name size md5; do
  for try in 1 2 3; do
    if [ -f "$name" ] && [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; then break; fi
    curl -sSL --retry 5 --retry-delay 10 -o "$name" "$url"
  done
  [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ] && echo "ok $name" || echo "MD5FAIL $name"
done < $FS/added_lineages/hickey_colon_files.txt
cd $FS/data/dl/maitra2023
G=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE213nnn/GSE213982/suppl
for f in GSE213982_combined_counts_matrix_cells_columns.csv.gz GSE213982_combined_counts_matrix_genes_rows.csv.gz GSE213982_combined_counts_matrix.mtx.gz; do
  curl -sSL --retry 5 --retry-delay 10 -o "$f" "$G/$f" && gzip -t "$f" && echo "ok $f" || echo "FAIL $f"
done
echo "[dl done]"
