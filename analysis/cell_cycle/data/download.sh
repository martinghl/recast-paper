#!/bin/bash
# Download the source datasets of the cell-cycle comparisons: five developmental atlases from CELLxGENE and the
# FUCCI iPSC counts of GEO GSE121265.
set -u
CX=https://datasets.cellxgene.cziscience.com
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE121nnn/GSE121265/suppl
dl () { # url outfile
  if [ -s "$2" ]; then echo "SKIP $2 (exists)"; return; fi
  echo "GET  $2"
  curl -sL --retry 3 --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2" && echo "OK   $2 ($(du -m "$2"|cut -f1) MB)" || echo "FAIL $2"
}
dl $CX/485304df-2065-48a6-8c16-3ff536e21eee.h5ad fetal_gonad.h5ad
dl $CX/5cf26393-87e2-4946-8a97-2d42625856a3.h5ad dev_gut.h5ad
dl $CX/30626c44-7a90-40ed-8f54-116bfaebc857.h5ad cerebellum_dev.h5ad
dl $CX/027cb841-f726-45e9-8cea-ac5484c24474.h5ad thymus_dev.h5ad
dl $CX/4d7b34ed-d725-43eb-865f-87be6a467323.h5ad bonemarrow_dev.h5ad
dl $GEO/GSE121265_fucci-counts.txt.gz          fucci_ipsc_counts.txt.gz
dl $GEO/GSE121265_fucci-annotation.txt.gz      fucci_ipsc_annotation.txt.gz
dl $GEO/GSE121265_fucci-annotation-description.txt.gz fucci_ipsc_annotation_desc.txt.gz
echo "ALL DONE"
