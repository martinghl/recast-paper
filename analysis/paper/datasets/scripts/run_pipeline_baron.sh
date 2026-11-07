set -e
PY=/path/to/envs/SC/bin/python
PC=/path/to/research/analysis/new_datasets
echo "[baron] building raw h5ad..."
$PY $PC/load_datasets.py baron
echo "[baron] running the dataset pipeline (align, embed, attribution, checks) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PY $PC/dataset_pipeline.py \
  --h5ad $PC/dataset_files/baron_pancreas/baron_pancreas_raw.h5ad \
  --id baron_pancreas --label-col label \
  --markers $PC/markers_baron_pancreas.json --gpu 0
echo "[baron done]"
