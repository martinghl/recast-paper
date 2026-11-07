set -e
PY=/path/to/envs/SC/bin/python
PC=/path/to/research/analysis/new_datasets; PD=/path/to/research/analysis/dataset_files
echo "[$(date +%H:%M:%S)] pelka: loading 10x h5 (1.12GB) + Epi subset + cap..."
$PY $PC/load_datasets.py pelka 2>&1 | grep -vE "FutureWarning|warn\(|UserWarning" | tail -6
echo "[$(date +%H:%M:%S)] pelka: dataset pipeline (align, embed, attribution, checks)..."
CUDA_VISIBLE_DEVICES=1 $PY $PC/dataset_pipeline.py --h5ad $PD/pelka_crc_epi/pelka_crc_epi_raw.h5ad \
  --id pelka_crc_epi --label-col label --markers $PC/markers_pelka_crc.json --out-dir $PD/pelka_crc_epi --gpu 1 2>&1 | grep -vE "FutureWarning|warn\(|UserWarning" | tail -8
echo "[pelka done]"
