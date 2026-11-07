#!/bin/bash
# Run new_datasets/dataset_pipeline.py (alignment to the encoder's genes, reference-free per-cell IG, quality
# gate) on each added dataset's raw h5ad.
# Usage: run_dataset_pipeline.sh <id> [<id> ...]   (ids are the raw/<id>/<id>_raw.h5ad stems)
PY=/path/to/envs/SC/bin/python
PC=/path/to/research/analysis/new_datasets
PROV=/path/to/research/analysis/paper/datasets/scripts
FS=/path/to/research/analysis/subtype_benchmark/full_benchmark
declare -A MK=( [rustam2023]=$PROV/lung_markers.json [natri2024]=$PROV/lung_markers.json
                [kong2023ti]=$PROV/gut_markers.json [hickey2023colon]=$PROV/gut_markers.json [burclaff2022colon]=$PROV/gut_markers.json
                [rexach2024]=$FS/data/new_datasets/brain_markers.json [gittings2023]=$FS/data/new_datasets/brain_markers.json [maitra2023]=$FS/data/new_datasets/brain_markers.json )
for id in "$@"; do
  echo "[run] $id  markers=${MK[$id]}  $(date)"
  CUDA_VISIBLE_DEVICES=0 $PY -I $PC/dataset_pipeline.py --h5ad $FS/data/raw/$id/${id}_raw.h5ad --id $id --label-col label \
      --markers ${MK[$id]} --out-dir $FS/data/new_datasets --gpu 0 > $FS/data/logs/dataset_$id.log 2>&1
  echo "[run] $id exit=$?  $(grep -E 'TRUST GATE|AUROC =|===>' $FS/data/logs/dataset_$id.log | tr '\n' ' ')"
done
echo "[done]"
