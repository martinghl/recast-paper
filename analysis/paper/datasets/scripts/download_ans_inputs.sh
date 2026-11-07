set -u
AR=/path/to/research/analysis/ans_replication
PY=/path/to/envs/SC/bin/python
echo "[0] install phate into SC"
$PY -m pip install -q phate 2>&1 | tail -2
echo "[1] Jerber neural-diff (4.9 GB) via curl"
cd $AR/data/neural_diff
if [ ! -s all_timepoints_subsampled.h5 ]; then
  curl -sL -o all_timepoints_subsampled.h5 "https://zenodo.org/api/records/4333872/files/all_timepoints_subsampled.h5/content"
fi
echo "  md5: $(md5sum all_timepoints_subsampled.h5 | cut -d' ' -f1) (expect 4695a336a12fcafb7152cd6a2e8a770f)"
ls -la all_timepoints_subsampled.h5
gd() {  # id output : gdown by id with 3 retries
  local id="$1" out="$2" i
  for i in 1 2 3; do
    $PY -c "import gdown; gdown.download(id='$id', output='$out', quiet=True)" 2>/dev/null
    if [ -s "$out" ] && ! head -c 20 "$out" | grep -qi "<!DOCTYPE\|<html"; then echo "  ok $out ($(du -h "$out"|cut -f1))"; return 0; fi
    echo "  retry $i for $out"; sleep 8
  done
  echo "  FAILED $out"
}
echo "[2] preprocessed h5ad files of the ANS benchmark (gdown by id)"
cd $AR/data/preprocessed
gd 1nb1dnH6bEWBhB3HZc46fKmqogKi4FhQi pp_pbmc_b_subtypes.h5ad
sleep 5
gd 1XxtbbocDubhDx_18w71V5fbT_xrMJuXc pp_luad_kim_malignant.h5ad
echo "[3] signatures"
cd $AR/data/annotations
gd 1a3Uqky2VZxCxLvGI-soCTUp3lijrfrx7 DE_by_celltype.csv
sleep 5
gd 1DNakekjPt4EhSKN3pWPLn0gLdVYPMqJb kim_3.csv
echo "[downloads done]"
ls -la $AR/data/preprocessed $AR/data/annotations
