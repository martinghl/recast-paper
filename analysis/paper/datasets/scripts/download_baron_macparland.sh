set -u
PD=/path/to/research/analysis/dataset_files
log(){ echo "[$(date +%H:%M:%S)] $*"; }
# Baron pancreas (28MB) + MacParland liver (103MB) — small, parallel
( cd $PD/baron_pancreas_gse84133 && curl -sSL -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84133/suppl/GSE84133_RAW.tar && tar xf GSE84133_RAW.tar && log "baron done $(ls *human*.csv.gz | wc -l) human csv" ) &
( cd $PD/macparland_liver_gse115469 && B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115469/suppl && curl -sSL -O "$B/GSE115469_Data.csv.gz" && curl -sSL -O "$B/GSE115469_CellClusterType.txt.gz" && log "macparland done $(du -h GSE115469_Data.csv.gz|cut -f1)" ) &
# Pelka CRC (1.12GB) — the big one
( cd $PD/pelka_crc_gse178341 && B=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178341/suppl && curl -sSL -O "$B/GSE178341_crc10x_full_c295v4_submit.h5" && curl -sSL -O "$B/GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz" && curl -sSL -O "$B/GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz" && log "pelka done $(du -h GSE178341_crc10x_full_c295v4_submit.h5|cut -f1)" ) &
wait
log "[downloads done]"
du -sh $PD/*/ 2>/dev/null
