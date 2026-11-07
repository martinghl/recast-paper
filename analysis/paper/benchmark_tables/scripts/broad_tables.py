"""Tables of the broad-population comparison (Supplementary Fig. 4a, b).  Nothing is recomputed: the stored per-population
results are joined.

  a  marker recall of the RECAST top 20 and of COSG, limma and Wilcoxon on 7 atlases (62 populations)
       RECAST: broad_populations/results_realcounts/recast_broad_marker_recall.csv (recast_panels.py; the RECAST
               selection on the integer counts of atlases/<atlas>_fullgene.h5ad)
       others: broad_populations/results/baseline_arm2.csv
  b  marker-independent specificity (AUROC and kNN F1 of the top 20) on 8 atlases, RECAST and every method
       RECAST: broad_populations/results_realcounts/recast_broad_specificity.csv
       others: broad_populations/results/baselines_<atlas>.csv

Out (paper/benchmark_tables/data/): broad_marker_recall.csv, broad_marker_recall_pooled.csv, broad_specificity_all_methods.csv
(code/supplementary_figure04/data/broad_specificity.csv keeps the methods drawn).
"""
import os

import pandas as pd

ROOT = "/path/to/research/analysis"
D = f"{ROOT}/paper/benchmark_tables/data"
os.makedirs(D, exist_ok=True)

ATLASES_A = ["pbmc3k", "ren", "stephenson", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]
ATLASES_B = ["pbmc3k", "ren", "stephenson", "citeseq", "dominguez_conde", "yoshida", "ss3_pbmc", "lawlor"]
ATLAS_LABEL = {"pbmc3k": "pbmc3k", "ren": "Ren", "stephenson": "Stephenson",
               "dominguez_conde": "Dominguez-Conde", "yoshida": "Yoshida",
               "ss3_pbmc": "Smart-seq3 PBMC", "lawlor": "Lawlor", "citeseq": "CITE-seq"}
K = 20

# ---------------------------------------------------------------- a: marker recall
base = pd.read_csv(f"{ROOT}/broad_populations/results/baseline_arm2.csv")
rec = pd.read_csv(f"{ROOT}/broad_populations/results_realcounts/recast_broad_marker_recall.csv")
assert set(rec.dataset) == set(ATLASES_A) and (rec.method == "m0").all(), "RECAST rows incomplete"
METHODS_A = {"RECAST": ("m0", rec), "COSG": ("cosg", base),
             "limma": ("limma_trend", base), "Wilcoxon": ("scanpy_wilcoxon_rankby_raw", base)}
rows = []
for a in ATLASES_A:
    for disp, (m, src) in METHODS_A.items():
        v = src[(src.dataset == a) & (src.method == m)]["recall"]
        rows.append(dict(atlas=a, atlas_label=ATLAS_LABEL[a], method=disp,
                         mean_recall=float(v.mean()), n_clusters=int(v.shape[0])))
pd.DataFrame(rows).to_csv(f"{D}/broad_marker_recall.csv", index=False)
pooled = {disp: float(src[(src.method == m) & src.dataset.isin(ATLASES_A)]["recall"].mean())
          for disp, (m, src) in METHODS_A.items()}
pd.DataFrame([dict(method=k, pooled_recall=v) for k, v in pooled.items()]).to_csv(
    f"{D}/broad_marker_recall_pooled.csv", index=False)
print("a  marker recall pooled over the populations:", {k: round(v, 3) for k, v in pooled.items()})

# ---------------------------------------------------------------- b: marker-independent specificity
rec3 = pd.read_csv(f"{ROOT}/broad_populations/results_realcounts/recast_broad_specificity.csv")
rows = []
for a in ATLASES_B:
    bl = pd.read_csv(f"{ROOT}/broad_populations/results/baselines_{a}.csv")
    m0 = rec3[(rec3.dataset == a) & (rec3.method == "m0") & (rec3.k == K)]
    rows.append(dict(atlas=a, method="RECAST", src="m0", k=K, auroc=float(m0.auroc.iloc[0]), knn=float(m0.knn.iloc[0])))
    for _, r in bl[bl.k == K].iterrows():
        rows.append(dict(atlas=a, method=r.method, src="baseline", k=K, auroc=float(r.auroc), knn=float(r.knn)))
spec = pd.DataFrame(rows)
spec.to_csv(f"{D}/broad_specificity_all_methods.csv", index=False)
mean = spec.groupby(["method", "atlas"]).auroc.mean().groupby("method").mean().sort_values(ascending=False)
print(f"b  AUROC@{K}, mean over {len(ATLASES_B)} atlases: RECAST {mean['RECAST']:.3f}, COSG {mean['cosg']:.3f}, "
      f"limma {mean['limma_trend']:.3f}")
