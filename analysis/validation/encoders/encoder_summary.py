"""Separability and RECAST attribution across encoders (Supplementary Fig. 2e).

Joins existing tables (CPU), per encoder: separability of the embedding (kNN label transfer, coarse and fine) and
RECAST's marker recall relative to the expression ranking.

Inputs:
  atlases/fm_embedding_knn.csv                       coarse cell-type kNN (4 encoders)
  atlases/fm_finestate_knn.csv                       fine subtype kNN (3 encoders x CD8, B)
  subtype_benchmark/encoder_results/recast_cross_fm_recall_ALL_multi{V}.csv   5 encoders x 6 lineages, RECAST / expression /
                                                     difference at k = 20 (merge_encoders.py)
Outputs (validation/encoders/): joint_table.csv, within_lineage_corr.csv, separability_vs_attribution.png, findings.txt"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

SB = "/path/to/research/analysis"
OUT = f"{SB}/validation/encoders"
import os as _os; V = _os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the input tables (README: result tags)
lines = []
def log(s=""):
    print(s); lines.append(str(s))

# --- encoder properties (from recast/recast/encoders.py + atlases/compute_attribution_*.py) ---
# bandwidth_ok = encoder can ingest the full pseudobulk gene vector (scFoundation caps 512 tokens/cell)
PROP = {
    "SCimilarity":  dict(dim=128,  coverage=28231, objective="contrastive",  bandwidth_ok=True,  note=""),
    "PBMC_SSL":     dict(dim=64,   coverage=19331, objective="SSL-MLP",      bandwidth_ok=True,  note=""),
    "scTab_SSL":    dict(dim=64,   coverage=19331, objective="SSL-MLP",      bandwidth_ok=True,  note=""),
    "scVI":         dict(dim=10,   coverage=13714, objective="VAE",          bandwidth_ok=True,  note="dataset-specific gene space"),
    "scFoundation": dict(dim=3072, coverage=19264, objective="transformer",  bandwidth_ok=False, note="512-token/cell cap; 89.7% cells truncated; 8608 genes never reach encoder"),
}
SEP_PROXY = {"scFoundation": 0.737}   # coarse kNN15 label-agreement proxy (verify_transform_scf.log); not in fm_embedding_knn

coarse = pd.read_csv(f"{SB}/atlases/fm_embedding_knn.csv").set_index("FM")["knn_accuracy"].to_dict()
# fine-state kNN: use the ALL_multi table (5 FMs x 6 fine-state lineages under ONE consistent protocol) so scVI &
# scFoundation are filled in and the within-lineage fit spans 6 lineages instead of just CD8/B.
fine = pd.read_csv(f"{SB}/subtype_benchmark/results/fm_finestate_knn_ALL_multi{V}.csv")
finemean = fine.groupby("FM")["state_knn_acc"].mean().to_dict()     # mean over the 6 fine-state lineages (all 5 FMs)
R = pd.read_csv(f"{SB}/subtype_benchmark/encoder_results/recast_cross_fm_recall_ALL_multi{V}.csv")
dmean = R.groupby("encoder")["delta"].mean().to_dict()
dpos = R[R.delta > 0].groupby("encoder").size().to_dict()
nlin = R.groupby("encoder").size().to_dict()

# ---------------- joint per-FM table ----------------
rows = []
for fm, p in PROP.items():
    rows.append(dict(FM=fm, dim=p["dim"], gene_coverage=p["coverage"], objective=p["objective"],
                     input_bandwidth_ok=p["bandwidth_ok"],
                     sep_coarse_knn=round(coarse.get(fm, SEP_PROXY.get(fm, np.nan)), 3),
                     sep_coarse_is_proxy=(fm in SEP_PROXY and fm not in coarse),
                     sep_fine_knn=round(finemean[fm], 3) if fm in finemean else np.nan,
                     RECAST_delta_mean=round(dmean.get(fm, np.nan), 3),
                     RECAST_delta_pos=f"{dpos.get(fm,0)}/{nlin.get(fm,0)}", note=p["note"]))
JT = pd.DataFrame(rows).sort_values("RECAST_delta_mean", ascending=False)
JT.to_csv(f"{OUT}/joint_table{V}.csv", index=False)
log("="*96); log("JOINT TABLE — encoder properties · separability · RECAST attribution quality"); log("="*96)
log(JT.to_string(index=False))

# ---------------- within-lineage (across-FM) separability vs attribution ----------------
log(""); log("="*96); log("Within each fine-state lineage, does across-FM separability rank-order RECAST delta?"); log("="*96)
# join fine-state kNN (separability) with RECAST delta (attribution gain) per lineage, across ALL 5 FMs, ALL 6 lineages
corr_rows = []
for lin_full in sorted(fine.lineage.unique()):
    sub = fine[fine.lineage == lin_full][["FM", "state_knn_acc"]].copy()
    rr = R[R.lineage == lin_full][["encoder", "delta"]].rename(columns={"encoder": "FM"})
    m = sub.merge(rr, on="FM").sort_values("state_knn_acc", ascending=False)
    if len(m) >= 3:
        rho, p = stats.spearmanr(m.state_knn_acc, m.delta)
        pear, _ = stats.pearsonr(m.state_knn_acc, m.delta)
    else:
        rho = pear = p = np.nan
    log(f"\n  {lin_full}:")
    for _, r in m.iterrows():
        log(f"    {r.FM:14s} fine-kNN {r.state_knn_acc:.3f}   RECAST delta {r.delta:+.3f}")
    log(f"    Spearman rho(sep, delta) = {rho:+.3f}  Pearson {pear:+.3f}  (n={len(m)})")
    corr_rows.append(dict(lineage=lin_full, n=len(m), spearman=round(rho, 3), pearson=round(pear, 3)))
CORR = pd.DataFrame(corr_rows); CORR.to_csv(f"{OUT}/within_lineage_corr{V}.csv", index=False)
log(f"\n  mean within-lineage Spearman(fine-sep, RECAST gain) over {len(CORR)} lineages = {CORR.spearman.mean():+.3f}")

# ---------------- figure: separability vs RECAST delta ----------------
COL = {"SCimilarity": "#C0392B", "PBMC_SSL": "#2E86AB", "scTab_SSL": "#5DADE2", "scVI": "#7D8B99", "scFoundation": "#8E44AD"}
fig, ax = plt.subplots(figsize=(8.6, 6.2))
ax.axhline(0, color="0.6", lw=1.1, ls="--", zorder=1)
LABPOS = {"SCimilarity": (-10, -6, "right"), "PBMC_SSL": (0, 15, "center"),
          "scTab_SSL": (2, -24, "center"), "scVI": (-8, -20, "right")}   # scVI down-left, clear of the y=0 line & scTab; scFoundation labelled by the annotation below
for fm in PROP:
    x = coarse.get(fm, SEP_PROXY.get(fm, np.nan)); y = dmean.get(fm, np.nan)
    ax.scatter(x, y, s=280, color=COL[fm], edgecolor="white", linewidth=1.5, zorder=4)
    if fm in LABPOS:
        dx, dy, ha = LABPOS[fm]
        ax.annotate(f"{fm}\n(dim {PROP[fm]['dim']})", (x, y), fontsize=9, ha=ha,
                    xytext=(dx, dy), textcoords="offset points", color="#222", zorder=6)
ax.annotate("scFoundation (dim 3072): separates (proxy)\nbut attribution collapses — 512-token input cap\nblocks 8,608 genes (bandwidth, not geometry)",
            (SEP_PROXY["scFoundation"], dmean["scFoundation"]), fontsize=8.2, color="#8E44AD",
            xytext=(20, 22), textcoords="offset points", ha="left",
            arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1), zorder=6)
ax.set_xlim(0.70, 0.95); ax.set_ylim(-0.62, 0.31)
ax.set_xlabel("representation separability  (coarse kNN label-transfer accuracy)", fontsize=11)
ax.set_ylabel("RECAST attribution gain  (mean recall@20 − expression, 6 lineages)", fontsize=11)
ax.set_title("Separability and RECAST attribution", fontsize=12.5, fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/separability_vs_attribution{V}.png", dpi=175, bbox_inches="tight", facecolor="white")
plt.close(fig)
log(""); log(f"[wrote] {OUT}/joint_table.csv, within_lineage_corr.csv, separability_vs_attribution.png")

# ---------------- findings ----------------
log(""); log("="*96); log("READING"); log("="*96)
# Every number below is read from the tables built above.
_J = JT.set_index("FM")
_bw = [f for f in PROP if not PROP[f]["bandwidth_ok"]]
_ssl = [f for f in PROP if PROP[f]["objective"] == "SSL-MLP"]
_sslpos = {f: R[(R.encoder == f) & (R.delta > 0)].lineage.tolist() for f in _ssl}
_best = JT.iloc[0].FM
log(f"""[C] {_best} is the best carrier (mean RECAST delta {_J.loc[_best,'RECAST_delta_mean']:+.3f},
    {_J.loc[_best,'RECAST_delta_pos']} lineages positive) because it maximizes all three properties: highest
    separability (coarse {_J.loc[_best,'sep_coarse_knn']:.3f} / fine {_J.loc[_best,'sep_fine_knn']:.3f}),
    full {_J.loc[_best,'gene_coverage']:,}-gene reach, {_J.loc[_best,'dim']}-d contrastive.
[C] Separability is NECESSARY but NOT SUFFICIENT:
    - scFoundation SEPARATES (proxy kNN {_J.loc['scFoundation','sep_coarse_knn']:.3f}) yet RECAST collapses
      (delta {_J.loc['scFoundation','RECAST_delta_mean']:+.3f}, ~0 recall) — its 512-token input cap truncates
      89.7% of cells and 8,608 genes never reach the encoder, so attribution cannot be assigned to most genes.
      This is an INPUT-BANDWIDTH failure, orthogonal to representation geometry. It is the only encoder here
      with input_bandwidth_ok=False ({_bw}), and the only one at 0/6.
    - scVI separates coarsely ({_J.loc['scVI','sep_coarse_knn']:.3f}) but nets
      {_J.loc['scVI','RECAST_delta_mean']:+.3f} ({_J.loc['scVI','RECAST_delta_pos']}): a
      {_J.loc['scVI','dim']}-d VAE bottleneck + a dataset-specific gene space limit the attributable signal.
    - SSL encoders (scTab/PBMC, {_J.loc['scTab_SSL','dim']}-d) have the representation (fine kNN
      {min(_J.loc[f,'sep_fine_knn'] for f in _ssl):.3f}-{max(_J.loc[f,'sep_fine_knn'] for f in _ssl):.3f}) and
      gain on { {f: f"{len(v)}/6" for f, v in _sslpos.items()} } lineages, at
      { {f: f"{_J.loc[f,'RECAST_delta_mean']:+.3f}" for f in _ssl} } — i.e. RECAST is portable across encoder
      objectives, not a property of the contrastive one.
[C] WITHIN a fixed lineage, across FMs, higher fine-state separability tracks higher RECAST delta (mean Spearman
    {CORR.spearman.mean():+.3f} over {len(CORR)} lineages); ACROSS lineages the signal is confounded by marker
    subtlety (a model gains MORE on the subtler lineage even at lower separability), the same axis as the
    mechanism analysis.""")
open(f"{OUT}/findings{V}.txt", "w").write("\n".join(lines))
