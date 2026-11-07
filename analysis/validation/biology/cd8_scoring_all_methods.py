"""Per-cell program scores of RECAST and the other scorers for one lineage, with a shared embedding
(Fig. 6b, c).

For the chosen lineage:
  (a) a 2-D UMAP of the SCimilarity embedding of every cell (no labels),
  (b) the subtype label of every cell,
  (c) out-of-fold per-cell RECAST program scores (5 folds; recast_weighted_expr, score_programs' M1),
  (d) the per-cell scores of ANS / Seurat / Scanpy / UCell on the same curated panels (no labels, so scored once on
      all cells) through ans_baselines.score_all_methods,
  (e) per-program (one-vs-rest) AUROC for RECAST and every other scorer.
The functions are those of score_programs (load_dataset / build_candidate_params / m1_scores) and
ans_baselines.score_all_methods, so the numbers equal metrics_all.csv.

Default lineage: tf_cd8_dom (four CD8 T subtypes: Tcm/Naive, Tem/Temra, Tem/Trm, gdT).  GPU (SC env; contrastive_ig
loads SCimilarity).

  cd /path/to/research/analysis
  CUDA_VISIBLE_DEVICES=1 /path/to/envs/SC/bin/python \
      validation/biology/cd8_scoring_all_methods.py tf_cd8_dom"""
import os, sys, json, time
import numpy as np, pandas as pd

SB = "/path/to/research/analysis"
for p in (os.path.join(SB, "program_scoring"), os.path.join(SB, "attribution_methods"),
          SB, os.path.join(SB, "subtype_benchmark"), os.path.join(SB, "ans_replication")):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = os.path.join(SB, "validation", "biology")
NAME = sys.argv[1] if len(sys.argv) > 1 else "tf_cd8_dom"
# competitor field shown in the panels (a representative, readable subset of the 8 ANS-family scorers)
FIELD = ["ANS", "Seurat", "Scanpy", "UCell"]
def log(s=""): print(s, flush=True)

t0 = time.time()
log(f"[import] score_programs (loads SCimilarity on GPU) …")
import score_programs as FB          # noqa: E402
import ans_baselines as ANSB            # noqa: E402
import contrastive_ig                              # noqa: E402
import umap                             # noqa: E402
from sklearn.model_selection import StratifiedKFold           # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
import anndata as ad                    # noqa: E402
from scoring_benchmark.datasets import _go_gene_order          # noqa: E402  (GO order for AnnData var)

log(f"[load] {NAME} …")
X, y, cand, sig_idx, sig_genes, regime = FB.load_dataset(NAME)
K = len(cand)
counts = {c: int((y == c).sum()) for c in cand}
log(f"  X={X.shape}  regime={regime}  states({K})={cand}  counts={counts}")

# ---- shared embedding (label-free -> safe visualization coords across all cells) ---------
log("[embed] SCimilarity …")
Z = contrastive_ig.embed(X)                        # (n, 128)

# ---- RECAST: out-of-fold per-cell M1 scores (parameters from the training cells of each fold) ----
log("[score] RECAST 5-fold OOF M1 (recast_weighted_expr) …")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FB.CV_SEED)
S_recast = np.zeros((X.shape[0], K), "float32")
for i, (tr, te) in enumerate(skf.split(X, y)):
    params = FB.build_candidate_params(X[tr], y[tr], cand, sig_idx)
    S_recast[te] = FB.m1_scores(X[te], params, cand, sig_idx)
    log(f"  fold {i+1}/5  ntr={len(tr)} nte={len(te)}")

# ---- FIELD: conventional per-cell scores on the SAME curated panels (label-free -> once) -------
log(f"[score] field {FIELD} via ans_baselines.score_all_methods …")
GO = _go_gene_order()
A = ad.AnnData(X=X, var=pd.DataFrame(index=GO))
A.obs_names = [str(i) for i in range(X.shape[0])]
base = ANSB.score_all_methods(A, {c: sig_genes[c] for c in cand}, verbose=True)
S_field = {m: base[m][cand].values.astype("float32") for m in FIELD if m in base}

# ---- per-program one-vs-rest AUROC for RECAST + each field method ------------------------
def per_prog_auroc(S):
    out = {}
    for k, c in enumerate(cand):
        yy = (y == c).astype(int)
        out[c] = float(roc_auc_score(yy, S[:, k])) if 0 < yy.sum() < len(yy) else float("nan")
    return out

auroc = {"RECAST": per_prog_auroc(S_recast)}
for m in S_field:
    auroc[m] = per_prog_auroc(S_field[m])
pred_recast = np.array(cand)[S_recast.argmax(1)]
ba_recast = float(balanced_accuracy_score(y, pred_recast))
log(f"[sanity] RECAST argmax BA={ba_recast:.3f}  meanAUROC={np.nanmean(list(auroc['RECAST'].values())):.3f}")
for m in S_field:
    log(f"         {m:8s} meanAUROC={np.nanmean(list(auroc[m].values())):.3f}")

# ---- per-program AUROC long table (method, state, auroc) for panel 6E --------------------
au_rows = [dict(method=m, state=c, auroc=auroc[m][c]) for m in auroc for c in cand]
au_df = pd.DataFrame(au_rows)
au_df.to_csv(os.path.join(OUT, f"program_auroc_{NAME}.csv"), index=False)

# ---- UMAP on the SCimilarity embedding (shared 2-D coords) ------------------------------
log("[umap] on 128-d SCimilarity embedding …")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.30, metric="euclidean", random_state=0)
XY = reducer.fit_transform(Z).astype("float32")

# ---- per-cell dump: GT + RECAST program scores + field program scores (for 6D score maps) ----
df = pd.DataFrame({"umap_x": XY[:, 0], "umap_y": XY[:, 1], "gt_state": y, "recast_pred": pred_recast})
for k, c in enumerate(cand):
    df[f"RECAST::{c}"] = S_recast[:, k]
for m in S_field:
    for k, c in enumerate(cand):
        df[f"{m}::{c}"] = S_field[m][:, k]
csv = os.path.join(OUT, f"program_scores_{NAME}.csv")
df.to_csv(csv, index=False)

meta = dict(lineage=NAME, regime=regime, states=cand, counts=counts, n_cells=int(X.shape[0]),
            field=list(S_field.keys()), recast_argmax_ba=ba_recast,
            mean_auroc={m: float(np.nanmean(list(auroc[m].values()))) for m in auroc},
            per_program_auroc=auroc, sig_genes={c: sig_genes[c] for c in cand})
json.dump(meta, open(os.path.join(OUT, f"program_scores_{NAME}_meta.json"), "w"), indent=2)
log(f"[wrote] {csv}")
log(f"[wrote] {os.path.join(OUT, f'program_auroc_{NAME}.csv')}")
log(f"[done] {time.time()-t0:.0f}s")
