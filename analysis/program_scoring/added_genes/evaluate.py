"""Evaluate every scorer x panel variant x fold scheme x calibration for one task (CPU server).

  python evaluate.py <task> <exp>          exp in B | A1 | A2   (score files under
                                           /path/to/cpu_server/program_scoring/added_genes/scores_<exp>/)
Calibration
  matched   every scorer: per fold of the scheme, mean and standard deviation of the fold's training-cell scores per
            candidate, applied to its test cells (z_train).  RECAST and SIGnature come from score_variants_recast.py
            already as z_train.
  original  RECAST and SIGnature z_train, every other scorer raw (as in Fig. 6a).
Empty panels: a candidate whose variant panel is empty is NaN for every scorer; argmax over the rest; AUROC averaged
over the candidates with a non-empty panel (the same set for every scorer).  A scorer that returns no score for a
non-empty candidate contributes AUROC 0.5 for it and cannot predict it (counted in n_failed_cols); a scored but
constant column is kept and counted in n_const_cols.  Metrics otherwise as score_programs.evaluate.
Writes /path/to/cpu_server/program_scoring/added_genes/eval_<exp>/<task>.csv"""
import json, os, sys
import numpy as np, pandas as pd, anndata as ad
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
ROOT = "/path/to/cpu_server/program_scoring"; R = "/path/to/cpu_server/program_scoring/added_genes"
task, exp = sys.argv[1:3]
SC, VF = f"{R}/scores_{exp}", f"{R}/variants_{exp}/{task}.json"
a = ad.read_h5ad(f"{ROOT}/inputs/{task}.h5ad", backed="r")
y = a.obs["label"].astype(str).to_numpy(); cand = list(a.uns["cand"]); n, K = len(y), len(cand)
gset = set(a.var_names)
V = json.load(open(VF)); vids = list(V)
PY = ["ANS", "Seurat", "Seurat_AG", "Seurat_LVG", "Scanpy", "Jasmine_LH", "Jasmine_OR", "UCell", "expr_mean", "SCSE",
      "decoupleR_ULM"]
RM = ["AUCell", "ssGSEA", "GSVA", "zscore", "VAM", "singscore"]


def load_free(m):
    if m in PY:
        z = np.load(f"{SC}/{task}__{m}.npz", allow_pickle=True)
        assert list(z["variants"]) == vids and list(z["cand"]) == cand, m
        return z["S"].astype("float64")
    D = pd.read_csv(f"{SC}/{task}__{m}.csv.gz")
    S = np.full((len(vids), n, K), np.nan)
    for vi, v in enumerate(vids):
        for k, c in enumerate(cand):
            col = f"{v}::{c}"
            if col in D.columns:
                S[vi, :, k] = D[col].to_numpy(dtype="float64")
    return S


def ztrain(S, fold):
    Z = np.full_like(S, np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        mu, sd = np.nanmean(S[tr], 0), np.nanstd(S[tr], 0)
        sd = np.where((sd == 0) | ~np.isfinite(sd), 1.0, sd)
        Z[te] = (S[te] - mu) / sd
    return Z


def evaluate(S, nonempty):
    S = S.copy(); S[:, ~nonempty] = np.nan
    ok = np.isfinite(S).all(0)                              # usable columns (non-empty and scored)
    pred = pd.DataFrame(S, columns=cand).idxmax(axis=1)
    pred = pred.fillna("__none__").to_numpy()
    ba = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=cand, zero_division=0)
    aucs = []
    for k, c in enumerate(cand):
        if not nonempty[k]:
            continue
        yc = (y == c).astype(int)
        aucs.append(roc_auc_score(yc, S[:, k]) if ok[k] else 0.5)
    const = ok & nonempty & (np.nanstd(S, 0) == 0)          # e.g. VAM's failed gamma fit -> cdf 0 for every cell
    return ba, f1, float(np.mean(aucs)), int((nonempty & ~ok).sum()), int(const.sum())


rc = np.load(f"{SC}/{task}__RECAST_SIGnature.npz", allow_pickle=True)
assert list(rc["variants"]) == vids and list(rc["cand"]) == cand
free = {m: load_free(m) for m in PY + RM}
rows = []
for scheme in ("donor", "cell"):
    fold = rc[f"fold__{scheme}"]
    freez = {m: np.stack([ztrain(free[m][vi], fold) for vi in range(len(vids))]) for m in free}
    for vi, v in enumerate(vids):
        nonempty = np.array([any(g in gset for g in V[v][c]) for c in cand])
        for cal in ("matched", "original"):
            for m in ["RECAST", "SIGnature"] + PY + RM:
                if m in ("RECAST", "SIGnature"):
                    S = rc[f"{m}__{scheme}__ztrain"][vi].astype("float64")
                else:
                    S = freez[m][vi] if cal == "matched" else free[m][vi]
                ba, f1, auc, nf, nc = evaluate(S, nonempty)
                rows.append(dict(task=task, scheme=scheme, calibration=cal, variant=v, method=m, balanced_acc=ba,
                                 macro_f1=f1, mean_auroc=auc, n_failed_cols=nf, n_const_cols=nc,
                                 n_nonempty=int(nonempty.sum())))
os.makedirs(f"{R}/eval_{exp}", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{R}/eval_{exp}/{task}.csv", index=False)
print("done", task, exp, len(rows), flush=True)
