"""Per-candidate calibration estimated on the training cells only (Fig. 6a).

score_programs.py calibrates the RECAST weighted-expression score with a per-candidate z (or rank) over the pooled
out-of-fold score matrix: no labels, but every test cell is needed.  Here every calibration statistic is estimated on
the training fold and applied to the held-out fold:

  raw          M1 score as is
  z_pooled     z over all out-of-fold cells (mean, standard deviation)        [sensitivity row]
  z_train      mean and standard deviation of the training-fold scores        [reported]
  rank_train   training-fold ECDF of candidate c evaluated at the test scores
  platt_train  per-candidate logistic fit on the training fold

Same tasks, cap, folds (StratifiedKFold 5, seed 20220502), RECAST parameters (build_candidate_params: u_c, C_ref,
phi_c on training cells) and evaluation (argmax balanced accuracy, macro-F1, mean one-vs-rest AUROC).  The other
scorers use no labels and were computed once on all cells, so their rows do not depend on the folds and are taken from
results_recast_pcell/metrics_all.csv.  Folds grouped by donor are in added_genes/export_weights.py.

-> program_scoring/results_m7/metrics_m7_<task>.csv and metrics_all_m7.csv
Env SC (GPU).  Usage: CUDA_VISIBLE_DEVICES=1 python train_only_calibration.py --dataset all"""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_programs as B          # noqa: E402  (loads the encoder; DATASETS; helpers)
from sklearn.model_selection import StratifiedKFold   # noqa: E402

OUT = os.path.join(B.SB, "program_scoring", "results_m7")


def ecdf_apply(S_tr, S_te):
    R = np.empty_like(S_te)
    for k in range(S_te.shape[1]):
        s = np.sort(S_tr[:, k]); R[:, k] = np.searchsorted(s, S_te[:, k], side="right") / len(s)
    return R.astype("float32")


def run(name):
    X, y, cand, sig_idx, sig_genes, regime = B.load_dataset(name)
    n, K = X.shape[0], len(cand)
    oof = {m: np.zeros((n, K), "float32") for m in ("raw", "z_train", "rank_train", "platt_train")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=B.CV_SEED)
    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        params = B.build_candidate_params(X[tr], y[tr], cand, sig_idx)
        S_tr, S_te = B.m1_scores(X[tr], params, cand, sig_idx), B.m1_scores(X[te], params, cand, sig_idx)
        mu, sd = S_tr.mean(0, keepdims=True), S_tr.std(0, keepdims=True); sd[sd == 0] = 1.0
        oof["raw"][te] = S_te
        oof["z_train"][te] = (S_te - mu) / sd
        oof["rank_train"][te] = ecdf_apply(S_tr, S_te)
        oof["platt_train"][te] = B.platt_calibrate(S_tr, y[tr], S_te, cand)
    oof["z_pooled"] = B._zcal(oof["raw"]); oof["rank_pooled"] = B._rankcal(oof["raw"])
    rows = []
    for m in ("raw", "z_pooled", "z_train", "rank_pooled", "rank_train", "platt_train"):
        r = B.evaluate(m, oof[m], y, cand); r.update(dataset=name, regime=regime, n_cells=n, n_states=K, calibration=m,
                                                    inductive=m in ("raw", "z_train", "rank_train", "platt_train")); rows.append(r)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, f"metrics_m7_{name}.csv"), index=False)
    print(f"[{name}] {regime} n={n} K={K} ({time.time()-t0:.0f}s)\n" +
          res[["calibration", "balanced_acc", "macro_f1", "mean_auroc"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="all"); A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    names = list(B.DATASETS) if A.dataset == "all" else [A.dataset]
    allres = []
    for nm in names:
        try:
            allres.append(run(nm))
        except Exception as e:  # noqa: BLE001
            import traceback; print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    comb = pd.concat(allres, ignore_index=True); comb.to_csv(os.path.join(OUT, "metrics_all_m7.csv"), index=False)
    print(f"\n[written] {OUT}/metrics_all_m7.csv ({len(allres)} tasks)")
    piv = comb.pivot_table(index="calibration", values=["balanced_acc", "macro_f1", "mean_auroc"], aggfunc="mean")
    print(piv.round(3).to_string())
