"""SIGnature's reference-free attribution as the program weights of the per-cell scoring (Fig. 6).

RECAST's program weights are psi_c = [phi(T_c, R_c)]_+ : the positive channel of contrastive integrated gradients of
the SCimilarity encoder, target centroid against the reference centroid (score_programs.build_candidate_params).
SIGnature's attribution is reference-free: integrated gradients on the same encoder without a reference contrast,
computed per cell by the package's SCimilarity wrapper and stored in every atlas attribution file as
layers['attribution_ig'].

Only the weights change: same 28 tasks, same curated panels, same 8,000-cell cap, same folds (StratifiedKFold 5, seed
20220502), same calibration on the training cells (z_train) and the same evaluation as train_only_calibration.py:

  RECAST     psi_c = [IG(C_T -> C_R) . u_c]_+                (contrastive attribution on the training fold)
  SIGnature  psi_c = [mean_{i in T_c, train} a_i]_+          (training-fold mean of the stored per-cell
                                                              reference-free attribution)

Two scoring rules:
  signature_native_gated   S_ic = mean_{g in G_c} psi_c[g] * [x_ig - C_R,g]_+     (the scoring rule of RECAST, other
                           weights)
  signature_native_plain   S_ic = mean_{g in G_c} psi_c[g] * x_ig                 (no reference; sensitivity row)

The training labels only say which cells belong to candidate c's population, as for RECAST.

The RECAST row is recomputed here from the same loader and folds and checked against results_m7/metrics_all_m7.csv
(calibration z_train).

Writes results_signature/ and *_signature.csv.
  cd /path/to/research/analysis
  NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=1 /path/to/envs/SC/bin/python \
      program_scoring/reference_free_scoring.py --dataset all --dump tf_cd8_dom"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_programs as B                                  # noqa: E402  loads SCimilarity on the GPU
from sklearn.model_selection import StratifiedKFold            # noqa: E402
from sklearn.metrics import roc_auc_score                      # noqa: E402

OUT = os.path.join(B.SB, "program_scoring", "results_signature")
BV = os.path.join(B.SB, "validation", "biology")
M7 = os.path.join(B.SB, "program_scoring", "results_m7", "metrics_all_m7.csv")
PARITY_TOL = 2e-3          # GPU integrated gradients are not bit-identical across runs


def load_dataset_with_attr(name, cap=B.CAP):
    """score_programs.load_dataset, plus the stored reference-free attribution of the same cells.

    Mirrors the benchmark loader line for line (same label map, same >=50-cell candidate rule, same
    RandomState(0) stratified cap) so X, y, cand and sig_idx are those of Fig. 6a; A is the
    layers['attribution_ig'] matrix of the same cells in the same SCimilarity gene order."""
    cfg = B.DATASETS[name]
    adata = ad.read_h5ad(cfg["h5ad"])
    n_obs = adata.n_obs
    y = np.array([l.strip() for l in open(cfg["labels"])])
    assert len(y) == n_obs, f"{name}: labels {len(y)} != cells {n_obs}"
    lmap = cfg.get("label_map", {})
    if lmap:
        y = np.array([lmap.get(v, v) for v in y])
    if cfg["regime"] == "distinct":
        panels = {k: v for k, v in json.load(open(cfg["sig_json"])).items()}
    else:
        panels = {s: cfg["panels"][s] for s in cfg["states"] if s in cfg["panels"]}
    cand = sorted(c for c in panels if int((y == c).sum()) >= B.MIN_CELLS)
    assert len(cand) >= 2, f"{name}: <2 viable candidate states"
    keep = np.isin(y, cand)
    assert "attribution_ig" in adata.layers, f"{name}: no attribution_ig layer (no SIGnature arm possible)"
    X = B._densify_go(adata.X[keep], adata.var_names)
    A = B._densify_go(adata.layers["attribution_ig"][keep], adata.var_names)
    y = y[keep]
    del adata
    if X.shape[0] > cap:
        rng = np.random.RandomState(0)
        per = max(B.MIN_CELLS, cap // len(cand))
        sel = []
        for c in cand:
            ci = np.where(y == c)[0]
            sel.append(ci if len(ci) <= per else rng.choice(ci, per, replace=False))
        sel = np.sort(np.concatenate(sel))
        X, A, y = X[sel], A[sel], y[sel]
    sig_idx, sig_genes = {}, {}
    for c in cand:
        present = [g for g in panels[c] if g in B.GI]
        sig_idx[c] = np.array([B.GI[g] for g in present], dtype=int)
        sig_genes[c] = present
    return X, A, y, cand, sig_idx, sig_genes, cfg["regime"]


def reference_free_params(A_tr, X_tr, y_tr, cand):
    """Training-fold SIGnature program: psi_c = [mean reference-free attribution over c's training cells]_+ ,
    plus the same reference profile C_R the RECAST scoring rule uses (mean of the other train cells)."""
    out = {}
    for c in cand:
        m = y_tr == c
        out[c] = dict(psi=np.clip(A_tr[m].mean(0), 0.0, None).astype("float32"),
                      C_ref=X_tr[~m].mean(0).astype("float32"))
    return out


def reference_free_scores(X, par, cand, sig_idx, gated=True):
    n, K = X.shape[0], len(cand)
    S = np.zeros((n, K), "float32")
    for k, c in enumerate(cand):
        gidx = sig_idx[c]
        if gidx.size:
            h = np.clip(X[:, gidx] - par[c]["C_ref"][gidx], 0.0, None) if gated else X[:, gidx]
            S[:, k] = (h * par[c]["psi"][gidx]).mean(1)
    return S


def _ztrain(S_tr, S_te):
    mu, sd = S_tr.mean(0, keepdims=True), S_tr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (S_te - mu) / sd


def run(name, dump=None, train_only=None):
    t0 = time.time()
    X, A, y, cand, sig_idx, sig_genes, regime = load_dataset_with_attr(name)
    n, K = X.shape[0], len(cand)
    methods = ["signature_native_gated", "signature_native_plain", "recast_parity"]
    oof = {f"{m}__{cal}": np.zeros((n, K), "float32") for m in methods for cal in ("raw", "z_train")}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=B.CV_SEED)
    for tr, te in skf.split(X, y):
        par = reference_free_params(A[tr], X[tr], y[tr], cand)
        for m, gated in (("signature_native_gated", True), ("signature_native_plain", False)):
            S_tr = reference_free_scores(X[tr], par, cand, sig_idx, gated=gated)
            S_te = reference_free_scores(X[te], par, cand, sig_idx, gated=gated)
            oof[f"{m}__raw"][te] = S_te
            oof[f"{m}__z_train"][te] = _ztrain(S_tr, S_te)
        rp = B.build_candidate_params(X[tr], y[tr], cand, sig_idx)      # RECAST, for the parity check
        R_tr, R_te = B.m1_scores(X[tr], rp, cand, sig_idx), B.m1_scores(X[te], rp, cand, sig_idx)
        oof["recast_parity__raw"][te] = R_te
        oof["recast_parity__z_train"][te] = _ztrain(R_tr, R_te)
    rows = []
    for m in methods:
        for cal in ("raw", "z_train"):
            r = B.evaluate(m, oof[f"{m}__{cal}"], y, cand)
            r.update(dataset=name, regime=regime, n_cells=n, n_states=K, calibration=cal,
                     weight_source=("SIGnature reference-free attribution" if m.startswith("signature")
                                    else "RECAST contrastive attribution"))
            rows.append(r)
    res = pd.DataFrame(rows)
    # ---- check against the RECAST row of train_only_calibration.py ------------------------------------
    dev = {}
    if train_only is not None and name in train_only.index:
        p = train_only.loc[name]
        mine = res[(res.method == "recast_parity") & (res.calibration == "z_train")].iloc[0]
        for k in ("balanced_acc", "macro_f1", "mean_auroc"):
            dev[k] = float(abs(mine[k] - p[k]))
        assert max(dev.values()) < PARITY_TOL, f"{name}: RECAST parity broken {dev}"
    res["recast_parity_maxdev"] = max(dev.values()) if dev else np.nan
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, f"metrics_signature_{name}.csv"), index=False)
    print(f"[{name}] {regime} n={n} K={K} ({time.time()-t0:.0f}s) parity_maxdev={res.recast_parity_maxdev.iloc[0]:.2e}\n" +
          res[res.calibration == "z_train"][["method", "balanced_acc", "macro_f1", "mean_auroc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    # ---- Fig. 6c/d per-cell dump (one named task) ---------------------------------------------------
    if dump == name:
        d = pd.DataFrame({"gt_state": y})
        for m in ("signature_native_gated", "signature_native_plain"):
            for k, c in enumerate(cand):
                d[f"{'SIGnature' if m.endswith('gated') else 'SIGnature_plain'}::{c}"] = oof[f"{m}__raw"][:, k]
        ref = os.path.join(BV, f"program_scores_{name}.csv")
        if os.path.exists(ref):
            old = pd.read_csv(ref)
            assert len(old) == len(d) and (old.gt_state.values == y).all(), \
                f"{name}: cell order differs from {ref} -- cannot share its embedding"
            d.insert(0, "umap_y", old.umap_y.values)
            d.insert(0, "umap_x", old.umap_x.values)
        d.to_csv(os.path.join(BV, f"program_scores_{name}_reference_free.csv"), index=False)
        au = []
        for m, tag in (("signature_native_gated", "SIGnature"), ("signature_native_plain", "SIGnature_plain")):
            for k, c in enumerate(cand):
                yy = (y == c).astype(int)
                au.append(dict(method=tag, state=c,
                               auroc=float(roc_auc_score(yy, oof[f"{m}__raw"][:, k])) if 0 < yy.sum() < len(yy) else np.nan))
        pd.DataFrame(au).to_csv(os.path.join(BV, f"program_auroc_{name}_reference_free.csv"), index=False)
        print(f"[{name}] dumped per-cell SIGnature scores + per-program AUROC to {BV}", flush=True)
    del X, A
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--dump", default="tf_cd8_dom", help="task whose per-cell SIGnature scores are dumped for Fig. 6c/d")
    Aa = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pub = pd.read_csv(M7)
    pub = pub[pub.calibration == "z_train"].set_index("dataset")
    names = list(B.DATASETS) if Aa.dataset == "all" else Aa.dataset.split(",")
    allres, failed = [], []
    for nm in names:
        try:
            allres.append(run(nm, dump=Aa.dump, train_only=pub))
        except Exception as e:                                        # noqa: BLE001
            import traceback
            print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
            failed.append(nm)
    comb = pd.concat(allres, ignore_index=True)
    comb.to_csv(os.path.join(OUT, "metrics_all_signature.csv"), index=False)
    print(f"\n[written] {OUT}/metrics_all_signature.csv ({comb.dataset.nunique()} tasks, failed={failed})")
    print(comb[comb.calibration == "z_train"].pivot_table(index="method",
          values=["balanced_acc", "macro_f1", "mean_auroc"], aggfunc="mean").round(3).to_string())
