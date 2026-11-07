"""Per-cell program scoring with RECAST weights (Fig. 6), against per-cell signature scorers (ANS / UCell /
JASMINE / Scanpy-AddModuleScore / expr-mean) on the same curated gene sets, as a per-cell argmax-over-candidates
classification.  Every method scores the same curated panels; only the score differs.

RECAST scores (candidate c's reference R = its siblings or the rest):
  M1 recast_weighted_expr    S_i = mean_{g in G_c} phi_c[g] * max(0, x_ig - C_R,g)             (gated)
     + _z / _rank calibrations (per candidate, without labels)
  M2 recast_counterfactual   S_i = f_c(x_i) - f_c(x_i^{G->R}),  f_c(x)=<enc(x),u_c>   (revert G_c to C_R)
  M3 recast_dist_calibrated  q_i=M1;  p_i=ECDF_test(q_i);  S~_i = quantile(F_{G,c}, p_i),
                            F_{G,c} = bootstrap-pseudobulk cluster-level M1 score distribution
Comparators: contrastive_pcell (per-cell contrastive integrated gradients) + expr_mean + the 8 ANS scorers.

Folds: 5-fold StratifiedKFold (seed 20220502).  Per fold, u_c / C_R / phi_c / the bootstrap distribution are built
from training cells only; test cells are scored against every candidate independently (a cell's own label never picks
its reference).  The other scorers use no labels and run once on all cells.

phi_c = positive-channel integrated gradients of f_c on the log-space centroid (mean of per-cell log-normalized values
over the training target cells), from the training reference centroid; everything stays in the log-normalized .X of
the attribution file.

Env SC + one GPU (importing contrastive_ig loads SCimilarity):
  cd /path/to/research/analysis
  CUDA_VISIBLE_DEVICES=0 /path/to/envs/SC/bin/python \
      program_scoring/score_programs.py --dataset all"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

SB = "/path/to/research/analysis"
for _p in (os.path.join(SB, "attribution_methods"), SB, os.path.join(SB, "ans_replication"),
           os.path.join(SB, "subtype_benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import contrastive_ig  # noqa: E402  -- module-level side effect: loads SCimilarity encoder on the GPU
from captum.attr import IntegratedGradients  # noqa: E402
import ans_baselines as ANSB  # noqa: E402
import ans_metrics as AM  # noqa: E402,F401  (helpers; the metrics are computed below)
from encoder_lineages import NEW_PANELS  # noqa: E402
from lineage_registry import LINEAGES  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix)

GO = np.asarray(contrastive_ig.go).astype(str)
GI = contrastive_ig.gi
DEV = contrastive_ig.dev
ENC = contrastive_ig.enc.eval()            # eval mode -> BatchNorm uses running stats (deterministic per cell)
ATLASES = os.path.join(SB, "atlases")
OUT = os.environ.get("RECAST_PCELL_OUT", os.path.join(SB, "program_scoring", "results_recast_pcell"))

CV_SEED = 20220502
MIN_CELLS = 50
CAP = 8000
N_BOOT = 200
N_STEPS = 50

RECAST_METHODS = ["recast_weighted_expr", "recast_weighted_expr_z", "recast_weighted_expr_rank",
                 "recast_weighted_expr_platt",
                 "recast_counterfactual", "recast_counterfactual_z", "recast_counterfactual_rank",
                 "recast_counterfactual_platt",
                 "recast_dist_calibrated", "contrastive_pcell"]
CONV_METHODS = ["expr_mean"] + ANSB.ANS_METHOD_NAMES   # expr_mean + the 8 ANS scorers

DATASETS = {
    "baron_pancreas": dict(
        regime="distinct",
        h5ad=os.path.join(SB, "dataset_files/baron_pancreas/baron_pancreas/"
                              "baron_pancreas_attribution_scimilarity.h5ad"),
        labels=os.path.join(SB, "dataset_files/baron_pancreas/baron_pancreas/"
                                "bench_labels_baron_pancreas.txt"),
        sig_json=os.path.join(SB, "new_datasets/markers_score_baron_pancreas.json"),
        label_map={"activated_stellate": "stellate", "quiescent_stellate": "stellate"}),
    "pelka_crc_epi": dict(
        regime="distinct",
        h5ad=os.path.join(SB, "dataset_files/pelka_crc_epi/pelka_crc_epi/"
                              "pelka_crc_epi_attribution_scimilarity.h5ad"),
        labels=os.path.join(SB, "dataset_files/pelka_crc_epi/pelka_crc_epi/"
                                "bench_labels_pelka_crc_epi.txt"),
        sig_json=os.path.join(SB, "new_datasets/markers_score_pelka_crc_epi.json"),
        label_map={  # one representative per distinct epithelial identity
            "cE01 (Stem/TA-like)": "Stem_TA", "cE03 (Stem/TA-like prolif)": "Prolif",
            "cE05 (Enterocyte 2)": "Enterocyte", "cE08 (Goblet)": "Goblet", "cE09 (Best4)": "Best4",
            "cE10 (Tuft)": "Tuft", "cE11 (Enteroendocrine)": "Enteroendocrine"}),
}
# 7 distinct-type datasets — classify among ALL the dataset's annotated cell types (one-vs-rest, cross-lineage,
# mostly well-separated). Labels = h5ad obs (dumped to distinct_labels/), panels = CellMarker expert markers.
# Skipped: citeseq (cluster-number labels, no type names), zhao (no expert panel).
for _dds in ["pbmc3k", "lawlor", "ss3_pbmc", "ren", "stephenson", "dominguez_conde", "yoshida"]:
    DATASETS[_dds] = dict(regime="distinct",
                          h5ad=os.path.join(ATLASES, f"{_dds}_attribution_scimilarity.h5ad"),
                          labels=os.path.join(SB, f"program_scoring/distinct_labels/labels_{_dds}.txt"),
                          sig_json=os.path.join(SB, f"atlases/broad_markers/{_dds}.json"))
# 18 fine sibling-state lineages, auto-built from lineage_registry.LINEAGES + NEW_PANELS curated panels
for _nm, _cfg in LINEAGES.items():
    DATASETS[_nm] = dict(regime="fine",
                         h5ad=os.path.join(ATLASES, f"{_cfg['ds']}_attribution_scimilarity.h5ad"),
                         labels=os.path.join(ATLASES, _cfg["label_file"]),
                         states=_cfg["states"], panels=NEW_PANELS, label_map={})


# ----------------------------------------------------------------------------- loading
def _densify_go(X, var_names):
    """Dense float32 in SCimilarity GO order (missing genes -> zero columns)."""
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype("float32")
    vn = np.asarray(var_names).astype(str)
    if np.array_equal(vn, GO):
        return X
    pos = {g: i for i, g in enumerate(vn)}
    col = np.array([pos.get(g, -1) for g in GO])
    Xg = np.zeros((X.shape[0], len(GO)), dtype="float32")
    ok = col >= 0
    Xg[:, ok] = X[:, col[ok]]
    return Xg


def load_dataset(name, cap=CAP):
    cfg = DATASETS[name]
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
    cand = sorted(c for c in panels if int((y == c).sum()) >= MIN_CELLS)
    assert len(cand) >= 2, f"{name}: <2 viable candidate states"

    keep = np.isin(y, cand)
    X = _densify_go(adata.X[keep], adata.var_names)
    y = y[keep]
    del adata

    if X.shape[0] > cap:                                   # stratified cap (fixed seed)
        rng = np.random.RandomState(0)
        per = max(MIN_CELLS, cap // len(cand))
        sel = []
        for c in cand:
            ci = np.where(y == c)[0]
            sel.append(ci if len(ci) <= per else rng.choice(ci, per, replace=False))
        sel = np.sort(np.concatenate(sel))
        X, y = X[sel], y[sel]

    sig_idx, sig_genes = {}, {}
    for c in cand:
        present = [g for g in panels[c] if g in GI]
        sig_idx[c] = np.array([GI[g] for g in present], dtype=int)
        sig_genes[c] = present
    return X, y, cand, sig_idx, sig_genes, cfg["regime"]


# ----------------------------------------------------------------------------- RECAST scoring
def build_candidate_params(Xtr, ytr, cand, sig_idx):
    """Train-only per-candidate params: u_c, C_ref, C_tgt (logspace centroids), phi_c (pseudobulk +IG)."""
    Ztr = contrastive_ig.embed(Xtr)
    mod = contrastive_ig.CentroidTarget(ENC).to(DEV).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    params = {}
    for c in cand:
        m = ytr == c
        zc, zr = Ztr[m].mean(0), Ztr[~m].mean(0)
        u = zc - zr
        u /= (np.linalg.norm(u) + 1e-12)
        C_ref = Xtr[~m].mean(0).astype("float32")
        C_tgt = Xtr[m].mean(0).astype("float32")
        u_t = torch.tensor(u, dtype=torch.float32, device=DEV).unsqueeze(0)
        inp = torch.tensor(C_tgt[None], dtype=torch.float32, device=DEV).requires_grad_(True)
        base = torch.tensor(C_ref[None], dtype=torch.float32, device=DEV)
        a = ig.attribute(inp, baselines=base, target=0, additional_forward_args=(u_t,),
                         n_steps=N_STEPS).detach().cpu().numpy()[0]
        params[c] = dict(u=u.astype("float32"), C_ref=C_ref, C_tgt=C_tgt,
                         phi=np.clip(a, 0, None).astype("float32"))   # positive channel
    return params


def m1_scores(X, params, cand, sig_idx):
    """M1 raw: gated RECAST-weighted reference-relative expression for any cell matrix. Pure numpy, no embed."""
    n, K = X.shape[0], len(cand)
    S = np.zeros((n, K), "float32")
    for k, c in enumerate(cand):
        gidx = sig_idx[c]
        if gidx.size:
            h = np.clip(X[:, gidx] - params[c]["C_ref"][gidx], 0.0, None)
            S[:, k] = (h * params[c]["phi"][gidx]).mean(1)
    return S


def m2_scores(X, params, cand, sig_idx):
    """M2 counterfactual: f_c(x) - f_c(x^{G->R}) for any cell matrix (base embed + K reverted re-embeds)."""
    n, K = X.shape[0], len(cand)
    Z = contrastive_ig.embed(X)                                       # shared: f_c(x)=Z@u_c
    S = np.zeros((n, K), "float32")
    for k, c in enumerate(cand):
        gidx = sig_idx[c]
        if gidx.size == 0:
            continue
        Xcf = X.copy()
        Xcf[:, gidx] = params[c]["C_ref"][gidx]            # revert G_c toward reference
        S[:, k] = Z @ params[c]["u"] - contrastive_ig.embed(Xcf) @ params[c]["u"]
    return S


def score_fold_recast(Xte, params, cand, sig_idx, with_pcell=True, with_m2=True):
    """M1 (gated weighted expr, == q for M3) + optional M2 (counterfactual) + optional pcell IG, TEST cells."""
    S1 = m1_scores(Xte, params, cand, sig_idx)                     # M1 (no embed needed)
    out = {"recast_weighted_expr": S1}
    if with_m2:
        out["recast_counterfactual"] = m2_scores(Xte, params, cand, sig_idx)
    if with_pcell:
        out["contrastive_pcell"] = score_fold_pcell(Xte, params, cand, sig_idx)
    return out, S1


def score_fold_pcell(Xte, params, cand, sig_idx, batch=256):
    """Leakage-safe per-cell contrastive IG (u_c/xbar from train): positive channel, row-sum-1000."""
    nte, K = Xte.shape[0], len(cand)
    mod = contrastive_ig.CentroidTarget(ENC).to(DEV).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    S = np.zeros((nte, K), "float32")
    for k, c in enumerate(cand):
        gidx = sig_idx[c]
        if gidx.size == 0:
            continue
        u_t = torch.tensor(params[c]["u"], dtype=torch.float32, device=DEV)
        xbar = params[c]["C_ref"]
        for s in range(0, nte, batch):
            e = min(s + batch, nte)
            n = e - s
            inp = torch.tensor(Xte[s:e], dtype=torch.float32, device=DEV).requires_grad_(True)
            bt = torch.tensor(np.tile(xbar, (n, 1)), dtype=torch.float32, device=DEV)
            ut = u_t.repeat(n, 1)
            a = ig.attribute(inp, baselines=bt, target=0, additional_forward_args=(ut,),
                             n_steps=N_STEPS, internal_batch_size=64 * 256).detach().cpu().numpy()
            a = np.clip(a, 0, None)
            ss = a.sum(1, keepdims=True)
            ss[ss == 0] = 1.0
            a = a / ss * 1000.0
            S[s:e, k] = a[:, gidx].mean(1)
    return S


def score_fold_m3(Xtr, ytr, S1_te, params, cand, sig_idx, rng):
    """M3: map each test cell's M1 percentile onto the bootstrap-pseudobulk cluster-level M1 distribution."""
    nte, K = S1_te.shape
    S3 = np.zeros((nte, K), "float32")
    boot = {}
    for k, c in enumerate(cand):
        gidx = sig_idx[c]
        if gidx.size == 0:
            continue
        w = params[c]["phi"][gidx]
        cref = params[c]["C_ref"][gidx]
        ci = np.where(ytr == c)[0]
        F = np.empty(N_BOOT, "float32")
        for b in range(N_BOOT):
            bs = rng.choice(ci, size=len(ci), replace=True)
            pb = Xtr[bs][:, gidx].mean(0)
            F[b] = float((np.clip(pb - cref, 0.0, None) * w).mean())    # cluster-level M1 on bootstrap PB
        q = S1_te[:, k]
        ranks = np.empty(nte)
        ranks[q.argsort()] = np.arange(1, nte + 1)
        p_pct = (ranks - 0.5) / nte                                     # ECDF percentile over test cells
        S3[:, k] = np.quantile(F, np.clip(p_pct, 0.0, 1.0))
        boot[c] = dict(mean=float(F.mean()), sd=float(F.std()),
                       q10=float(np.quantile(F, .10)), q50=float(np.quantile(F, .50)),
                       q90=float(np.quantile(F, .90)))
    return S3, boot


# ----------------------------------------------------------------------------- calibration + baselines
def _zcal(S):
    mu, sd = S.mean(0, keepdims=True), S.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return ((S - mu) / sd).astype("float32")


def _rankcal(S):
    n, K = S.shape
    R = np.empty_like(S)
    for k in range(K):
        r = np.empty(n)
        r[S[:, k].argsort()] = np.arange(1, n + 1)
        R[:, k] = (r - 0.5) / n
    return R.astype("float32")


def platt_calibrate(S_tr, y_tr, S_te, cand):
    """Supervised per-candidate calibration (step A): fit a 1-D logistic  score_c -> P(label==c)  on TRAIN,
    apply to TEST -> probabilities comparable across candidates. Robust to class count (unlike z, which
    distorts binary) and leakage-safe (fit on train only)."""
    K = len(cand)
    out = np.zeros((S_te.shape[0], K), "float32")
    for k, c in enumerate(cand):
        yk = (y_tr == c).astype(int)
        s_tr = S_tr[:, k:k + 1]
        if yk.sum() == 0 or yk.sum() == len(yk) or np.allclose(s_tr, s_tr[0]):
            out[:, k] = S_te[:, k]                              # degenerate candidate -> passthrough
            continue
        # class_weight='balanced' -> prior-neutral intercept, so argmax is not biased toward frequent
        # candidates (a plain OvR logistic bakes the base rate into b_c and predicts the majority state).
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(s_tr, yk)
        out[:, k] = lr.predict_proba(S_te[:, k:k + 1])[:, 1].astype("float32")
    return out


# ----------------------------------------------------------------------------- evaluation
def evaluate(method, S, y, cand):
    df = pd.DataFrame(S, columns=cand)
    y_pred = df.idxmax(axis=1).values                       # NaN-safe: idxmax ignores NaN unless all-NaN
    bal = float(balanced_accuracy_score(y, y_pred))
    mf1 = float(f1_score(y, y_pred, average="macro", labels=cand, zero_division=0))
    wf1 = float(f1_score(y, y_pred, average="weighted", labels=cand, zero_division=0))
    aucs = []
    for k, c in enumerate(cand):
        yc = (y == c).astype(int)
        s = S[:, k]
        if 0 < yc.sum() < len(yc) and np.isfinite(s).all():
            aucs.append(roc_auc_score(yc, s))
    auroc = float(np.mean(aucs)) if aucs else float("nan")
    return dict(method=method, balanced_acc=bal, macro_f1=mf1, weighted_f1=wf1, mean_auroc=auroc)


# ----------------------------------------------------------------------------- driver
def run(name, with_pcell=True, with_m2=True, cap=CAP, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    X, y, cand, sig_idx, sig_genes, regime = load_dataset(name, cap=cap)
    n, K = X.shape[0], len(cand)
    if verbose:
        counts = {c: int((y == c).sum()) for c in cand}
        print(f"[{name}] regime={regime}  X={X.shape}  states={K}  counts={counts}", flush=True)
        print(f"[{name}] sig sizes: " + ", ".join(f"{c}:{sig_idx[c].size}" for c in cand), flush=True)

    oof = {m: np.zeros((n, K), "float32") for m in
           ["recast_weighted_expr", "recast_weighted_expr_platt", "recast_counterfactual",
            "recast_counterfactual_platt", "recast_dist_calibrated", "contrastive_pcell"]}
    boot_all = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        tf = time.time()
        params = build_candidate_params(X[tr], y[tr], cand, sig_idx)
        sc, S1_te = score_fold_recast(X[te], params, cand, sig_idx, with_pcell=with_pcell, with_m2=with_m2)
        S3, bstats = score_fold_m3(X[tr], y[tr], S1_te, params, cand, sig_idx,
                                   np.random.RandomState(1000 + fold))
        for m, arr in sc.items():
            oof[m][te] = arr
        oof["recast_dist_calibrated"][te] = S3
        # step A: supervised per-candidate Platt fit on TRAIN scores only, applied to TEST
        oof["recast_weighted_expr_platt"][te] = platt_calibrate(
            m1_scores(X[tr], params, cand, sig_idx), y[tr], S1_te, cand)
        if with_m2:
            oof["recast_counterfactual_platt"][te] = platt_calibrate(
                m2_scores(X[tr], params, cand, sig_idx), y[tr], sc["recast_counterfactual"], cand)
        boot_all[f"fold{fold}"] = bstats
        if verbose:
            print(f"  fold {fold}: train={len(tr)} test={len(te)} ({time.time()-tf:.1f}s)", flush=True)

    # Label-free calibration ablations (per-candidate monotone rescale of the OOF score matrix).
    oof["recast_weighted_expr_z"] = _zcal(oof["recast_weighted_expr"])
    oof["recast_weighted_expr_rank"] = _rankcal(oof["recast_weighted_expr"])
    if with_m2:
        oof["recast_counterfactual_z"] = _zcal(oof["recast_counterfactual"])
        oof["recast_counterfactual_rank"] = _rankcal(oof["recast_counterfactual"])

    # conventional baselines (label-free -> once on all cells, same curated panels)
    oof["expr_mean"] = np.column_stack(
        [X[:, sig_idx[c]].mean(1) if sig_idx[c].size else np.zeros(n) for c in cand]).astype("float32")
    A = ad.AnnData(X=X, var=pd.DataFrame(index=GO))
    A.obs_names = [str(i) for i in range(n)]
    base = ANSB.score_all_methods(A, {c: sig_genes[c] for c in cand}, verbose=verbose)
    for m, dfm in base.items():
        oof[m] = dfm[cand].values.astype("float32")

    # evaluate every method (drop any all-zero column, e.g. contrastive_pcell when --no_pcell)
    order = [m for m in RECAST_METHODS + CONV_METHODS
             if m in oof and not np.all(oof[m] == 0)]
    rows = [evaluate(m, oof[m], y, cand) for m in order]
    res = pd.DataFrame(rows)
    res.insert(0, "dataset", name)
    res.insert(1, "regime", regime)
    res["family"] = ["RECAST" if m in RECAST_METHODS else "conventional" for m in res["method"]]
    res["n_cells"] = n
    res["n_states"] = K
    res.to_csv(os.path.join(OUT, f"metrics_{name}.csv"), index=False)

    # side artifacts: bootstrap stats (M3 exposure) + a couple of confusion matrices
    json.dump(boot_all, open(os.path.join(OUT, f"boot_stats_{name}.json"), "w"), indent=1)
    for m in ("recast_counterfactual", "recast_dist_calibrated", "ANS", "UCell"):
        if m in oof:
            yp = pd.DataFrame(oof[m], columns=cand).idxmax(axis=1).values
            cm = confusion_matrix(y, yp, labels=cand, normalize="true")
            pd.DataFrame(cm, index=cand, columns=cand).to_csv(
                os.path.join(OUT, f"confusion_{name}_{m}.csv"))

    if verbose:
        show = res[["method", "family", "balanced_acc", "macro_f1", "mean_auroc"]].copy()
        show = show.sort_values("balanced_acc", ascending=False)
        print(f"\n=== [{name}] regime={regime}  ({time.time()-t0:.0f}s) ===", flush=True)
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-cell RECAST gene-set scoring prototype.")
    ap.add_argument("--dataset", required=True,
                    help="a dataset name | all | fine (18 lineages) | distinct (9: 7 atlases + baron + pelka)")
    ap.add_argument("--no_pcell", action="store_true", help="skip the per-cell contrastive IG comparator")
    ap.add_argument("--no_m2", action="store_true", help="skip the M2 counterfactual (fast M1-only pass)")
    ap.add_argument("--cap", type=int, default=CAP)
    A = ap.parse_args()
    groups = {"all": list(DATASETS),
              "fine": [n for n in DATASETS if DATASETS[n]["regime"] == "fine"],
              "distinct": [n for n in DATASETS if DATASETS[n]["regime"] == "distinct"]}
    names = groups.get(A.dataset, [A.dataset])
    allres = []
    for nm in names:
        try:
            allres.append(run(nm, with_pcell=not A.no_pcell, with_m2=not A.no_m2, cap=A.cap))
        except Exception as e:  # noqa: BLE001 -- one bad lineage must not sink the landscape
            import traceback
            print(f"[{nm}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if len(allres) > 1:
        comb = pd.concat(allres, ignore_index=True)
        comb.to_csv(os.path.join(OUT, "metrics_all.csv"), index=False)
        print(f"\n[written] {os.path.join(OUT, 'metrics_all.csv')}  ({len(allres)} datasets)", flush=True)
