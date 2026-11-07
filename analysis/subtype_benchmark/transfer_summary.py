"""Marker transfer of the attribution methods (RECAST, its composites and reference-free per-cell attribution) across
ren / stephenson / dominguez_conde (9 shared immune subtypes).  For each dataset the method's top-K genes per shared
subtype are computed (GPU: RECAST on mean-lognorm centroids, with the tauE / discr / discrRU weights; reference-free =
subtype mean of attribution_ig), then A's genes are evaluated in B (transfer AUROC / on-target / retention) as in
transfer.py.  Env: SC (GPU)."""
import os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig  # noqa: E402  SCimilarity GPU
from captum.attr import IntegratedGradients  # noqa: E402
from recast.stats import tauE, mw_auc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from transfer import DATASETS, SHARED, K, CAP  # reuse config
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
_sys.path.insert(0, "/path/to/research/analysis/validation")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
import recast_centroid as fc     # the paper's ONE centroid definition (mean_lognorm)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")
ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
rng = np.random.default_rng(0)
VARIANTS = {"M0": lambda a, t, d, r: np.clip(a, 0, None),
            "m0_tau": lambda a, t, d, r: np.clip(a, 0, None) * t,
            "m0_discr": lambda a, t, d, r: np.clip(a, 0, None) * d,
            "m0_discrRU": lambda a, t, d, r: np.clip(a, 0, None) * r,
            "m0_tau_discr": lambda a, t, d, r: np.clip(a, 0, None) * t * d,
            "m0_tau_discrRU": lambda a, t, d, r: np.clip(a, 0, None) * t * r}


def genelists(d):
    """Return {method: {type: [topK genes]}} for RECAST / composites / reference-free attribution on dataset d, for the shared types present."""
    a = ad.read_h5ad(f"{ATLASES}/{d}_attribution_scimilarity.h5ad")
    Xln = (a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype("float32")
    cnt = real_counts.load(d, a.obs_names)   # NOT .layers["counts"] -- that is lognorm
    print(f"  real counts {d}: {real_counts.summary(cnt)}", flush=True)
    van = a.layers.get("attribution_ig"); van = (van.toarray() if sp.issparse(van) else np.asarray(van)).astype("float32") if van is not None else None
    lab = a.obs["label"].astype(str).to_numpy(); genes = np.asarray(a.var_names).astype(str)
    Z = contrastive_ig.embed(Xln); G = Xln.shape[1]
    mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
    # IG endpoints: mean-lognorm centroids (recast_centroid.Centroider).
    CENT = fc.Centroider(cnt)
    types = [t for t in SHARED if (lab == t).sum() >= 20]
    allc = [t for t in np.unique(lab) if (lab == t).sum() >= 10]
    EM = np.vstack([Xln[lab == c].mean(0) for c in allc]); tau_g = tauE(EM).astype("float32")
    cap = {c: (rng.choice(np.where(lab == c)[0], CAP, replace=False) if (lab == c).sum() > CAP else np.where(lab == c)[0]) for c in allc}
    out = {m: {} for m in list(VARIANTS) + (["native_direct"] if van is not None else [])}
    for T in types:
        tgt = lab == T
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        Cc, Cr = CENT(tgt), CENT(~tgt)
        inp = torch.tensor(Cc[None], dtype=torch.float32, device=contrastive_ig.dev).requires_grad_(True)
        att = ig.attribute(inp, baselines=torch.tensor(Cr[None], dtype=torch.float32, device=contrastive_ig.dev), target=0,
                           additional_forward_args=(torch.tensor(u[None], dtype=torch.float32, device=contrastive_ig.dev),), n_steps=50).detach().cpu().numpy().ravel()
        pos = (Cc - Cr) > 0; ci = cap[T]; rest = np.concatenate([cap[c] for c in allc if c != T])
        auc = mw_auc(Xln[ci], Xln[rest]); discr = np.maximum(0, 2 * np.nan_to_num(auc, nan=.5) - 1).astype("float32")
        i = allc.index(T); EMm = EM.copy(); EMm[i] = -np.inf; ru = EMm.argmax(0); dRU = np.zeros(G, "float32")
        for r in np.unique(ru):
            if r == i: continue
            gs = np.where(ru == r)[0]; au2 = mw_auc(Xln[ci][:, gs], Xln[cap[allc[r]]][:, gs]); dRU[gs] = np.maximum(0, 2 * np.nan_to_num(au2, nan=.5) - 1)
        for name, fn in VARIANTS.items():
            sc = np.where(pos, fn(att, tau_g, discr, dRU), -np.inf)
            out[name][T] = [genes[j] for j in np.argsort(-sc)[:K]]
        if van is not None:
            nm = van[tgt].mean(0); out["native_direct"][T] = [genes[j] for j in np.argsort(-nm)[:K]]
    return out, Xln, lab, {g: i for i, g in enumerate(genes)}


def eval_in_B(markers, T, XB, labB, giB):
    idx = [giB[g] for g in markers if g in giB]
    if not idx: return np.nan
    y = (labB == T).astype(int)
    if y.sum() < 5 or y.sum() == len(y): return np.nan
    sig = XB[:, idx].mean(1)
    if len(y) > CAP:
        pos = np.where(y == 1)[0]; neg = rng.choice(np.where(y == 0)[0], min((y == 0).sum(), CAP), replace=False)
        sel = np.concatenate([pos, neg]); y, sig = y[sel], sig[sel]
    return float(roc_auc_score(y, sig))


if __name__ == "__main__":
    GL, XX, LL, GI = {}, {}, {}, {}
    for d in DATASETS:
        GL[d], XX[d], LL[d], GI[d] = genelists(d); print(f"[{d}] gene lists done", flush=True)
    methods = sorted(set.union(*[set(GL[d]) for d in DATASETS]))
    rows = []
    for m in methods:
        for T in SHARED:
            self_auc = {B: eval_in_B(GL[B].get(m, {}).get(T, []), T, XX[B], LL[B], GI[B]) for B in DATASETS if T in GL[B].get(m, {})}
            for A in DATASETS:
                mk = GL[A].get(m, {}).get(T)
                if not mk: continue
                for B in DATASETS:
                    if B == A: continue
                    auc = eval_in_B(mk, T, XX[B], LL[B], GI[B])
                    ret = auc / self_auc[B] if self_auc.get(B) and self_auc[B] > 0 else np.nan
                    rows.append(dict(method=m, cell_type=T, src=A, tgt=B, transfer_auc=auc, retention=ret))
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/transferability_currency{VTAG}.csv", index=False)
    print("\n=== FM-method transfer (mean transfer AUROC) ===")
    print(df.groupby("method")[["transfer_auc", "retention"]].mean().round(3).to_string())
    print(f"wrote {OUT}/transferability_currency{VTAG}.csv")
