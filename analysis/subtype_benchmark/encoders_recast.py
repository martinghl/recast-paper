"""RECAST through three encoders on the 6 cross-encoder lineages (Supplementary Fig. 2e): SCimilarity (contrastive),
scTab_SSL and PBMC_SSL (SSL MLPs).  Same recipe for each: embed cells, u = mean(subtype) - mean(siblings), IG between
mean-lognorm centroids (positive channel).  Reports RECAST and expression-ranking recall@10/20 and subtype kNN macro-F1
per (encoder, lineage).  Companions: encoders_scvi.py (VAE), encoders_scfoundation.py (transformer).
Backed read, then subset -> to_memory per lineage.  Env: SC (GPU 0).

All three encoders take a log-normalized profile as input (SCimilarity through contrastive_ig.enc, the SSL backbones
after preprocess_adata), so the centroid is the mean-lognorm one; encoders_scvi.py differs because scVI's encoder takes
raw counts and log-transforms internally."""
import sys, os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, torch.nn as nn
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/SIGnature/src")
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
import contrastive_ig
from captum.attr import IntegratedGradients
from SIGnature.models.ssl import SSLWrapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from encoder_lineages import CFG, ORDER
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
_sys.path.insert(0, "/path/to/research/analysis/validation")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
import recast_centroid as fc     # mean-lognorm centroid
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")
# RECAST_CENTROID=pseudobulk switches the IG endpoints to the pooled-counts centroid (sensitivity analysis).
_REC = _os.environ.get("RECAST_CENTROID", fc.PAPER_DEFAULT)
_cent = lambda C: (fc.Centroider(C) if _REC == "mean_lognorm"
                   else (lambda idx: fc.centroid(C, idx, _REC)))

ATLASES = "/path/to/research/analysis/atlases"
R = "/path/to/research/analysis/subtype_benchmark/results"
KS = (10, 20)
SSL_PATHS = [("scTab_SSL", "/path/to/research/analysis/models/prepared/ssl_sctab"),
             ("PBMC_SSL", "/path/to/research/analysis/models/prepared/ssl_pbmc")]


def recall(top, mk, gset, K):
    mk = [g for g in mk if g in gset]
    return (len(set(top[:K]) & set(mk)) / len(mk)) if mk else np.nan


class SSLCentroid(nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, x, u): s = (self.e(x) * u).sum(1); return torch.stack([s, s], 1)


def knn_macro(Z, lab, cl):
    m = np.isin(lab, cl)
    if m.sum() < 10 or len(cl) < 2: return np.nan, np.nan
    Zc = normalize(Z[m].astype("float32")); y = lab[m]
    yhat = cross_val_predict(KNeighborsClassifier(15, metric="cosine"), Zc, y, cv=5)
    return round(float(accuracy_score(y, yhat)), 3), round(float(f1_score(y, yhat, average="macro")), 3)


def recast_expr(embed_fn, ig, dev, genes, gset, Xln, C, lab, cl, curated):
    # IG endpoints: mean-lognorm centroids, transformed once per gene space so the per-cell normalization is not
    # redone for every subtype.  C is integer counts (real_counts.load), which mean_lognorm requires.
    CENT = _cent(C)
    Z = embed_fn(Xln); rows = {}
    recast, expr = {}, {}
    for c in cl:
        tgt = lab == c
        if tgt.sum() < 5 or (~tgt).sum() < 5: recast[c] = []; expr[c] = []; continue
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        Cc, Cr = CENT(tgt), CENT(~tgt)
        inp = torch.tensor(Cc[None], device=dev).requires_grad_(True)
        a = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=dev), target=0,
                         additional_forward_args=(torch.tensor(u[None], device=dev),), n_steps=50).detach().cpu().numpy().ravel()
        a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf)
        recast[c] = [genes[j] for j in np.argsort(-a)[:max(KS)]]
        m = Xln[tgt].mean(0) - Xln[~tgt].mean(0)
        expr[c] = [genes[j] for j in np.argsort(-m)[:max(KS)]]
    out = []
    for K in KS:
        rf = np.nanmean([recall(recast[c], curated[c], gset, K) for c in cl if recast[c]])
        re = np.nanmean([recall(expr[c], curated[c], gset, K) for c in cl if expr[c]])
        out.append((K, round(float(rf), 3), round(float(re), 3)))
    return out, Z


def load_lin(stem, lab_file, states):
    A = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r"); n = A.n_obs
    if lab_file:
        lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{lab_file}") if l.strip()])
    elif "label" in A.obs.columns:
        lab_all = A.obs["label"].astype(str).to_numpy()
    else:
        lab_all = np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_{stem}.txt") if l.strip()])
    assert len(lab_all) == n, f"{stem}: labels {len(lab_all)} != cells {n}"
    mask = np.isin(lab_all, states)
    sub = A[mask].to_memory()
    Xln = (sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)).astype("float32")
    cnt = real_counts.load(stem, sub.obs_names)   # NOT .layers["counts"] -- that is lognorm
    print(f"  real counts {stem}: {real_counts.summary(cnt)}", flush=True)
    sym = np.asarray(sub.var_names).astype(str); lab = lab_all[mask]
    del A, sub
    return Xln, cnt, sym, lab


def mk_ssl_embed(enc, dev):
    def embed(X, bs=512):
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], bs):
                out.append(enc(torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)).cpu().numpy())
        return np.vstack(out)
    return embed


ig_sc = IntegratedGradients(contrastive_ig.CentroidTarget(contrastive_ig.enc), multiply_by_inputs=True)
ssl = []
for name, path in SSL_PATHS:
    w = SSLWrapper(model_path=path, model_filename="classifier", n_layers=5, use_gpu=True)
    ssl.append((name, w, w.model.eval(), next(w.model.parameters()).device))

_only = [a for a in sys.argv[1:] if not a.startswith("-")]
LINS = [t for t in ORDER if t in _only] if _only else ORDER
print(f"running lineages: {LINS}", flush=True)
rec_rows, knn_rows = [], []
for tag in LINS:
    c = CFG[tag]; states = list(c["curated"])
    Xln0, cnt0, sym, lab = load_lin(c["stem"], c["lab"], states)
    cl = [s for s in states if (lab == s).sum() >= 60]
    print(f"\n### {tag}: {len(lab)} cells, {len(cl)}/{len(states)} states (≥60) ###", flush=True)
    if len(cl) < 2:
        print(f"  [skip] <2 usable states", flush=True); continue
    # SCimilarity
    assert list(sym) == list(contrastive_ig.go), f"{tag} not in SCim gene order"
    gset = set(sym)
    out, Z = recast_expr(contrastive_ig.embed, ig_sc, contrastive_ig.dev, sym, gset, Xln0, cnt0, lab, cl, c["curated"])
    ka, kf = knn_macro(Z, lab, cl)
    knn_rows.append(dict(FM="SCimilarity", lineage=tag, dim=Z.shape[1], state_knn_acc=ka, state_macroF1=kf))
    for K, rf, re in out:
        rec_rows.append(dict(encoder="SCimilarity", lineage=tag, k=K, RECAST=rf, expression=re, n_states=len(cl)))
        print(f"  [SCimilarity] recall@{K}: RECAST={rf:.3f} expr={re:.3f}", flush=True)
    # SSL backbones
    for name, w, enc, dev in ssl:
        a_ = ad.AnnData(X=cnt0.copy()); a_.var_names = sym
        a_.obs_names = [f"c{i}" for i in range(cnt0.shape[0])]; a_.layers["counts"] = a_.X.copy()
        adh = w.preprocess_adata(a_)
        genes = np.asarray(adh.var_names).astype(str); gs = set(genes)
        Xln = (adh.X.toarray() if sp.issparse(adh.X) else np.asarray(adh.X)).astype("float32")
        # NOT adh.layers["counts"]: preprocess_adata -> lognorm_counts aliases that layer into X and
        # normalises in place, so post-preprocess it IS the lognorm matrix (see real_counts.align_to).
        # Rebuild the real counts in this FM's gene space instead.
        C = real_counts.align_to(cnt0, sym, genes)
        ig = IntegratedGradients(SSLCentroid(enc), multiply_by_inputs=True)
        out, Z = recast_expr(mk_ssl_embed(enc, dev), ig, dev, genes, gs, Xln, C, lab, cl, c["curated"])
        ka, kf = knn_macro(Z, lab, cl)
        knn_rows.append(dict(FM=name, lineage=tag, dim=Z.shape[1], state_knn_acc=ka, state_macroF1=kf))
        for K, rf, re in out:
            rec_rows.append(dict(encoder=name, lineage=tag, k=K, RECAST=rf, expression=re, n_states=len(cl)))
            print(f"  [{name}] recall@{K}: RECAST={rf:.3f} expr={re:.3f}", flush=True)
        del a_, adh, Xln, C
    del Xln0, cnt0

def merge_csv(path, new_df, keys):
    if os.path.exists(path) and len(new_df):
        old = pd.read_csv(path)
        old = old[~old.set_index(keys).index.isin(new_df.set_index(keys).index)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


merge_csv(f"{R}/recast_cross_fm_recall_multi{VTAG}.csv", pd.DataFrame(rec_rows), ["encoder", "lineage", "k"])
merge_csv(f"{R}/fm_finestate_knn_multi{VTAG}.csv", pd.DataFrame(knn_rows), ["FM", "lineage"])
print("\n[encoders_recast done]", flush=True)
print(pd.DataFrame(rec_rows).to_string(index=False))
