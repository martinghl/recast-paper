"""RECAST through scVI (VAE) on the 6 cross-encoder lineages (Supplementary Fig. 2e).  scVI is trained unsupervised per
lineage (HVG union curated markers, raw counts) and embeds to the n_latent = 10 posterior mean; RECAST runs with u in
the latent space and IG between raw-count centroids through f(x) = <z_enc(log1p(x)).loc, u>, positive channel.
Reports RECAST and expression recall@10/20 and subtype kNN.  Env: SC (GPU 1)."""
import sys, os, argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, torch.nn as nn
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
import scvi
from captum.attr import IntegratedGradients
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from encoder_lineages import CFG, ORDER
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")

ATLASES = "/path/to/research/analysis/atlases"
R = "/path/to/research/analysis/subtype_benchmark/results"
KS = (10, 20)
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--lineage", default="all")
A = ap.parse_args(); LINS = ORDER if A.lineage == "all" else [A.lineage]
scvi.settings.seed = 0


def recall(top, mk, gset, K):
    mk = [g for g in mk if g in gset]
    return (len(set(top[:K]) & set(mk)) / len(mk)) if mk else np.nan


class SCVICentroid(nn.Module):
    def __init__(self, zenc, lv=True): super().__init__(); self.z = zenc; self.lv = lv
    def forward(self, x, u):
        x_ = torch.log1p(x) if self.lv else x
        out = self.z(x_); first = out[0] if isinstance(out, (tuple, list)) else out
        loc = first.loc if hasattr(first, "loc") else first
        return torch.stack([(loc * u).sum(1)] * 2, 1)


def hvg_union(C, sym, curated, n_top=4000):
    mean = np.asarray(C.mean(0)).ravel(); sq = np.asarray(C.multiply(C).mean(0)).ravel()
    disp = np.clip(sq - mean ** 2, 0, None) / (mean + 1e-8)
    order = np.argsort(-np.where(mean > 0.01, disp, -np.inf))
    markers = set().union(*[set(v) for v in curated.values()])
    return np.array(sorted(set(order[:n_top].tolist()) | {i for i, g in enumerate(sym) if g in markers}))


def load_lin(stem, lab_file, states):
    Aa = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r"); n = Aa.n_obs
    lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{lab_file}") if l.strip()]) if lab_file else (
        Aa.obs["label"].astype(str).to_numpy() if "label" in Aa.obs.columns else
        np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_{stem}.txt") if l.strip()]))
    assert len(lab_all) == n
    mask = np.isin(lab_all, states); sub = Aa[mask].to_memory()
    cnt = real_counts.load(stem, sub.obs_names)   # NOT .layers["counts"] -- that is lognorm
    print(f"  real counts {stem}: {real_counts.summary(cnt)}", flush=True)
    sym = np.asarray(sub.var_names).astype(str); lab = lab_all[mask]; del Aa, sub
    return cnt, sym, lab


rows, knn_rows = [], []
for tag in LINS:
    c = CFG[tag]; states = list(c["curated"])
    C0, sym, lab = load_lin(c["stem"], c["lab"], states)
    cl = [s for s in states if (lab == s).sum() >= 60]
    if len(cl) < 2: print(f"[skip] {tag}"); continue
    keep = hvg_union(C0, sym, c["curated"]); genes = sym[keep]; gset = set(genes)
    a = ad.AnnData(X=C0[:, keep].astype("float32").copy()); a.var_names = genes
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]; a.layers["counts"] = a.X.copy()
    print(f"\n=== {tag}: {a.n_obs} cells, {len(cl)} states, {len(genes)} genes — scVI {A.epochs}ep ===", flush=True)
    scvi.model.SCVI.setup_anndata(a, layer="counts"); m = scvi.model.SCVI(a, n_latent=10)
    m.train(max_epochs=A.epochs, early_stopping=True, accelerator="gpu", devices=1, enable_progress_bar=False)
    Z = m.get_latent_representation().astype("float32")
    zenc = m.module.z_encoder.eval(); dev = next(zenc.parameters()).device
    ig = IntegratedGradients(SCVICentroid(zenc, bool(getattr(m.module, "log_variational", True))), multiply_by_inputs=True)
    km = np.isin(lab, cl)
    yhat = cross_val_predict(KNeighborsClassifier(15), Z[km], lab[km], cv=5)
    ka, kf = round(float(accuracy_score(lab[km], yhat)), 3), round(float(f1_score(lab[km], yhat, average="macro")), 3)
    knn_rows.append(dict(FM="scVI", lineage=tag, dim=10, state_knn_acc=ka, state_macroF1=kf))
    Cd = np.asarray(C0[:, keep].todense(), dtype="float64"); tot = Cd.sum(0)
    Xln = np.log1p(1e4 * Cd / np.clip(Cd.sum(1, keepdims=True), 1, None))
    recast, expr = {}, {}
    for cc in cl:
        tgt = lab == cc
        if tgt.sum() < 5 or (~tgt).sum() < 5: recast[cc] = []; expr[cc] = []; continue
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        cs = Cd[tgt].sum(0); Cc = (cs / max(tgt.sum(), 1)).astype("float32"); Cr = ((tot - cs) / max((~tgt).sum(), 1)).astype("float32")
        inp = torch.tensor(Cc[None], device=dev).requires_grad_(True)
        at = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=dev), target=0,
                          additional_forward_args=(torch.tensor(u[None], device=dev),), n_steps=50).detach().cpu().numpy().ravel()
        at = np.where((Cc - Cr) > 0, np.clip(at, 0, None), -np.inf)
        recast[cc] = [genes[j] for j in np.argsort(-at)[:max(KS)]]
        md = Xln[tgt].mean(0) - Xln[~tgt].mean(0); expr[cc] = [genes[j] for j in np.argsort(-md)[:max(KS)]]
    for K in KS:
        rf = np.nanmean([recall(recast[cc], c["curated"][cc], gset, K) for cc in cl if recast[cc]])
        re = np.nanmean([recall(expr[cc], c["curated"][cc], gset, K) for cc in cl if expr[cc]])
        rows.append(dict(encoder="scVI", lineage=tag, k=K, RECAST=round(float(rf), 3), expression=round(float(re), 3), n_states=len(cl)))
        print(f"  recall@{K}: RECAST={rf:.3f} expr={re:.3f}  (kNN F1={kf})", flush=True)
    del C0, Cd, a, m

def merge_csv(path, new_df, keys):
    if os.path.exists(path) and len(new_df):
        old = pd.read_csv(path); old = old[~old.set_index(keys).index.isin(new_df.set_index(keys).index)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


merge_csv(f"{R}/recast_on_scvi_multi{VTAG}.csv", pd.DataFrame(rows), ["encoder", "lineage", "k"])
merge_csv(f"{R}/recast_scvi_finestate_knn_multi{VTAG}.csv", pd.DataFrame(knn_rows), ["FM", "lineage"])
print("\n[encoders_scvi done]", flush=True); print(pd.DataFrame(rows).to_string(index=False))
