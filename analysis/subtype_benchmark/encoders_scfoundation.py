"""RECAST through scFoundation (a gene-expression transformer) on the 6 cross-encoder lineages (Supplementary Fig. 2e).
Same recipe: scFoundation cell embedding (_get_geneembmerge, 3072-d), u = mean(subtype) - mean(siblings), IG between
mean-lognorm centroids through f(x) = <emb(x), u>, positive channel.  The wrapper truncates each input to its first
512 non-zero genes (gene-index order) before the transformer; for a dense centroid almost every gene is non-zero, so
RECAST can rank only the ~512 genes that remain.  Low recall here reflects that input limit of the scFoundation
interface.
Usage: RECAST_RESULT_TAG=_v3 python encoders_scfoundation.py [--lineage NAME].  Env: SC (GPU)."""
import sys, os, argparse, time, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, torch.nn as nn
sys.path.insert(0, "/path/to/SIGnature/src")
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
from captum.attr import IntegratedGradients
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
from SIGnature.models.scfoundation import SCFoundationWrapper
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
SCF_PATH = "/path/to/research/analysis/models/prepared/scfoundation"
KS = (10, 20)
ap = argparse.ArgumentParser(); ap.add_argument("--lineage", default="all"); A = ap.parse_args()
LINS = ORDER if A.lineage == "all" else [A.lineage]

w = SCFoundationWrapper(model_path=SCF_PATH, use_gpu=True, pool_type="all")
dev = next(w.model.parameters()).device; NG = w.n_genes


def recall(top, mk, gset, K):
    mk = [g for g in mk if g in gset]
    return (len(set(top[:K]) & set(mk)) / len(mk)) if mk else np.nan


class SCFCentroid(nn.Module):
    def __init__(self, wr): super().__init__(); self.w = wr
    def forward(self, x, gene_ids, u):
        emb = self.w._get_geneembmerge(x, gene_ids)
        return torch.stack([(emb * u).sum(1)] * 2, 1)


ig = IntegratedGradients(SCFCentroid(w), multiply_by_inputs=True)


def embed(Xln, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, Xln.shape[0], bs):
            xb = torch.tensor(Xln[i:i + bs], dtype=torch.float32, device=dev)
            gid = torch.arange(NG, device=dev).unsqueeze(0).repeat(xb.shape[0], 1)
            out.append(w._get_geneembmerge(xb, gid).detach().cpu().numpy())
    return np.vstack(out)


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
    cnt_sc, sym, lab = load_lin(c["stem"], c["lab"], states)
    cl = [s for s in states if (lab == s).sum() >= 60]
    if len(cl) < 2: print(f"[skip] {tag}"); continue
    a_ = ad.AnnData(X=cnt_sc.copy()); a_.var_names = sym
    a_.obs_names = [f"c{i}" for i in range(cnt_sc.shape[0])]; a_.layers["counts"] = a_.X.copy()
    adh = w.preprocess_adata(a_)
    genes = np.asarray(adh.var_names).astype(str); gset = set(genes)
    Xln = (adh.X.toarray() if sp.issparse(adh.X) else np.asarray(adh.X)).astype("float32")
    # NOT adh.layers["counts"]: preprocess_adata -> lognorm_counts aliases that layer into X and
    # normalises in place, so post-preprocess it IS the lognorm matrix (see real_counts.align_to).
    C = real_counts.align_to(cnt_sc, sym, genes)
    # was an inline pool-counts-then-log ("pseudobulk") profile; now the paper's one shared definition
    CENT = _cent(C)
    t0 = time.time(); print(f"\n=== {tag}: {Xln.shape[0]} cells, {len(cl)} states, {len(genes)} scF genes — embedding ===", flush=True)
    Z = embed(Xln); print(f"  embedded in {time.time()-t0:.0f}s", flush=True)
    km = np.isin(lab, cl); Zc = normalize(Z[km].astype("float32"))
    yhat = cross_val_predict(KNeighborsClassifier(15, metric="cosine"), Zc, lab[km], cv=5)
    ka, kf = round(float(accuracy_score(lab[km], yhat)), 3), round(float(f1_score(lab[km], yhat, average="macro")), 3)
    knn_rows.append(dict(FM="scFoundation", lineage=tag, dim=Z.shape[1], state_knn_acc=ka, state_macroF1=kf))
    gid1 = torch.arange(NG, device=dev).unsqueeze(0)
    recast, expr = {}, {}
    for cc in cl:
        tgt = lab == cc
        if tgt.sum() < 5 or (~tgt).sum() < 5: recast[cc] = []; expr[cc] = []; continue
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        Cc, Cr = CENT(tgt), CENT(~tgt)
        inp = torch.tensor(Cc[None], device=dev).requires_grad_(True)
        at = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=dev), target=0,
                          additional_forward_args=(gid1, torch.tensor(u[None], device=dev)), n_steps=50).detach().cpu().numpy().ravel()
        at = np.where((Cc - Cr) > 0, np.clip(at, 0, None), -np.inf)
        recast[cc] = [genes[j] for j in np.argsort(-at)[:max(KS)]]
        md = Xln[tgt].mean(0) - Xln[~tgt].mean(0); expr[cc] = [genes[j] for j in np.argsort(-md)[:max(KS)]]
    for K in KS:
        rf = np.nanmean([recall(recast[cc], c["curated"][cc], gset, K) for cc in cl if recast[cc]])
        re = np.nanmean([recall(expr[cc], c["curated"][cc], gset, K) for cc in cl if expr[cc]])
        rows.append(dict(encoder="scFoundation", lineage=tag, k=K, RECAST=round(float(rf), 3), expression=round(float(re), 3), n_states=len(cl)))
        print(f"  recall@{K}: RECAST={rf:.3f} expr={re:.3f}  (kNN F1={kf})  [note: input truncated to 512 tokens]", flush=True)
    del a_, adh, Xln, C, cnt_sc

def merge_csv(path, new_df, keys):
    if os.path.exists(path) and len(new_df):
        old = pd.read_csv(path); old = old[~old.set_index(keys).index.isin(new_df.set_index(keys).index)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


merge_csv(f"{R}/recast_on_scfoundation_multi{VTAG}.csv", pd.DataFrame(rows), ["encoder", "lineage", "k"])
merge_csv(f"{R}/recast_scf_finestate_knn_multi{VTAG}.csv", pd.DataFrame(knn_rows), ["FM", "lineage"])
print("\n[encoders_scfoundation done]", flush=True); print(pd.DataFrame(rows).to_string(index=False))
