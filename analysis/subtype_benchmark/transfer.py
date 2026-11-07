"""Marker transfer across datasets: ren / stephenson / dominguez_conde share 9 immune subtypes (exact labels).  For each
method and shared subtype T, the method's top-K genes selected in dataset A are evaluated in dataset B: one-vs-rest
AUROC and on-target ratio of A's genes for T in B, and retention = metric(A's genes in B) / metric(B's own genes in B),
aggregated per method over all (T, A -> B).  Comparator lists come from the harness .rds files (no GPU); the attribution
methods are added by transfer_summary.py (GPU).  Env: SC."""
import os, glob, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from score_comparators import geneset_from_rds
ATLASES = "/path/to/research/analysis/atlases"
RDS = "/path/to/research/analysis/marker_methods_whole_atlas/results"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
DATASETS = ["ren", "stephenson", "dominguez_conde"]
SHARED = ["CD16+ NK cells", "Classical monocytes", "Non-classical monocytes", "Tcm/Naive cytotoxic T cells",
          "Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Tem/Temra cytotoxic T cells",
          "Tem/Trm cytotoxic T cells", "gdT"]
K = 20
CAP = 8000
rng = np.random.default_rng(0)


def load(d):
    a = ad.read_h5ad(f"{ATLASES}/{d}_attribution_scimilarity.h5ad")
    X = (a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype("float32")
    lab = a.obs["label"].astype(str).to_numpy(); genes = np.asarray(a.var_names).astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    # cap cells per dataset for AUROC speed (stratified-ish: keep all shared-type cells + a background sample)
    return X, lab, gi


def eval_markers_in_B(markers, T, XB, labB, giB):
    """AUROC + on-target of `markers` for type T evaluated in dataset B."""
    idx = [giB[g] for g in markers if g in giB]
    if not idx:
        return np.nan, np.nan
    y = (labB == T).astype(int)
    if y.sum() < 5 or y.sum() == len(y):
        return np.nan, np.nan
    sig = XB[:, idx].mean(1)
    # subsample for AUROC speed
    if len(y) > CAP:
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        neg = rng.choice(neg, min(len(neg), CAP), replace=False)
        sel = np.concatenate([pos, neg]); y, sig = y[sel], sig[sel]
    auc = roc_auc_score(y, sig)
    inm = XB[labB == T][:, idx].mean(0); outm = XB[labB != T][:, idx].mean(0)
    ont = float(np.mean(inm / (inm + outm + 1e-12)))
    return float(auc), ont


if __name__ == "__main__":
    D = {d: load(d) for d in DATASETS}
    # methods = intersection of .rds tags present for all 3 datasets
    def tags(d): return {os.path.basename(f).split(f"{d}_allgenes-")[1][:-4] for f in glob.glob(f"{RDS}/{d}_allgenes-*.rds")}
    methods = sorted(set.intersection(*[tags(d) for d in DATASETS]))
    print(f"{len(methods)} methods shared across the 3 datasets", flush=True)
    gsets = {d: {} for d in DATASETS}
    for d in DATASETS:
        for m in methods:
            try:
                gsets[d][m] = geneset_from_rds(f"{RDS}/{d}_allgenes-{m}.rds")
            except Exception:
                gsets[d][m] = {}
    rows = []
    for m in methods:
        for T in SHARED:
            # self metric in each B (ceiling)
            self_auc = {}
            for B in DATASETS:
                XB, labB, giB = D[B]
                gsB = gsets[B].get(m, {})
                if T in gsB:
                    a, o = eval_markers_in_B(list(gsB[T])[:K], T, XB, labB, giB); self_auc[B] = a
            for A in DATASETS:
                gsA = gsets[A].get(m, {})
                if T not in gsA:
                    continue
                mk = list(gsA[T])[:K]
                for B in DATASETS:
                    if B == A:
                        continue
                    XB, labB, giB = D[B]
                    auc, ont = eval_markers_in_B(mk, T, XB, labB, giB)
                    ret = auc / self_auc[B] if self_auc.get(B) and self_auc[B] == self_auc[B] and self_auc[B] > 0 else np.nan
                    rows.append(dict(method=m, cell_type=T, src=A, tgt=B, transfer_auc=auc, transfer_ontarget=ont, retention=ret))
        print(f"  [{m}] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/transferability_baselines.csv", index=False)
    agg = df.groupby("method")[["transfer_auc", "transfer_ontarget", "retention"]].mean().sort_values("transfer_auc", ascending=False)
    print("\n=== transfer (mean over shared types x dataset pairs), top 15 ===")
    print(agg.head(15).round(3).to_string())
    print(f"\nwrote {OUT}/transferability_baselines.csv")
