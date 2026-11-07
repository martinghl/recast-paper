"""Marker-free scores of the subtype gene lists (kNN / sum_max / ontarget / AUROC over k in {3, 5, 10, 20}),
RECAST and the comparator methods, with the marker-free evaluator of Pullin and McCarthy (2024); no marker panel is involved.  The scoring is that of
score_comparators.score_method (SVM skipped) without its rdata import, since the comparator gene lists are read from
the CSV exports fs_genesets/<id>.csv.

Inputs on the CPU server: atlases/<stem>_attribution_scimilarity.h5ad (X = log-normalized expression in the encoder's
gene order), bench_labels/<label_file> (per-cell subtype labels in h5ad order), fs_genesets/<id>.csv (comparators),
panels/topN_<id>_recast_M0.csv (RECAST).  Writes results_remote/fs_cm_arm3_<id>.csv."""
import os, sys, glob
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score
# NB: knn() below constructs KNeighborsClassifier(3) with no n_jobs (default = 1 job) -> no loky pool, so
# no monkeypatch is needed (and patching __init__ with varargs would break sklearn's get_params/clone).

BASE = "/path/to/cpu_server/analysis/subtype_benchmark"
H5 = "/path/to/cpu_server/analysis/atlases"
LAB = f"{BASE}/bench_labels"
GENESETS = f"{BASE}/fs_genesets"
PANELS = f"{BASE}/panels"
OUT = f"{BASE}/results_remote"
KS = [3, 5, 10, 20]
SEED_CV = 20220502
sys.path.insert(0, BASE)
import lineage_registry as C


def score_arm3(name, gsets, Xln, y, gi, clusters):
    """The marker-free scores of score_comparators.score_method (SVM skipped)."""
    def union(gs): return [gi[g] for g in sorted({g for c in clusters for g in gs.get(c, []) if g in gi})]
    def med_f1(yt, yp): return float(np.median([f1_score(yt == c, yp == c) for c in clusters]))

    def knn(gs):
        idx = union(gs)
        if not idx: return np.nan
        Xf = Xln[:, idx]; oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xf, y):
            clf = KNeighborsClassifier(3); clf.fit(Xf[tr], y[tr]); oof[te] = clf.predict(Xf[te])
        return med_f1(y, oof)

    def maxsum(gs):
        cols = {c: [gi[g] for g in gs.get(c, []) if g in gi] for c in clusters}
        allidx = sorted({i for v in cols.values() for i in v})
        if not allidx: return np.nan
        remap = {i: j for j, i in enumerate(allidx)}
        Xsub = Xln[:, allidx]
        colr = {c: [remap[i] for i in cols[c]] for c in clusters}
        oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xsub, y):
            mu = Xsub[tr].mean(0); sd = Xsub[tr].std(0) + 1e-9; Zs = (Xsub - mu) / sd
            S = np.column_stack([Zs[:, colr[c]].sum(1) if colr[c] else np.full(len(y), -np.inf) for c in clusters])
            oof[te] = np.asarray(clusters)[S.argmax(1)][te]
        return med_f1(y, oof)

    def ontarget(gs):
        v = [Xln[y == c, j].mean() / (Xln[y == c, j].mean() + Xln[y != c, j].mean() + 1e-12)
             for c in clusters for j in [gi[g] for g in gs.get(c, []) if g in gi]]
        return float(np.mean(v)) if v else np.nan

    def auroc(gs):
        a = [roc_auc_score(y == c, Xln[:, idx].mean(1)) for c in clusters
             for idx in [[gi[g] for g in gs.get(c, []) if g in gi]] if idx]
        return float(np.mean(a)) if a else np.nan

    out = []
    for K in KS:
        gsk = {c: list(gsets.get(c, []))[:K] for c in clusters}
        out.append(dict(method=name, k=K, knn=knn(gsk), svm=np.nan,
                        sum_max=maxsum(gsk), ontarget=ontarget(gsk), auroc=auroc(gsk)))
    return out


def m0_geneset(fsid):
    df = pd.read_csv(f"{PANELS}/topN_{fsid}_recast_M0.csv").sort_values(["cluster", "rank"])
    return {str(c): list(g["gene"].astype(str)) for c, g in df.groupby("cluster")}


def de_genesets(fsid):
    df = pd.read_csv(f"{GENESETS}/{fsid}.csv").sort_values(["method", "cluster", "rank"])
    return {str(m): {str(c): list(g["gene"].astype(str)) for c, g in gm.groupby("cluster")}
            for m, gm in df.groupby("method")}


def load_parent(stem):
    att = ad.read_h5ad(f"{H5}/{stem}_attribution_scimilarity.h5ad")
    X = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    genes = np.asarray(att.var_names).astype(str)
    return X, genes, {g: i for i, g in enumerate(genes)}


def run(fsid, Xfull, genes, gi):
    e = C.LINEAGES[fsid]
    fine = np.array([l.strip() for l in open(f"{LAB}/{e['label_file']}") if l.strip()])
    assert len(fine) == Xfull.shape[0], f"{fsid}: {len(fine)} != {Xfull.shape[0]}"
    clusters = sorted([s for s in e["states"] if (fine == s).sum() >= e["min_cells"]])
    keep = np.isin(fine, clusters); Xln, y = Xfull[keep], fine[keep]
    rows = []
    rows += score_arm3("recast_M0", m0_geneset(fsid), Xln, y, gi, clusters)
    for m, gs in de_genesets(fsid).items():
        rows += score_arm3(m, gs, Xln, y, gi, clusters)
    df = pd.DataFrame(rows); df.insert(0, "lineage", fsid)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/fs_cm_arm3_{fsid}.csv", index=False)
    a = df[df.k == 20].dropna(subset=["knn"]).sort_values("knn", ascending=False).reset_index(drop=True)
    rk = int((a.method == "recast_M0").values.argmax()) + 1 if "recast_M0" in set(a.method) else -1
    print(f"[{fsid}] {len(y)} cells {len(clusters)} states | knn@20: recast_M0 rank {rk}/{len(a)} "
          f"({float(a.loc[a.method=='recast_M0','knn'].iloc[0]):.3f}) top={a.iloc[0].method}({a.iloc[0].knn:.3f})", flush=True)


if __name__ == "__main__":
    ids = sys.argv[1:] or C.NEW_IDS
    by_parent = {}
    for f in ids:
        by_parent.setdefault(C.LINEAGES[f]["ds"], []).append(f)
    for stem, lids in by_parent.items():
        print(f"=== parent {stem}: {lids} ===", flush=True)
        Xf, g, gi = load_parent(stem)
        for f in lids:
            run(f, Xf, g, gi)
        del Xf
    print("[marker-free scores done]", flush=True)
