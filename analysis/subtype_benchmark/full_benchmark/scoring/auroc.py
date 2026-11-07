"""Per-subtype one-vs-rest AUROC of each method's top-20 genes on the added lineages (Fig. 3a, AUROC endpoint); the
same computation as subtype_auroc.py, which covers the original lineages.

For subtype c: AUROC of the mean log-normalized expression of c's top-20 genes, c's cells against the lineage's other
cells.  The AUROC is the Mann-Whitney U statistic with average ranks for ties (scipy.stats.rankdata), equal to
sklearn.metrics.roc_auc_score for binary labels.
Runs on the CPU server (anndata, numpy, pandas, scipy).  Usage: python auroc.py [lineage ...]  (default: all added)
->  results/fs_cm_arm3_perstate_<lineage><tag>.csv   [lineage, method, k, state, n_pos, n_neg, auroc]
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import rankdata

ADDED = "/path/to/cpu_server/added_lineages"
BASE = f"{ADDED}/auroc"                    # panels/ and the depth-50 gene lists
H5 = f"{ADDED}/atlases"                    # attribution h5ad files and label files
LAB = H5
GENESETS = os.environ.get("GENESETS_DIR") or f"{BASE}/fs_genesets"
PANELS = f"{BASE}/panels"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
OUT = os.environ.get("AUROC_OUT_DIR", f"{BASE}/results")
KS = [20]                                  # the benchmark uses k = 20
sys.path.insert(0, f"{ADDED}/code")        # a copy of full_benchmark/added_lineages/added_datasets.py
import added_datasets as C


def fast_auroc(y_true, score):
    """Mann-Whitney U AUROC with average ranks for ties == sklearn.metrics.roc_auc_score (binary)."""
    y_true = np.asarray(y_true, dtype=bool)
    n1 = int(y_true.sum()); n0 = int(y_true.size - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)                     # average ranks for ties, 1-based
    return float((r[y_true].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


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


def score_perstate(name, gsets, Xln, y, gi, clusters):
    rows = []
    for K in KS:
        gsk = {c: list(gsets.get(c, []))[:K] for c in clusters}
        for c in clusters:
            idx = [gi[g] for g in gsk.get(c, []) if g in gi]
            if not idx:
                au = np.nan
            else:
                au = fast_auroc(y == c, Xln[:, idx].mean(1))
            npos = int((y == c).sum())
            rows.append(dict(method=name, k=K, state=c, n_pos=npos, n_neg=int(len(y) - npos), auroc=au))
    return rows


def run(fsid, Xfull, genes, gi):
    e = C.LINEAGES[fsid]
    fine = np.array([l.strip() for l in open(f"{LAB}/{e['label_file']}") if l.strip()])
    assert len(fine) == Xfull.shape[0], f"{fsid}: {len(fine)} != {Xfull.shape[0]}"
    clusters = sorted([s for s in e["states"] if (fine == s).sum() >= e["min_cells"]])
    keep = np.isin(fine, clusters); Xln, y = Xfull[keep], fine[keep]
    rows = []
    rows += score_perstate("recast_M0", m0_geneset(fsid), Xln, y, gi, clusters)
    for m, gs in de_genesets(fsid).items():
        rows += score_perstate(m, gs, Xln, y, gi, clusters)
    df = pd.DataFrame(rows); df.insert(0, "lineage", fsid)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/fs_cm_arm3_perstate_{fsid}{VTAG}.csv", index=False)
    chk = df[df.k == 20].groupby("method").auroc.mean()
    fm = chk.get("recast_M0", np.nan)
    print(f"[{fsid}] {len(y)} cells {len(clusters)} states | per-state AUROC written; "
          f"mean-over-states recast_M0={fm:.4f}", flush=True)


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
    print("[per-subtype AUROC done]", flush=True)
