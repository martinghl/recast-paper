"""Per-subtype one-vs-rest AUROC of each method's top-20 genes on the original lineages (Fig. 3a, AUROC endpoint;
full_benchmark/scoring/auroc.py does the added lineages).

For subtype c: AUROC of the mean log-normalized expression of c's top-20 genes, c's cells against the lineage's other
cells.  The AUROC is the Mann-Whitney U statistic with average ranks for ties (scipy.stats.rankdata), equal to
sklearn.metrics.roc_auc_score for binary labels.  It is the per-subtype version of the AUROC in
marker_free_scores.py (no mean over subtypes).
Runs on the CPU server (anndata, numpy, pandas, scipy).  Usage: python subtype_auroc.py [lineage ...]
->  results_remote/fs_cm_arm3_perstate_<id><tag>.csv   [lineage, method, k, state, n_pos, n_neg, auroc]"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import rankdata

BASE = "/path/to/cpu_server/analysis/subtype_benchmark"
H5 = "/path/to/cpu_server/analysis/atlases"
LAB = f"{BASE}/bench_labels"
# GENESETS_DIR: comparator gene lists (fs_genesets_v2 for the integer-count runs); RECAST_RESULT_TAG: result tag.
GENESETS = os.environ.get("GENESETS_DIR") or f"{BASE}/fs_genesets"
PANELS = f"{BASE}/panels"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
OUT = f"{BASE}/results_remote"
KS = [20]                                  # the benchmark uses k = 20
sys.path.insert(0, BASE)
import lineage_registry as C

# the 20 original lineages of the benchmark = LINEAGES minus fs_panc_baron (distinct-type control) and fs_b_pbmcbmn
# (no subtype with CellMarker-unique markers); overridable via argv
LIN20 = ['fs_asc_zhao', 'fs_gut_smillie', 'fs_lung_hlca', 'fs_nk_pbmcbmn', 'fs_nk_zhao',
         'tf_b_ren', 'tf_b_steph', 'tf_b_yosh', 'tf_cd4_dom', 'tf_cd4_ren', 'tf_cd4_steph',
         'tf_cd4_yosh', 'tf_cd8_dom', 'tf_cd8_ren', 'tf_cd8_steph', 'tf_cd8_yosh',
         'tf_mono_dom', 'tf_mono_ren', 'tf_mono_steph', 'tf_mono_yosh']


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
    ids = sys.argv[1:] or LIN20
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
