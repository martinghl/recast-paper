"""Per-cell attribution rankings on four immune lineages, scored with the marker-free evaluator of Pullin and McCarthy (2024) (kNN k = 3 / linear SVM / max-sum /
ontarget / AUROC, 5-fold stratified CV seed 20220502, median F1 over subtypes, k in {3, 5, 10, 20}) and against
curated subtype markers:

  contrastive_direct : per-cell contrastive IG against the sibling subtypes (u = mu_subtype - mu_siblings,
                       baseline = sibling cells, positive channel, sum 1000)   [GPU]
  native_direct      : reference-free per-cell IG (attribution_ig layer)
  expression_direct  : log-normalized expression, subtype mean
  wilcoxon / ttest   : scanpy rank_genes_groups on the lineage

Cells are first subset to the lineage, so "rest" within it is the sibling subtypes.  The three per-cell rankings' top-20
lists are written to results/fine_state_direct_genelist_<lineage><tag>.csv (read by the aggregation analyses).
Env: SC (/path/to/envs/SC/bin/python).  `import contrastive_ig` loads SCimilarity on the GPU."""
import argparse, sys, os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
import scanpy as sc
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig  # noqa: E402  (module import loads the SCimilarity encoder on GPU)
from captum.attr import IntegratedGradients                       # noqa: E402
from sklearn.model_selection import StratifiedKFold               # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                # noqa: E402
from sklearn.svm import LinearSVC                                 # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score               # noqa: E402

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
# Internal method name -> the label used in the aggregation tables (mr4_*).
_RUNG = {"native_direct": "native", "contrastive_direct": "contrastive", "expression_direct": "expression"}
KS = [3, 5, 10, 20]
SEED_CV = 20220502

# ---- lineages ----
CFG = {
    "pbmcbmn_l2": ("pbmcbmn", "bench_labels_pbmcbmn_l2.txt",
                   ["B naive", "B intermediate", "B memory"]),
    "yoshida": ("yoshida", "bench_labels_yoshida.txt",
                ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells", "Memory B cells"]),
    "helperT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Regulatory T cells"]),
    "cytoT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
              ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
}

# ---- curated subtype markers, used for marker recall ----
MARKERS = {
    "naive":        {"TCL1A", "IGHD", "FCER2", "IL4R", "YBX3", "TSPAN13", "PLPP5", "BACH2", "SELL", "VPREB3", "CD72", "IGHM", "APBB2", "PCDH9"},
    "memory":       {"CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH", "CRTAM", "CD80", "ITGB1", "TEX9", "GRIP2", "LINC01781", "BANK1"},
    "switched":     {"IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHGP", "CD27", "TNFRSF13B", "AIM2", "S100A10"},
    "nonswitched":  {"IGHM", "IGHD", "CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH"},
    "intermediate": {"TNFRSF13B", "COCH", "AIM2", "CRTAM", "MARCKS", "RALGPS2", "SSPN", "SCIMP", "CD72"},
    "plasma":       {"MZB1", "XBP1", "PRDM1", "DERL3", "JCHAIN", "TNFRSF17", "CD38", "SLAMF7", "ITM2C", "POU2AF1", "SEC11C", "FKBP11"},
    "naiveT":       {"CCR7", "SELL", "TCF7", "LEF1", "IL7R", "TXK", "NOSIP", "MAL", "TRABD2A", "ADTRP", "PLAC8", "LDHB", "CD27", "ACTN1"},
    "eff_helper":   {"GZMK", "GZMA", "CCL5", "KLRB1", "ANXA1", "S100A4", "IL32", "AHNAK", "CCR6", "GPR183", "CRTAM", "TNFRSF4", "CCR4"},
    "treg":         {"FOXP3", "IL2RA", "IKZF2", "CTLA4", "TNFRSF4", "TNFRSF18", "TIGIT", "RTKN2", "TBC1D4", "LAYN", "IL32", "IL10RA"},
    "eff_cyto":     {"GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "FGFBP2", "FCGR3A", "KLRD1", "CX3CR1", "SPON2", "KLRG1", "CMC1", "GZMA", "KLRF1"},
    "trm":          {"ITGAE", "ITGA1", "ZNF683", "CXCR6", "XCL1", "XCL2", "CD69", "GZMK", "DUSP2", "CD8A", "CD160", "KLRC1"},
    "gdt":          {"TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "KLRC1", "KLRD1", "TRGV9", "TRDV3", "KLRG1", "NKG7"},
}
TMAP = {"Regulatory T cells": "treg", "Tcm/Naive helper T cells": "naiveT", "Tem/Effector helper T cells": "eff_helper",
        "Tcm/Naive cytotoxic T cells": "naiveT", "Tem/Temra cytotoxic T cells": "eff_cyto",
        "Tem/Trm cytotoxic T cells": "trm", "gdT": "gdt"}


def state_key(lbl):
    if lbl in TMAP:
        return TMAP[lbl]
    l = lbl.lower()
    if "plasma" in l:
        return "plasma"
    if "switched" in l and "non" not in l and "unswitched" not in l:
        return "switched"
    if "non-switched" in l or "unswitched" in l:
        return "nonswitched"
    if "intermediate" in l:
        return "intermediate"
    if "naive" in l:
        return "naive"
    if "memory" in l:
        return "memory"
    return "memory"


def run_lineage(key):
    stem, finef, STATES = CFG[key]
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    van = att.layers["attribution_ig"]
    van = (van.toarray() if sp.issparse(van) else np.asarray(van)).astype("float32")
    genes = np.asarray(att.var_names).astype(str)
    fine = np.array([l.strip() for l in open(f"{ATLASES}/{finef}") if l.strip()])
    assert len(fine) == att.n_obs, f"[{key}] label/obs mismatch {len(fine)} vs {att.n_obs}"
    del att
    gi = {g: i for i, g in enumerate(genes)}

    STATES = [s for s in STATES if (fine == s).sum() >= 60]
    keep = np.isin(fine, STATES)
    Xln, van, labels = Xln[keep], van[keep], fine[keep]
    clusters = sorted(STATES)
    print(f"[{key}] {len(labels)} cells, states={[(s, int((labels == s).sum())) for s in clusters]}", flush=True)

    # ---------- GPU: per-cell contrastive attribution against the siblings (within the lineage, rest = siblings) ----------
    Z = contrastive_ig.embed(Xln)

    def contrastive_currency():
        cur = np.zeros_like(van)
        mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
        ig = IntegratedGradients(mod, multiply_by_inputs=True)
        for c in clusters:
            m = labels == c
            rows = np.where(m)[0]
            muc = Z[m].mean(0); mur = Z[~m].mean(0)
            u = muc - mur; u /= (np.linalg.norm(u) + 1e-12)
            xbar = Xln[~m].mean(0).astype("float32")
            u_t = torch.tensor(u, dtype=torch.float32, device=contrastive_ig.dev)
            for s in range(0, len(rows), 256):
                b = rows[s:s + 256]; n = len(b)
                inp = torch.tensor(Xln[b], dtype=torch.float32, device=contrastive_ig.dev).requires_grad_(True)
                bt = torch.tensor(np.tile(xbar, (n, 1)), dtype=torch.float32, device=contrastive_ig.dev)
                ut = u_t.repeat(n, 1)
                a = ig.attribute(inp, baselines=bt, target=0, additional_forward_args=(ut,),
                                 n_steps=50, internal_batch_size=64 * 256).detach().cpu().numpy()
                a = np.clip(a, 0, None); ss = a.sum(1, keepdims=True); ss[ss == 0] = 1.0
                cur[b] = (a / ss * 1000.0).astype("float32")
        return cur

    cur = contrastive_currency()
    print(f"[{key}] contrastive currency done", flush=True)

    # ---------- selection ----------
    adata = ad.AnnData(X=Xln.copy()); adata.var_names = genes
    adata.layers["attribution_ig"] = van
    adata.layers["attribution_cig"] = cur
    y = labels

    def clmean(M):
        return {c: M[labels == c].mean(0) for c in clusters}

    _cm = {"contrastive_direct": clmean(cur), "native_direct": clmean(van), "expression_direct": clmean(Xln)}

    def direct_sets(name, K):
        return {c: [genes[j] for j in np.argsort(-_cm[name][c])[:K]] for c in clusters}

    # Write the three per-cell rankings' top-20 lists (read by the aggregation tables, mr4_*); they come from the
    # same code path as the scoring below.
    import os as _os
    _gl = [dict(lineage=key, rung=_RUNG[n], state=c, rank=i + 1, gene=g)
           for n in ("native_direct", "contrastive_direct", "expression_direct")
           for c, gs in direct_sets(n, 20).items() for i, g in enumerate(gs)]
    _glp = f"{OUT}/fine_state_direct_genelist_{key}{_os.environ.get('RECAST_RESULT_TAG', '')}.csv"
    pd.DataFrame(_gl).to_csv(_glp, index=False)
    print(f"  [genelist] wrote {len(_gl)} rows -> {_os.path.basename(_glp)}", flush=True)

    _de_cache = {}

    def de_sets(method, K):
        if method not in _de_cache:
            ada = ad.AnnData(X=Xln.copy()); ada.var_names = genes
            ada.obs["state"] = pd.Categorical(labels, categories=clusters)
            sc.tl.rank_genes_groups(ada, "state", method=("wilcoxon" if method == "wilcoxon" else "t-test"),
                                    n_genes=60)
            dfn = pd.DataFrame(ada.uns["rank_genes_groups"]["names"])
            _de_cache[method] = {c: list(dfn[c]) for c in clusters}
        return {c: _de_cache[method][c][:K] for c in clusters}

    # method registry: name -> (K -> gene-set dict)
    methods = []
    for name in ["contrastive_direct", "native_direct", "expression_direct"]:
        methods.append((name, lambda K, n=name: direct_sets(n, K)))
    for name in ["wilcoxon", "ttest"]:
        methods.append((name, lambda K, n=name: de_sets(n, K)))

    # ---------- marker-free evaluator ----------
    def _union(gs):
        return [gi[g] for g in sorted({g for c in clusters for g in gs.get(c, []) if g in gi})]

    def _median_f1(yt, yp):
        return float(np.median([f1_score(yt == c, yp == c) for c in clusters]))

    def knn_svm(gs, kind):
        idx = _union(gs)
        if not idx:
            return np.nan
        Xf = Xln[:, idx]; oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xf, y):
            clf = KNeighborsClassifier(n_neighbors=3) if kind == "knn" else LinearSVC(max_iter=5000)
            clf.fit(Xf[tr], y[tr]); oof[te] = clf.predict(Xf[te])
        return _median_f1(y, oof)

    def maxsum(gs):
        cols = {c: [gi[g] for g in gs.get(c, []) if g in gi] for c in clusters}
        if not any(cols.values()):
            return np.nan
        oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xln, y):
            mu = Xln[tr].mean(0); sd = Xln[tr].std(0) + 1e-9; Zs = (Xln - mu) / sd
            S = np.column_stack([Zs[:, cols[c]].sum(1) if cols[c] else np.full(len(y), -np.inf) for c in clusters])
            oof[te] = np.asarray(clusters)[S.argmax(1)][te]
        return _median_f1(y, oof)

    def ontarget(gs):
        v = [Xln[y == c, j].mean() / (Xln[y == c, j].mean() + Xln[y != c, j].mean() + 1e-12)
             for c in clusters for j in [gi[g] for g in gs.get(c, []) if g in gi]]
        return float(np.mean(v)) if v else np.nan

    def auroc(gs):
        a = []
        for c in clusters:
            idx = [gi[g] for g in gs.get(c, []) if g in gi]
            if idx:
                a.append(roc_auc_score(y == c, Xln[:, idx].mean(1)))
        return float(np.mean(a)) if a else np.nan

    def recovery(gs, K):
        recs = []
        for c in clusters:
            mk = MARKERS[state_key(c)] & set(genes)
            if not mk:
                continue
            top = list(gs.get(c, []))[:K]
            recs.append(len(set(top) & mk) / len(mk))
        return float(np.mean(recs)) if recs else np.nan

    def safe(f, *a):
        try:
            return f(*a)
        except Exception as e:
            print(f"  [err {getattr(f, '__name__', f)}: {e}]", flush=True)
            return np.nan

    # ---------- run ----------
    rows, recrows = [], []
    for nm, fn in methods:
        for K in KS:
            gs = safe(fn, K)
            if not isinstance(gs, dict):
                gs = {}
            rows.append(dict(lineage=key, method=nm, k=K,
                             knn=safe(knn_svm, gs, "knn"), svm=safe(knn_svm, gs, "svm"),
                             sum_max=safe(maxsum, gs), ontarget=safe(ontarget, gs), auroc=safe(auroc, gs)))
        gs30 = safe(fn, 30)
        if isinstance(gs30, dict):
            for K in (10, 20, 30):
                recrows.append(dict(lineage=key, method=nm, k=K, marker_recall=recovery(gs30, K)))
        print(f"  [{nm}] done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    dfp = pd.DataFrame(rows); dfr = pd.DataFrame(recrows)
    dfp.to_csv(f"{OUT}/fine_state_{key}.csv", index=False)
    dfr.to_csv(f"{OUT}/fine_state_recovery_{key}.csv", index=False)
    print(f"[{key}] wrote results/fine_state_{key}.csv (+recovery)", flush=True)
    return dfp, dfr


def leaderboard(R, REC):
    """Mean over k; predictive=mean(knn,svm,sum_max), specificity=mean(ontarget,auroc)."""
    pm = R.groupby(["lineage", "method"])[["knn", "svm", "sum_max", "ontarget", "auroc"]].mean().reset_index()
    pm["predictive"] = pm[["knn", "svm", "sum_max"]].mean(1)
    pm["specificity"] = pm[["ontarget", "auroc"]].mean(1)
    rec = REC[REC.k == 20].groupby(["lineage", "method"])["marker_recall"].mean().reset_index()
    lb = pm.merge(rec, on=["lineage", "method"], how="left")
    return lb


def aggregate_from_disk():
    """Rebuild the three cross-lineage tables from the per-lineage CSVs on disk.

    Each lineage is its own GPU job, so this script is normally invoked four times with
    `--lineage <key>`.  Concatenating only the lineages of the CURRENT invocation left
    fine_state_ALL.csv holding whichever lineage happened to run last -- so the aggregate
    must be rebuilt from every per-lineage file present, not from what is in memory.
    """
    allr, allrec = [], []
    for k in CFG:
        pr, pc = f"{OUT}/fine_state_{k}.csv", f"{OUT}/fine_state_recovery_{k}.csv"
        if os.path.exists(pr) and os.path.exists(pc):
            allr.append(pd.read_csv(pr)); allrec.append(pd.read_csv(pc))
    if not allr:
        return None, None, None
    R = pd.concat(allr, ignore_index=True); REC = pd.concat(allrec, ignore_index=True)
    R.to_csv(f"{OUT}/fine_state_ALL.csv", index=False)
    REC.to_csv(f"{OUT}/fine_state_recovery_ALL.csv", index=False)
    lb = leaderboard(R, REC)
    lb.to_csv(f"{OUT}/fine_state_leaderboard.csv", index=False)
    print(f"[aggregate] {len(R.lineage.unique())} lineages -> fine_state_ALL.csv "
          f"({len(R)} rows), recovery ({len(REC)} rows), leaderboard ({len(lb)} rows)", flush=True)
    return R, REC, lb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="all", help="one CFG key or 'all'")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="skip the GPU run; just rebuild the cross-lineage tables from disk")
    A = ap.parse_args()
    keys = list(CFG) if A.lineage == "all" else [A.lineage]
    os.makedirs(OUT, exist_ok=True)
    if not A.aggregate_only:
        for k in keys:
            run_lineage(k)
    R, REC, lb = aggregate_from_disk()
    if lb is None:
        sys.exit("no per-lineage results on disk to aggregate")
    print("\n=== leaderboard (mean over k; higher=better) ===", flush=True)
    for key in keys:
        sub = lb[lb.lineage == key].sort_values("predictive", ascending=False)
        print(f"\n--- {key} ---", flush=True)
        print(sub[["method", "predictive", "specificity", "marker_recall", "knn", "svm", "sum_max", "ontarget"]]
              .round(3).to_string(index=False), flush=True)
    print("\n[done]", flush=True)
