import os
"""RECAST (population contrastive IG, positive channel) on four immune lineages, scored with the marker-free evaluator of Pullin and McCarthy (2024) and against
curated subtype markers.

Per subtype c (target against the lineage's other subtypes):
  u    = normalize(mu_c - mu_siblings)            in the SCimilarity embedding
  C_c  = centroid of c;  C_o = centroid of the pooled sibling cells (population_ig.cluster_C, mean-lognorm)
  M0_g = positive channel of IG(C_c ; baseline = C_o ; target F = <enc(x), u>), genes with dC > 0
CFG, MARKERS and state_key are read by deletion_insertion.py and embedding_reference.py.
Writes results/fine_state_m0_<lineage><tag>.csv (+ recall and gene lists).  Env SC (sibling_targets loads the model)."""
import argparse, sys, os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")
import population_ig as L                                             # noqa: E402  props_and_C / cluster_countsums / go
from sibling_targets import embed, attr_dir, ig_of                   # noqa: E402  loads SCimilarity encoder on GPU
from sklearn.model_selection import StratifiedKFold              # noqa: E402
from sklearn.neighbors import KNeighborsClassifier               # noqa: E402
from sklearn.svm import LinearSVC                                 # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score              # noqa: E402
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
KS = [3, 5, 10, 20]
SEED_CV = 20220502

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
    genes = np.asarray(att.var_names).astype(str)
    bc = np.asarray(att.obs_names).astype(str)
    fine = np.array([l.strip() for l in open(f"{ATLASES}/{finef}") if l.strip()])
    assert len(fine) == att.n_obs, f"[{key}] label/obs mismatch {len(fine)} vs {att.n_obs}"
    del att
    gi = {g: i for i, g in enumerate(genes)}

    # raw counts (fullgene), aligned to att cell order
    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pmap = {b: i for i, b in enumerate(np.asarray(fg.obs_names).astype(str))}
    Craw_all = fg.X.tocsr()[np.array([pmap[b] for b in bc])]
    rawvar = np.asarray(fg.var_names).astype(str)
    del fg

    STATES = [s for s in STATES if (fine == s).sum() >= 60]
    keep = np.isin(fine, STATES)
    Xln, labels, Craw = Xln[keep], fine[keep], Craw_all[keep]
    del Craw_all
    clusters = sorted(STATES)
    y = labels
    print(f"[{key}] {len(labels)} cells, states={[(s, int((labels == s).sum())) for s in clusters]}", flush=True)

    # ---------- M0 gene sets ----------
    Z = embed(Xln)
    m0_sets = {}
    for c in clusters:
        tgt = labels == c
        zc_t = Z[tgt].mean(0); zc_o = Z[~tgt].mean(0)
        u = (zc_t - zc_o); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        binlab = np.where(tgt, c, "OTHER")
        # mean-lognorm centroid ($RECAST_CENTROID=pseudobulk switches to pooled counts); population_ig.cluster_C.
        C = L.cluster_C(Craw, binlab, [c, "OTHER"], rawvar)               # [2, 28231] in go order
        C_c, C_o = C[0], C[1]
        pos = (C_c - C_o) > 0
        m0 = np.where(pos, attr_dir(C_c, C_o, u), -np.inf)               # [28231] go order
        order = np.argsort(-m0)
        m0_sets[c] = [ig_of[int(j)] for j in order[:30]]
    print(f"[{key}] M0 done", flush=True)

    # ---------- marker-free evaluator (as in run_benchmark.py) ----------
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
            recs.append(len(set(list(gs.get(c, []))[:K]) & mk) / len(mk))
        return float(np.mean(recs)) if recs else np.nan

    def sub(gs, K):
        return {c: gs[c][:K] for c in clusters}

    rows, recrows = [], []
    for K in KS:
        gs = sub(m0_sets, K)
        rows.append(dict(lineage=key, method="M0", k=K,
                         knn=knn_svm(gs, "knn"), svm=knn_svm(gs, "svm"),
                         sum_max=maxsum(gs), ontarget=ontarget(gs), auroc=auroc(gs)))
    for K in (10, 20, 30):
        recrows.append(dict(lineage=key, method="M0", k=K, marker_recall=recovery(m0_sets, K)))

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/fine_state_m0_{key}{VTAG}.csv", index=False)
    pd.DataFrame(recrows).to_csv(f"{OUT}/fine_state_m0_recovery_{key}{VTAG}.csv", index=False)
    # also dump the M0 gene lists for inspection (are they textbook markers?)
    gl = [dict(lineage=key, state=c, rank=r + 1, gene=g, is_marker=int(g in (MARKERS[state_key(c)] & set(genes))))
          for c in clusters for r, g in enumerate(m0_sets[c])]
    pd.DataFrame(gl).to_csv(f"{OUT}/fine_state_m0_genelist_{key}{VTAG}.csv", index=False)
    print(f"[{key}] wrote M0 results", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(recrows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="all")
    A = ap.parse_args()
    keys = list(CFG) if A.lineage == "all" else [A.lineage]
    allr, allrec = [], []
    for k in keys:
        r, rec = run_lineage(k)
        allr.append(r); allrec.append(rec)
    pd.concat(allr, ignore_index=True).to_csv(f"{OUT}/fine_state_m0_ALL{VTAG}.csv", index=False)
    pd.concat(allrec, ignore_index=True).to_csv(f"{OUT}/fine_state_m0_recovery_ALL{VTAG}.csv", index=False)
    print("\n[M0 done]", flush=True)
