"""Score the comparator gene lists of the marker_methods harness (.rds files) with the marker-free evaluator of Pullin and McCarthy (2024) and against the curated
subtype markers used by run_benchmark.py, so the comparators are scored like the attribution rankings.  Writes
fine_state_baselines_<key>.csv (kNN / SVM / sum_max / ontarget / AUROC x k) and fine_state_baselines_recovery_<key>.csv
(marker recall), with the fine_state_<key>.csv columns.  Also provides geneset_from_rds and _make_names.
Env: SC (numpy, sklearn, anndata, rdata)."""
import os, re, sys, glob, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, rdata
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
RDS_DIR = "/path/to/research/analysis/subtype_benchmark/baseline_rds"   # .rds results copied from the CPU server
KS = [3, 5, 10, 20]
SEED_CV = 20220502

# ---- immune lineages and curated markers (as in run_benchmark.py) ----
IMMUNE = {
    "fs_pbmcbmn_l2": ("pbmcbmn", "bench_labels_pbmcbmn_l2.txt", ["B naive", "B intermediate", "B memory"]),
    "fs_yoshida": ("yoshida", "bench_labels_yoshida.txt", ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells", "Memory B cells"]),
    "fs_helperT": ("dominguez_conde", "bench_labels_dominguez_conde.txt", ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Regulatory T cells"]),
    "fs_cytoT": ("dominguez_conde", "bench_labels_dominguez_conde.txt", ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
}
KEYMAP = {"fs_pbmcbmn_l2": "pbmcbmn_l2", "fs_yoshida": "yoshida", "fs_helperT": "helperT", "fs_cytoT": "cytoT",
          "fs_radc_in": "RADC_IN", "fs_radc_en": "RADC_EN"}
MARKERS = {
    "naive": {"TCL1A", "IGHD", "FCER2", "IL4R", "YBX3", "TSPAN13", "PLPP5", "BACH2", "SELL", "VPREB3", "CD72", "IGHM", "APBB2", "PCDH9"},
    "memory": {"CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH", "CRTAM", "CD80", "ITGB1", "TEX9", "GRIP2", "LINC01781", "BANK1"},
    "switched": {"IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHGP", "CD27", "TNFRSF13B", "AIM2", "S100A10"},
    "nonswitched": {"IGHM", "IGHD", "CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH"},
    "intermediate": {"TNFRSF13B", "COCH", "AIM2", "CRTAM", "MARCKS", "RALGPS2", "SSPN", "SCIMP", "CD72"},
    "naiveT": {"CCR7", "SELL", "TCF7", "LEF1", "IL7R", "TXK", "NOSIP", "MAL", "TRABD2A", "ADTRP", "PLAC8", "LDHB", "CD27", "ACTN1"},
    "eff_helper": {"GZMK", "GZMA", "CCL5", "KLRB1", "ANXA1", "S100A4", "IL32", "AHNAK", "CCR6", "GPR183", "CRTAM", "TNFRSF4", "CCR4"},
    "treg": {"FOXP3", "IL2RA", "IKZF2", "CTLA4", "TNFRSF4", "TNFRSF18", "TIGIT", "RTKN2", "TBC1D4", "LAYN", "IL32", "IL10RA"},
    "eff_cyto": {"GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "FGFBP2", "FCGR3A", "KLRD1", "CX3CR1", "SPON2", "KLRG1", "CMC1", "GZMA", "KLRF1"},
    "trm": {"ITGAE", "ITGA1", "ZNF683", "CXCR6", "XCL1", "XCL2", "CD69", "GZMK", "DUSP2", "CD8A", "CD160", "KLRC1"},
    "gdt": {"TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "KLRC1", "KLRD1", "TRGV9", "TRDV3", "KLRG1", "NKG7"},
}
TMAP = {"Regulatory T cells": "treg", "Tcm/Naive helper T cells": "naiveT", "Tem/Effector helper T cells": "eff_helper",
        "Tcm/Naive cytotoxic T cells": "naiveT", "Tem/Temra cytotoxic T cells": "eff_cyto",
        "Tem/Trm cytotoxic T cells": "trm", "gdT": "gdt"}
# RADC V1 Allen-subclass markers (MARK_BRAIN); the home cell type is the label itself.
MARK_BRAIN = {
    "IN_VIP": {"VIP", "CALB2", "TAC3", "CXCL14", "PENK", "THSD7B", "CHRNA2", "PROX1"},
    "IN_PVALB": {"PVALB", "ERBB4", "KCNS3", "MEF2C", "BTBD11", "WFDC2", "TAC1", "MYBPC1", "LGR5"},
    "IN_PVALB_CHC": {"PVALB", "TRPS1", "PTHLH", "COL15A1", "RBP4", "UNC5B", "NOG"},
    "IN_SST": {"SST", "GRIK1", "CORT", "NPY", "CHODL", "ELFN1", "RELN", "SYNPR", "GRIN3A"},
    "IN_ADARB2": {"ADARB2", "SNCG", "NDNF", "CPLX3", "KIT", "PAX6"},
    "IN_LAMP5_RELN": {"LAMP5", "RELN", "NDNF", "NMBR", "CA13", "NPY1R"},
    "IN_LAMP5_LHX6": {"LAMP5", "LHX6", "NKX2-1", "CHRNA7"},
    "EN_L2_3_IT": {"CUX2", "CARTPT", "LINC00507", "CBLN2", "RASGRF2", "SEMA3C", "PDGFD"},
    "EN_L3_5_IT_1": {"RORB", "PRSS12", "ANXA1", "TRABD2A"},
    "EN_L3_5_IT_2": {"RORB", "RSPO3", "RPRM", "TSHZ2"},
    "EN_L3_5_IT_3": {"RORB", "PCP4", "IZUMO3", "TDO2"},
    "EN_L6_IT_1": {"THEMIS", "TMEM233", "NR4A2", "POU6F2"},
    "EN_L6_IT_2": {"THEMIS", "CDH13", "NTNG2", "PDZRN4"},
    "EN_L6_CT": {"FEZF2", "FOXP2", "TLE4", "SYT6", "GRIK3", "HTR2C", "SEMA3E", "OPRK1"},
    "EN_L6B": {"CTGF", "CCN2", "MOXD1", "CPLX3", "NXPH4", "NR4A2", "TLL1"},
    "EN_L5_6_NP": {"TSHZ2", "NXPH2", "SLC17A8", "HTR2C", "FAM84B"},
}


def immune_state_key(lbl):
    if lbl in TMAP:
        return TMAP[lbl]
    l = lbl.lower()
    if "plasma" in l: return "plasma"
    if "switched" in l and "non" not in l and "unswitched" not in l: return "switched"
    if "non-switched" in l or "unswitched" in l: return "nonswitched"
    if "intermediate" in l: return "intermediate"
    if "naive" in l: return "naive"
    if "memory" in l: return "memory"
    return "memory"


def load_dataset(fsid):
    """Return Xln[cells,genes] (SCimilarity lognorm), labels, genes, gi, clusters, markers dict{cluster:set}."""
    if fsid in IMMUNE:
        stem, lf, states = IMMUNE[fsid]
        att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
        Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
        genes = np.asarray(att.var_names).astype(str)
        fine = np.array([l.strip() for l in open(f"{ATLASES}/{lf}") if l.strip()])
        states = [s for s in states if (fine == s).sum() >= 60]
        keep = np.isin(fine, states); Xln, labels = Xln[keep], fine[keep]
        clusters = sorted(states)
        markers = {c: (MARKERS[immune_state_key(c)] & set(genes)) for c in clusters}
    else:
        parent = {"fs_radc_in": "IN", "fs_radc_en": "EN"}[fsid]
        att = ad.read_h5ad(f"{ATLASES}/RADC_attribution_scimilarity.h5ad")
        Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
        genes = np.asarray(att.var_names).astype(str)
        cls = np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_RADC_class.txt")])
        st = np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_RADC_subclass.txt")])
        inlin = cls == parent
        states = [s for s in pd.unique(st[inlin]) if (st[inlin] == s).sum() >= 60]
        keep = inlin & np.isin(st, states); Xln, labels = Xln[keep], st[keep]
        clusters = sorted(states)
        markers = {c: (MARK_BRAIN.get(c, set()) & set(genes)) for c in clusters}
    gi = {g: i for i, g in enumerate(genes)}
    return Xln, labels, genes, gi, clusters, markers


def _make_names(s):
    """R's make.names(): every character outside [A-Za-z0-9._] becomes a dot."""
    return re.sub(r"[^A-Za-z0-9._]", ".", str(s))


def geneset_from_rds(f, clusters=None):
    """{cluster: [ranked genes]} from one baseline .rds.

    `clusters`, when given, is the caller's own state names. Some R methods (Cepo, on 9 of the 41
    datasets) return their results with cluster names already passed through `make.names`, so
    "Naive B cells" arrives as "Naive.B.cells". Callers look the gene list up by the un-mangled name
    and silently get nothing -- the method then scores 0 on that state rather than being skipped.
    Passing `clusters` renames any such key back, which is safe because the mapping is checked to be
    unambiguous (a mangled key is only renamed when exactly one requested name maps onto it, and the
    requested name is not already a key in its own right).
    """
    d = rdata.read_rds(f); res = d["result"] if isinstance(d, dict) and "result" in d else d
    df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    cols = {c.lower(): c for c in df.columns}
    cc = next((cols[k] for k in ("cluster", "group", "cell_type", "cell", "celltype", "label") if k in cols), None)
    gc = next((cols[k] for k in ("gene", "feature", "names", "symbol", "gene_name") if k in cols), None)
    if cc is None or gc is None: raise ValueError(f"cols {list(df.columns)}")
    out = {str(c): list(grp[gc].astype(str)) for c, grp in df.groupby(cc, sort=False)}
    if clusters:
        for want in clusters:
            want = str(want)
            if want in out:
                continue
            hits = [k for k in out if k == _make_names(want)]
            if len(hits) == 1:
                out[want] = out.pop(hits[0])
    return out


def score_method(name, gsets, Xln, y, gi, clusters, markers):
    """gsets: {cluster:[ranked genes]}. Returns (arm3 rows over k, recovery rows over k)."""
    def union(gs): return [gi[g] for g in sorted({g for c in clusters for g in gs.get(c, []) if g in gi})]
    def med_f1(yt, yp): return float(np.median([f1_score(yt == c, yp == c) for c in clusters]))

    def knn_svm(gs, kind):
        idx = union(gs)
        if not idx: return np.nan
        Xf = Xln[:, idx]; oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xf, y):
            clf = KNeighborsClassifier(3, n_jobs=-1) if kind == "knn" else LinearSVC(dual=False, max_iter=2000)
            clf.fit(Xf[tr], y[tr]); oof[te] = clf.predict(Xf[te])
        return med_f1(y, oof)

    def maxsum(gs):
        cols = {c: [gi[g] for g in gs.get(c, []) if g in gi] for c in clusters}
        allidx = sorted({i for v in cols.values() for i in v})
        if not allidx: return np.nan
        remap = {i: j for j, i in enumerate(allidx)}
        Xsub = Xln[:, allidx]                                  # cells x ~180 only (not 28k) -> ~100x faster
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

    def recovery(gs, K):
        recs = []
        for c in clusters:
            mk = markers.get(c, set())
            if not mk: continue
            top = list(gs.get(c, []))[:K]
            recs.append(len(set(top) & mk) / len(mk))
        return float(np.mean(recs)) if recs else np.nan

    arm3, rec = [], []
    for K in KS:
        gsk = {c: list(gsets.get(c, []))[:K] for c in clusters}
        arm3.append(dict(method=name, k=K, knn=knn_svm(gsk, "knn"), svm=np.nan,   # svm skipped (not shown; speed)
                         sum_max=maxsum(gsk), ontarget=ontarget(gsk), auroc=auroc(gsk)))
    for K in (10, 20, 30):
        rec.append(dict(method=name, k=K, marker_recall=recovery(gsets, K)))
    return arm3, rec


def run(fsid, njobs=8):
    key = KEYMAP[fsid]
    Xln, labels, genes, gi, clusters, markers = load_dataset(fsid)
    y = labels
    files = sorted(glob.glob(f"{RDS_DIR}/{fsid}-*.rds"))
    print(f"[{key}] {len(y)} cells, {len(clusters)} states, {len(files)} baseline methods", flush=True)

    def one(f):
        meth = os.path.basename(f)[len(fsid) + 1:-4]
        try:
            gs = geneset_from_rds(f, clusters)          # `clusters` also un-mangles make.names keys
            gs = {c: gs[c] for c in gs if c in clusters}   # align cluster names
            return score_method(meth, gs, Xln, y, gi, clusters, markers)
        except Exception as e:
            print(f"  [{meth}] ERR {e}", flush=True); return None

    res = [one(f) for f in files]   # sequential methods; KNN uses all cores per call (n_jobs=-1)
    arm3 = [r for x in res if x for r in x[0]]; rec = [r for x in res if x for r in x[1]]
    a3 = pd.DataFrame(arm3); a3.insert(0, "lineage", key)
    rc = pd.DataFrame(rec); rc.insert(0, "lineage", key)
    os.makedirs(OUT, exist_ok=True)
    a3.to_csv(f"{OUT}/fine_state_baselines_{key}.csv", index=False)
    rc.to_csv(f"{OUT}/fine_state_baselines_recovery_{key}.csv", index=False)
    print(f"[{key}] wrote {len(a3)} marker-free rows, {a3.method.nunique()} methods", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ids = list(KEYMAP) if which == "all" else [which]
    for fsid in ids:
        run(fsid)
    print("[scoring done]", flush=True)
