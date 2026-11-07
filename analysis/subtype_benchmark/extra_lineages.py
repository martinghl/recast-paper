"""Additional subtype lineages for the marker-free endpoints, with the same methods (RECAST and its composites,
per-cell contrastive, reference-free per-cell IG, expression) and the same marker-free evaluator (knn / sum_max /
ontarget / auroc, k in {3, 5, 10, 20}) as the benchmark.  Reuses the SCimilarity attribution files (counts and
attribution_ig layers).  LINEAGES is read by export_lineage_counts.py.  Env: SC (GPU)."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys, collections
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig  # noqa: E402
from captum.attr import IntegratedGradients  # noqa: E402
from recast.stats import tauE, mw_auc  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")
ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
KS = [3, 5, 10, 20]; SEED = 20220502; CAP = 6000
rng = np.random.default_rng(0)
VAR = {"M0": lambda a, t, d, r: np.clip(a, 0, None), "m0_tau": lambda a, t, d, r: np.clip(a, 0, None) * t,
       "m0_discr": lambda a, t, d, r: np.clip(a, 0, None) * d, "m0_discrRU": lambda a, t, d, r: np.clip(a, 0, None) * r,
       "m0_tau_discr": lambda a, t, d, r: np.clip(a, 0, None) * t * d, "m0_tau_discrRU": lambda a, t, d, r: np.clip(a, 0, None) * t * r}
METHODS = list(VAR) + ["contrastive_direct", "native_direct", "expression_direct"]

# lineage tag -> (dataset, [sibling states])
LINEAGES = {
    "zhao_Mono": ("zhao", ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"]),
    "zhao_CD8":  ("zhao", ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"]),
    "zhao_CD4":  ("zhao", ["CCR7+ CD4", "CXCR3+ CD4", "Treg"]),
    "zhao_B":    ("zhao", ["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"]),
    "ren_Mono":  ("ren", ["Classical monocytes", "Non-classical monocytes"]),
    "stephenson_Mono": ("stephenson", ["Classical monocytes", "Non-classical monocytes"]),
    "dominguez_Mono":  ("dominguez_conde", ["Classical monocytes", "Non-classical monocytes"]),
    # 4/6 extension: helper-T / cyto-T / B-switch replicated in ren + stephenson (immune, Xu-atlas vocab)
    "ren_helperT":   ("ren", ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Tem/Temra helper T cells", "Regulatory T cells"]),
    "ren_cytoT":     ("ren", ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
    "ren_Bswitch":   ("ren", ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells"]),
    "steph_helperT": ("stephenson", ["Tcm/Naive helper T cells", "Tem/Effector helper T cells"]),
    "steph_cytoT":   ("stephenson", ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
    "steph_Bswitch": ("stephenson", ["Naive B cells", "Memory B cells", "Switched memory B cells"]),
}

cig_mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
ig = IntegratedGradients(cig_mod, multiply_by_inputs=True)


def run_lineage(tag, Xln, cnt, van, genes, gi, labels):
    clusters = sorted(np.unique(labels)); y = labels; G = Xln.shape[1]
    print(f"[{tag}] {Xln.shape}, {len(clusters)} states: {clusters}", flush=True)
    Z = contrastive_ig.embed(Xln)
    tot = np.asarray(cnt.sum(0)).ravel().astype("float64")
    EM = np.vstack([Xln[labels == c].mean(0) for c in clusters]); tau_g = tauE(EM).astype("float32")
    cap = {c: (rng.choice(np.where(labels == c)[0], CAP, replace=False) if (labels == c).sum() > CAP else np.where(labels == c)[0]) for c in clusters}
    cur = np.zeros_like(van); fac = {}
    for i, c in enumerate(clusters):
        tgt = labels == c; rows = np.where(tgt)[0]
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        cs = np.asarray(cnt[tgt].sum(0)).ravel().astype("float64")
        Cc = np.log1p(1e4 * cs / max(cs.sum(), 1)).astype("float32")
        Cr = np.log1p(1e4 * (tot - cs) / max((tot - cs).sum(), 1)).astype("float32")
        inp = torch.tensor(Cc[None], dtype=torch.float32, device=contrastive_ig.dev).requires_grad_(True)
        a = ig.attribute(inp, baselines=torch.tensor(Cr[None], dtype=torch.float32, device=contrastive_ig.dev), target=0,
                         additional_forward_args=(torch.tensor(u[None], dtype=torch.float32, device=contrastive_ig.dev),), n_steps=50).detach().cpu().numpy().ravel()
        ci = cap[c]; rest = np.concatenate([cap[cc] for cc in clusters if cc != c])
        auc = mw_auc(Xln[ci], Xln[rest]); discr = np.maximum(0, 2 * np.nan_to_num(auc, nan=.5) - 1).astype("float32")
        EMm = EM.copy(); EMm[i] = -np.inf; ru = EMm.argmax(0); dRU = np.zeros(G, "float32")
        for r in np.unique(ru):
            if r == i: continue
            gs = np.where(ru == r)[0]; au2 = mw_auc(Xln[ci][:, gs], Xln[cap[clusters[r]]][:, gs]); dRU[gs] = np.maximum(0, 2 * np.nan_to_num(au2, nan=.5) - 1)
        fac[c] = dict(a=a, pos=(Cc - Cr) > 0, discr=discr, dRU=dRU)
        xbar = Xln[~tgt].mean(0).astype("float32"); u_t = torch.tensor(u, dtype=torch.float32, device=contrastive_ig.dev)
        for s in range(0, len(rows), 256):
            b = rows[s:s + 256]; n = len(b)
            inp2 = torch.tensor(Xln[b], dtype=torch.float32, device=contrastive_ig.dev).requires_grad_(True)
            bt = torch.tensor(np.tile(xbar, (n, 1)), dtype=torch.float32, device=contrastive_ig.dev)
            aa = ig.attribute(inp2, baselines=bt, target=0, additional_forward_args=(u_t.repeat(n, 1),), n_steps=50, internal_batch_size=64 * 256).detach().cpu().numpy()
            aa = np.clip(aa, 0, None); ssum = aa.sum(1, keepdims=True); ssum[ssum == 0] = 1.0; cur[b] = (aa / ssum * 1000).astype("float32")
    print(f"[{tag}] attributions done", flush=True)

    def gl(name, K):
        if name in VAR:
            return {c: [genes[j] for j in np.argsort(-np.where(fac[c]["pos"], VAR[name](fac[c]["a"], tau_g, fac[c]["discr"], fac[c]["dRU"]), -np.inf))[:K]] for c in clusters}
        M = {"contrastive_direct": cur, "native_direct": van, "expression_direct": Xln}[name]
        return {c: [genes[j] for j in np.argsort(-M[labels == c].mean(0))[:K]] for c in clusters}

    def union(gs): return [gi[g] for g in sorted({g for c in clusters for g in gs.get(c, []) if g in gi})]
    def med(yt, yp): return float(np.median([f1_score(yt == c, yp == c) for c in clusters]))
    def knn(gs):
        idx = union(gs)
        if not idx: return np.nan
        Xf = Xln[:, idx]; oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xf, y):
            cl = KNeighborsClassifier(3, n_jobs=-1); cl.fit(Xf[tr], y[tr]); oof[te] = cl.predict(Xf[te])
        return med(y, oof)
    def maxsum(gs):
        cols = {c: [gi[g] for g in gs.get(c, []) if g in gi] for c in clusters}; allidx = sorted({i for v in cols.values() for i in v})
        if not allidx: return np.nan
        rm = {i: j for j, i in enumerate(allidx)}; Xs = Xln[:, allidx]; oof = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, y):
            mu = Xs[tr].mean(0); sd = Xs[tr].std(0) + 1e-9; Zs = (Xs - mu) / sd
            S = np.column_stack([Zs[:, [rm[i] for i in cols[c]]].sum(1) if cols[c] else np.full(len(y), -np.inf) for c in clusters])
            oof[te] = np.asarray(clusters)[S.argmax(1)][te]
        return med(y, oof)
    def ont(gs):
        v = [Xln[y == c, j].mean() / (Xln[y == c, j].mean() + Xln[y != c, j].mean() + 1e-12) for c in clusters for j in [gi[g] for g in gs.get(c, []) if g in gi]]
        return float(np.mean(v)) if v else np.nan
    def auroc(gs):
        a = [roc_auc_score(y == c, Xln[:, idx].mean(1)) for c in clusters for idx in [[gi[g] for g in gs.get(c, []) if g in gi]] if idx]
        return float(np.mean(a)) if a else np.nan

    rows_out = []
    for nm in METHODS:
        for K in KS:
            gs = gl(nm, K)
            rows_out.append(dict(lineage=tag, method=nm, k=K, knn=knn(gs), sum_max=maxsum(gs), ontarget=ont(gs), auroc=auroc(gs)))
        print(f"  [{tag}/{nm}] done", flush=True)
    df = pd.DataFrame(rows_out); df.to_csv(f"{OUT}/fine_state_extra_{tag}_arm3{VTAG}.csv", index=False)
    print(f"[{tag}] WROTE fine_state_extra_{tag}_arm3.csv", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(LINEAGES)
    by_ds = {}
    for tag in which:
        ds = LINEAGES[tag][0]; by_ds.setdefault(ds, []).append(tag)
    for ds, tags in by_ds.items():
        print(f"=== loading {ds} ===", flush=True)
        att = ad.read_h5ad(f"{ATLASES}/{ds}_attribution_scimilarity.h5ad")
        Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
        cnt = real_counts.load(ds, att.obs_names)   # NOT .layers["counts"] -- that is lognorm
        print(f"  real counts {ds}: {real_counts.summary(cnt)}", flush=True)
        van = att.layers["attribution_ig"]; van = (van.toarray() if sp.issparse(van) else np.asarray(van)).astype("float32")
        genes = np.asarray(att.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
        alllab = att.obs["label"].astype(str).to_numpy()
        del att
        for tag in tags:
            states = LINEAGES[tag][1]; keep = np.isin(alllab, states)
            run_lineage(tag, Xln[keep], cnt[keep], van[keep], genes, gi, alllab[keep])
        del Xln, cnt, van
    print("[extra lineages done]", flush=True)
