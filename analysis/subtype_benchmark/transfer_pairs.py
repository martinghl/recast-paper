"""Transfer of subtype panels between the four blood datasets that share subtype names (Fig. 6e).  Runs on the CPU
server (marker_methods).  A panel selected on dataset A is evaluated on dataset B's expression with
marker_free_scores.score_arm3 (the marker-free evaluator of Pullin and McCarthy (2024): kNN / sum_max / ontarget / AUROC, k in {3, 5, 10, 20}, 5-fold CV seed
20220502).

Pairwise: for each ordered pair (A, B), A != B, restricted to S = subtypes(A) ∩ subtypes(B) with >= min_cells in B; both
panel(M, A) on B|S (transfer) and panel(M, B) on B|S (same-dataset ceiling for this pair) -> retention = transfer /
same-dataset.
Leave one dataset out: for held-out H, consensus panel per subtype = mean-rank aggregation of M's panels over the other
datasets (absent gene = worst rank + 1, re-sorted, top 50), evaluated on H.

Cells are capped per target (CAP), the same cells for every method and source.  Methods: RECAST, the comparators and a
random floor.  -> transfer/arm3_transfer_pairwise<tag>.csv, transfer/arm3_transfer_lodo<tag>.csv."""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
BASE = "/path/to/cpu_server/analysis/subtype_benchmark"
H5 = "/path/to/cpu_server/analysis/atlases"
sys.path.insert(0, BASE)
import marker_free_scores as A3          # score_arm3 and KS
OUT = f"{BASE}/transfer"; os.makedirs(OUT, exist_ok=True)
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
# The result tag also fixes the comparator gene lists (the _v2 / _v3 runs use the integer-count lists);
# GENESETS_DIR overrides.
GEN_BY_TAG = {"": "fs_genesets", "_v2": "fs_genesets_v2", "_v3": "fs_genesets_v2"}
GEN = os.environ.get("GENESETS_DIR") or f"{BASE}/{GEN_BY_TAG[VTAG]}"
PAN, LAB = f"{BASE}/panels", f"{BASE}/bench_labels"
MIN_CELLS, CAP, TOPK = 50, 8000, 50
# "pairwise" | "lodo" | "both". The two protocols are independent, so either half can be re-run alone.
ONLY = os.environ.get("TRANSFER_ONLY", "both")
assert ONLY in ("both", "pairwise", "lodo"), ONLY

COMP = {
    "cd8": dict(ds={"dom": "dominguez_conde", "ren": "ren", "steph": "stephenson", "yosh": "yoshida"}, id="tf_cd8_{k}"),
    "cd4": dict(ds={"dom": "dominguez_conde", "ren": "ren", "steph": "stephenson", "yosh": "yoshida"}, id="tf_cd4_{k}"),
    "b":   dict(ds={"ren": "ren", "steph": "stephenson", "yosh": "yoshida"}, id="tf_b_{k}"),
    "mono": dict(ds={"dom": "dominguez_conde", "ren": "ren", "steph": "stephenson", "yosh": "yoshida"}, id="tf_mono_{k}"),
}
LABF = {"dominguez_conde": "bench_labels_dominguez_conde.txt", "ren": "bench_labels_ren.txt",
        "stephenson": "bench_labels_stephenson.txt", "yoshida": "bench_labels_yoshida.txt"}


def panels_of(fsid):
    """{method: {state:[ranked genes<=TOPK]}} for RECAST_M0 + every DE method + random."""
    out = {}
    f = pd.read_csv(f"{PAN}/topN_{fsid}_recast_M0.csv").sort_values(["cluster", "rank"])
    out["RECAST_M0"] = {str(c): list(g["gene"].astype(str))[:TOPK] for c, g in f.groupby("cluster")}
    de = pd.read_csv(f"{GEN}/{fsid}.csv").sort_values(["method", "cluster", "rank"])
    for m, gm in de.groupby("method"):
        out[str(m)] = {str(c): list(g["gene"].astype(str))[:TOPK] for c, g in gm.groupby("cluster")}
    return out


def load_target(stem):
    att = ad.read_h5ad(f"{H5}/{stem}_attribution_scimilarity.h5ad")
    genes = np.asarray(att.var_names).astype(str)
    X = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    del att
    gi = {g: i for i, g in enumerate(genes)}
    return X, genes, gi


def subset(y_full, X, S, rng):
    """cells of B in shared states S, capped at CAP total (stratified), same for all methods."""
    mask = np.isin(y_full, list(S))
    idx = np.where(mask)[0]
    if len(idx) > CAP:
        # stratified cap
        keep = []
        per = max(MIN_CELLS, CAP // len(S))
        for s in S:
            si = idx[y_full[idx] == s]
            keep.append(si if len(si) <= per else rng.choice(si, per, replace=False))
        idx = np.sort(np.concatenate(keep))
    return X[idx], y_full[idx]


def eval_panel(panel, Xs, ys, gi, S):
    """mean over k of each metric for panel restricted to S -> {metric: value}."""
    gs = {s: panel.get(s, []) for s in S}
    rows = A3.score_arm3("m", gs, Xs, ys, gi, sorted(S))
    df = pd.DataFrame(rows)
    return {m: float(df[m].mean()) for m in ("knn", "sum_max", "ontarget", "auroc")}


def rand_panel(genes, states, rng):
    return {s: list(rng.choice(genes, TOPK, replace=False)) for s in states}


def states_of(panel_dict):
    return set(panel_dict["RECAST_M0"].keys())


def consensus(method_panels_bysrc, state, worst=TOPK + 1):
    """mean-rank aggregation of one method's ranked lists for `state` across sources that have it -> top-TOPK."""
    lists = [p[state] for p in method_panels_bysrc if state in p]
    if not lists:
        return []
    genes = set(g for L in lists for g in L)
    mr = {}
    for g in genes:
        ranks = [(L.index(g) + 1) if g in L else worst for L in lists]
        mr[g] = np.mean(ranks)
    # Tie-break on the gene symbol, not on set-iteration order: many genes share a mean rank (e.g. any
    # gene present in exactly one source list at the same position), and resolving those by the order a
    # Python set happens to yield made the consensus panel differ between processes -- the same run
    # re-executed elsewhere moved RECAST's LODO sum-max by up to 0.018 with the inputs held fixed.
    return [g for g, _ in sorted(mr.items(), key=lambda kv: (kv[1], kv[0]))][:TOPK]


def run_compartment(cp):
    cfg = COMP[cp]; keys = list(cfg["ds"].keys())
    ids = {k: cfg["id"].format(k=k) for k in keys}
    P = {k: panels_of(ids[k]) for k in keys}                     # panels per dataset
    methods = ["RECAST_M0"] + sorted(m for m in P[keys[0]] if m != "RECAST_M0") + ["random"]
    prow, lrow = [], []
    for tk in keys:                                              # target loaded once
        stem = cfg["ds"][tk]; t0 = time.time()
        X, genes, gi = load_target(stem)
        y_full = np.array([l.strip() for l in open(f"{LAB}/{LABF[stem]}") if l.strip()])
        assert len(y_full) == X.shape[0], f"{stem}: {len(y_full)} != {X.shape[0]}"
        st_tgt = {s for s in states_of(P[tk]) if (y_full == s).sum() >= MIN_CELLS}
        gv = np.array([g for g in genes])
        # ---------- Protocol ① pairwise: sources A != target ----------
        for ak in (keys if ONLY in ("both", "pairwise") else []):
            if ak == tk:
                continue
            S = sorted(states_of(P[ak]) & st_tgt)
            if len(S) < 2:
                continue
            rng = np.random.RandomState(0)
            Xs, ys = subset(y_full, X, S, rng)
            rp_a = rand_panel(gv, S, np.random.RandomState(1)); rp_b = rand_panel(gv, S, np.random.RandomState(2))
            for M in methods:
                pa = P[ak].get(M) if M != "random" else rp_a
                pb = P[tk].get(M) if M != "random" else rp_b
                if pa is None or pb is None:
                    continue
                tr = eval_panel(pa, Xs, ys, gi, S)              # transfer: A-panel on B
                na = eval_panel(pb, Xs, ys, gi, S)              # same-dataset ceiling: B's panel on B|S (column 'native')
                for met in ("knn", "sum_max", "ontarget", "auroc"):
                    prow.append(dict(compartment=cp, method=M, src=ak, tgt=tk, n_states=len(S),
                                     metric=met, transfer=tr[met], native=na[met]))
            print(f"[{cp}] pair {ak}->{tk} S={len(S)} ({time.time()-t0:.0f}s)", flush=True)
        # ---------- Protocol ② LODO: consensus of others on this held-out target ----------
        others = [k for k in keys if k != tk] if ONLY in ("both", "lodo") else []
        for M in methods if others else []:
            if M == "random":
                continue
            src_panels = [P[k][M] for k in others if M in P[k]]
            if len(src_panels) < 2:
                continue
            cons = {s: consensus(src_panels, s) for s in st_tgt}
            cons = {s: g for s, g in cons.items() if g}
            S = sorted(set(cons) & st_tgt)
            if len(S) < 2 or P[tk].get(M) is None:
                continue
            rng = np.random.RandomState(0)
            Xs, ys = subset(y_full, X, S, rng)
            tr = eval_panel(cons, Xs, ys, gi, S)
            na = eval_panel(P[tk][M], Xs, ys, gi, S)
            for met in ("knn", "sum_max", "ontarget", "auroc"):
                lrow.append(dict(compartment=cp, method=M, held_out=tk, n_states=len(S),
                                 metric=met, consensus=tr[met], native=na[met]))
        print(f"[{cp}] target {tk} done ({time.time()-t0:.0f}s)", flush=True)
        del X
    return prow, lrow


if __name__ == "__main__":
    print(f"[transfer] VTAG={VTAG!r}  ONLY={ONLY}  GEN={GEN}  PAN={PAN}", flush=True)
    comps = sys.argv[1:] or list(COMP.keys())
    allp, alll = [], []
    for cp in comps:
        p, l = run_compartment(cp)
        allp += p; alll += l
        # only write the halves this run actually produced -- a lodo-only run must not blank the
        # pairwise table (and vice versa)
        if ONLY in ("both", "pairwise"):
            pd.DataFrame(allp).to_csv(f"{OUT}/arm3_transfer_pairwise{VTAG}.csv", index=False)
        if ONLY in ("both", "lodo"):
            pd.DataFrame(alll).to_csv(f"{OUT}/arm3_transfer_lodo{VTAG}.csv", index=False)
        print(f"[{cp}] written incrementally", flush=True)
    print(f"[transfer done] pairwise {len(allp)} rows, lodo {len(alll)} rows", flush=True)
