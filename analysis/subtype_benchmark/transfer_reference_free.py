"""Reference-free SIGnature panels in the pairwise cross-atlas transfer (Fig. 6e).  Runs on the CPU server.

transfer_pairs.py evaluates RECAST and the comparators pairwise: for each ordered atlas pair (A, B), restricted to the
subtypes shared by A's panel and B's >= 50-cell subtypes, A's panel is scored on B (transfer) and B's own panel on B
(same-dataset ceiling).  This script runs the same protocol for one more method, the SIGnature panels
(panels/topN_<id>_signature_native.csv, select_reference_free.py), so its rows join the same table.

Everything except the panel is that of transfer_pairs.py: the same targets, the same subtype sets (defined by the RECAST
panel of the source atlas), the same RandomState(0) stratified 8,000-cell subset of the target, TOPK = 50, and
marker_free_scores.score_arm3 (kNN / sum_max / ontarget / AUROC, k in {3, 5, 10, 20}, StratifiedKFold 5, seed 20220502).

-> transfer/arm3_transfer_pairwise_signature<tag>.csv

  cd /path/to/cpu_server/analysis/subtype_benchmark
  RECAST_RESULT_TAG=_v3 setsid nohup /path/to/envs/marker_methods/bin/python \
      transfer_reference_free.py > transfer_reference_free.log 2>&1 &"""
import os, sys, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
BASE = "/path/to/cpu_server/analysis/subtype_benchmark"
H5 = "/path/to/cpu_server/analysis/atlases"
sys.path.insert(0, BASE)
import marker_free_scores as A3                      # score_arm3
import transfer_pairs as TR                # COMP / LABF / subset / eval_panel / load_target

OUT = f"{BASE}/transfer"; os.makedirs(OUT, exist_ok=True)
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
PAN, LAB = f"{BASE}/panels", f"{BASE}/bench_labels"
MIN_CELLS, CAP, TOPK = TR.MIN_CELLS, TR.CAP, TR.TOPK
METHOD = "SIGnature_native"


def panels_of(fsid):
    """{RECAST_M0: ..., SIGnature_native: ...}; the RECAST panels only define the subtype sets."""
    out = {}
    f = pd.read_csv(f"{PAN}/topN_{fsid}_recast_M0.csv").sort_values(["cluster", "rank"])
    out["RECAST_M0"] = {str(c): list(g["gene"].astype(str))[:TOPK] for c, g in f.groupby("cluster")}
    s = pd.read_csv(f"{PAN}/topN_{fsid}_signature_native.csv").sort_values(["cluster", "rank"])
    out[METHOD] = {str(c): list(g["gene"].astype(str))[:TOPK] for c, g in s.groupby("cluster")}
    return out


def run_compartment(cp):
    cfg = TR.COMP[cp]; keys = list(cfg["ds"].keys())
    ids = {k: cfg["id"].format(k=k) for k in keys}
    P = {k: panels_of(ids[k]) for k in keys}
    prow = []
    for tk in keys:
        stem = cfg["ds"][tk]; t0 = time.time()
        X, genes, gi = TR.load_target(stem)
        y_full = np.array([l.strip() for l in open(f"{LAB}/{TR.LABF[stem]}") if l.strip()])
        assert len(y_full) == X.shape[0], f"{stem}: {len(y_full)} != {X.shape[0]}"
        st_tgt = {s for s in set(P[tk]["RECAST_M0"].keys()) if (y_full == s).sum() >= MIN_CELLS}
        for ak in keys:
            if ak == tk:
                continue
            S = sorted(set(P[ak]["RECAST_M0"].keys()) & st_tgt)      # same subtype set as transfer_pairs.py
            if len(S) < 2:
                continue
            rng = np.random.RandomState(0)
            Xs, ys = TR.subset(y_full, X, S, rng)                   # same cells as transfer_pairs.py
            pa, pb = P[ak].get(METHOD), P[tk].get(METHOD)
            if pa is None or pb is None:
                continue
            tr = TR.eval_panel(pa, Xs, ys, gi, S)
            na = TR.eval_panel(pb, Xs, ys, gi, S)
            for met in ("knn", "sum_max", "ontarget", "auroc"):
                prow.append(dict(compartment=cp, method=METHOD, src=ak, tgt=tk, n_states=len(S),
                                 metric=met, transfer=tr[met], native=na[met]))
            print(f"[{cp}] pair {ak}->{tk} S={len(S)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{cp}] target {tk} done ({time.time()-t0:.0f}s)", flush=True)
        del X
    return prow


if __name__ == "__main__":
    print(f"[signature transfer] VTAG={VTAG!r} PAN={PAN}", flush=True)
    out_csv = f"{OUT}/arm3_transfer_pairwise_signature{VTAG}.csv"
    assert not os.path.exists(out_csv) or os.environ.get("FORCE"), f"{out_csv} exists"
    allp = []
    for cp in (sys.argv[1:] or list(TR.COMP.keys())):
        allp += run_compartment(cp)
        pd.DataFrame(allp).to_csv(out_csv, index=False)
        print(f"[{cp}] written incrementally -> {out_csv}", flush=True)
    print(f"[signature transfer done] {len(allp)} rows", flush=True)
