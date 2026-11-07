#!/usr/bin/env python
"""One scGeneFit selection: one lineage, one subtype, one thread.

Timing: the other methods' selection times were measured on one CPU thread, so scGeneFit is measured the same way
(left alone the solver uses every core).  One process per subtype also lets the 61 subtypes run side by side.  Gene
choice does not depend on the thread count.  Each subtype writes its own CSV and json; assemble_scgenefit.py merges
them per lineage.

scGeneFit returns tied unit weights, i.e. a set of exactly `num_markers` genes with no internal order, so it is asked
once for 20 and once for 200 genes, and the two answers become ranks 1-20 and 21-200."""
import argparse, json, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np, scipy.sparse as sp
from scipy.io import mmread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sce", default="/path/to/cpu_server/dedicated_tools/sce_export_v2")
    ap.add_argument("--part", default="/path/to/cpu_server/dedicated_tools/sgf_parts")
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--cluster-index", type=int, required=True)
    ap.add_argument("--method", default="pairwise")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.part, exist_ok=True)
    from scGeneFit.functions import get_markers

    labels = np.array([l.strip() for l in open(f"{a.sce}/{a.lineage}_labels.txt")])
    cl = sorted(set(labels))[a.cluster_index]
    stem = f"{a.part}/{a.lineage}__{a.cluster_index}"
    if os.path.exists(f"{stem}.json"):
        print(f"[skip] {a.lineage} {cl}", flush=True); return

    X = sp.csr_matrix(mmread(f"{a.sce}/{a.lineage}_counts.mtx.gz").T)
    genes = np.array([l.strip() for l in open(f"{a.sce}/{a.lineage}_genes.txt")])
    cs = np.asarray(X.sum(1)).ravel(); cs[cs == 0] = 1.0
    Xn = sp.diags(1e4 / cs) @ X; Xn.data = np.log1p(Xn.data)
    D = np.asarray(Xn.todense(), dtype=np.float64)
    bl = np.array([1 if l == cl else 0 for l in labels])

    np.random.seed(a.seed)
    s = time.perf_counter()
    top20 = [str(genes[i]) for i in get_markers(D, bl, num_markers=20, method=a.method,
                                                epsilon=a.epsilon, verbose=False)]
    t20 = time.perf_counter() - s
    np.random.seed(a.seed)
    s = time.perf_counter()
    top200 = [str(genes[i]) for i in get_markers(D, bl, num_markers=200, method=a.method,
                                                 epsilon=a.epsilon, verbose=False)]
    t200 = time.perf_counter() - s

    ranked = top20 + [g for g in top200 if g not in set(top20)][:180]
    with open(f"{stem}.csv", "w") as fo:
        fo.write("method,cluster,rank,gene\n")
        for i, g in enumerate(ranked):
            fo.write(f'scgenefit,"{cl}",{i+1},{g}\n')
    json.dump({"lineage": a.lineage, "cluster": cl, "cluster_index": a.cluster_index,
               "seconds": t20 + t200, "seconds_budget20": t20, "seconds_budget200": t200,
               "depth": len(ranked), "n_cells": int(D.shape[0]), "n_genes": int(D.shape[1]),
               "threads": 1, "scgenefit_method": a.method, "epsilon": a.epsilon},
              open(f"{stem}.json", "w"), indent=1)
    print(f"[ok] {a.lineage} {cl}: depth {len(ranked)}  {t20:.0f}s+{t200:.0f}s", flush=True)


if __name__ == "__main__":
    main()
