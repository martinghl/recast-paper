"""RECAST panels for the phase-defined cell-cycle comparisons (GPU), read from
phase_contrasts/<name>/{counts.npz, labels.txt} and written to phase_recast/<name>/{panel.csv, meta.json}, with RECAST's
recipe (recast_subtypes.py: run_lineage) in its two-group form:

    Xln  = log1p(1e4 * counts / rowsum)                     per cell, from the raw counts
    Z    = embed(Xln)                                       SCimilarity encoder
    u    = normalize(mean Z[target] - mean Z[reference])    the comparison direction
    C_t, C_r = log1p(1e4 * proportion) profiles             pooled counts, then log
    RECAST = IG(C_t ; baseline = C_r ; F = <enc(x), u>), restricted to dC = C_t - C_r > 0

The embedding is computed in row chunks (a 47k-cell comparison never builds a dense 47k x 28,231 matrix).
Env: SC (sibling_targets loads the encoder on the GPU)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")
import population_ig as L                                    # noqa: E402  cluster_countsums / props_and_C / go
from sibling_targets import attr_dir, embed, go, ig_of       # noqa: E402  loads SCimilarity on GPU

CR = "/path/to/research/analysis/cell_cycle"
SRC = f"{CR}/phase_contrasts"
OUT = f"{CR}/phase_recast"
TOPK = 200          # deep enough for precision@200
CHUNK = 4096


def lognorm_chunk(C):
    """log1p(1e4 * x / rowsum) for a CSR chunk -> dense float32."""
    X = np.asarray(C.todense(), dtype="float32")
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * X / tot)


def run(name):
    d = f"{SRC}/{name}"
    counts = sp.load_npz(f"{d}/counts.npz").tocsr()
    y = np.array([l.strip() for l in open(f"{d}/labels.txt") if l.strip()])
    assert len(y) == counts.shape[0], f"{name}: {len(y)} labels vs {counts.shape[0]} cells"
    assert counts.shape[1] == len(go), f"{name}: {counts.shape[1]} genes, expected {len(go)}"
    if counts.nnz and not np.all(counts.data == np.round(counts.data)):
        raise RuntimeError(f"{name}: counts.npz is not integer")
    n_t, n_r = int((y == "target").sum()), int((y == "reference").sum())
    print(f"=== {name}: {counts.shape[0]} cells ({n_t} target / {n_r} reference) ===", flush=True)

    # ---- contrast direction in embedding space (chunked so memory stays flat) ----
    Z = np.vstack([embed(lognorm_chunk(counts[i:i + CHUNK]))
                   for i in range(0, counts.shape[0], CHUNK)])
    u = Z[y == "target"].mean(0) - Z[y == "reference"].mean(0)
    u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
    del Z

    # ---- proportion-then-log pseudobulk endpoints, then positive-channel IG ----
    CS = L.cluster_countsums(counts, y, ["target", "reference"], np.asarray(go))
    _, C = L.props_and_C(CS)
    C_t, C_r = C[0], C[1]
    dC = C_t - C_r
    m0 = np.where(dC > 0, attr_dir(C_t, C_r, u), -np.inf)
    n_pos = int((dC > 0).sum())

    order = np.argsort(-m0)[:TOPK]
    order = [int(j) for j in order if np.isfinite(m0[int(j)])]
    panel = pd.DataFrame(dict(rank=np.arange(1, len(order) + 1),
                              gene=[ig_of[j] for j in order],
                              m0=[float(m0[j]) for j in order],
                              dC=[float(dC[j]) for j in order]))
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    panel.to_csv(f"{OUT}/{name}/panel.csv", index=False)
    json.dump(dict(contrast=name, n_target=n_t, n_reference=n_r, n_genes=int(counts.shape[1]),
                   n_positive_channel=n_pos, n_panel=len(order), topk_requested=TOPK,
                   method="RECAST_M0", recipe="proportion-then-log pseudobulk; IG(C_t;C_r;<enc,u>); dC>0"),
              open(f"{OUT}/{name}/meta.json", "w"), indent=1)
    print(f"    dC>0 on {n_pos}/{len(go)} genes | panel {len(order)} | top10 "
          f"{', '.join(panel.gene.head(10))}", flush=True)
    return dict(contrast=name, n_target=n_t, n_reference=n_r,
                n_positive_channel=n_pos, n_panel=len(order),
                top10=";".join(panel.gene.head(10)))


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or sorted(n for n in os.listdir(SRC)
                                   if os.path.exists(f"{SRC}/{n}/counts.npz"))
    rows = []
    for n in names:
        try:
            rows.append(run(n))
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append(dict(contrast=n, n_panel=f"ERROR {type(e).__name__}: {e}"))
    df = pd.DataFrame(rows)
    f = f"{CR}/phase_recast_panels.csv"
    if os.path.exists(f) and len(names) < len(os.listdir(SRC)):
        old = pd.read_csv(f)
        df = pd.concat([old[~old.contrast.isin(df.contrast)], df], ignore_index=True)
    df.to_csv(f, index=False)
    print("\n=== RECAST SELECTION SUMMARY ===")
    print(df.drop(columns=["top10"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
