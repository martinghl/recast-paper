"""Export each phase-defined cell-cycle comparison for the comparator methods on the CPU server, as
the benchmark harness's `_allgenes` files:
    <name>_counts.mtx.gz   genes x cells, integer counts
    <name>_genes.txt       gene symbols, in the order of the matrix rows
    <name>_labels.txt      "target" / "reference", in the order of the matrix columns

Genes detected in >= 3 cells of the comparison are kept.  Values are checked to be integer and within int32 range
before the cast.  Env: SC."""
import gzip
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

CR = "/path/to/research/analysis/cell_cycle"
SRC = f"{CR}/phase_contrasts"
OUT = f"{CR}/phase_sce_export"
MIN_CELLS = 3
INT32_MAX = np.iinfo(np.int32).max


def export(name):
    d = f"{SRC}/{name}"
    C = sp.load_npz(f"{d}/counts.npz").tocsr()
    y = np.array([l.strip() for l in open(f"{d}/labels.txt") if l.strip()])
    meta = json.load(open(f"{d}/meta.json"))
    genes = np.asarray(json.load(open(f"{CR}/cc_gene_order.json"))) if os.path.exists(
        f"{CR}/cc_gene_order.json") else None
    assert genes is not None, "run with cc_gene_order.json present (written by main())"
    assert len(y) == C.shape[0] and len(genes) == C.shape[1], f"{name}: shape/label mismatch"

    det = np.asarray((C > 0).sum(0)).ravel()
    keep = det >= MIN_CELLS
    C = C[:, keep]
    g = genes[keep]

    # --- integer check before the int32 cast ---
    if C.nnz:
        if not np.all(C.data == np.round(C.data)):
            raise RuntimeError(f"{name}: matrix is not integral -- refusing to cast to int32 "
                               "(it holds log-normalized values?)")
        if C.data.max() > INT32_MAX:
            raise RuntimeError(f"{name}: max {C.data.max()} exceeds int32")

    os.makedirs(OUT, exist_ok=True)
    mtx = f"{OUT}/{name}_counts.mtx"
    sio.mmwrite(mtx, C.T.tocoo().astype(np.int32))
    with open(mtx, "rb") as fi, gzip.open(mtx + ".gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(mtx)
    pd.Series(g).to_csv(f"{OUT}/{name}_genes.txt", index=False, header=False)
    pd.Series(y).to_csv(f"{OUT}/{name}_labels.txt", index=False, header=False)
    print(f"[{name}] {C.shape[0]} cells x {C.shape[1]} genes "
          f"({meta['n_target']} target / {meta['n_reference']} reference) max={int(C.data.max())}",
          flush=True)
    return dict(contrast=name, n_cells=int(C.shape[0]), n_genes=int(C.shape[1]),
                n_target=meta["n_target"], n_reference=meta["n_reference"],
                max_count=int(C.data.max()) if C.nnz else 0)


def main():
    # freeze the 28,231-gene order once so the export never depends on importing the GPU stack
    if not os.path.exists(f"{CR}/cc_gene_order.json"):
        sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
        sys.path.insert(0, "/path/to/research/external/Signature/src")
        from sibling_targets import go  # noqa: E402
        json.dump(list(map(str, go)), open(f"{CR}/cc_gene_order.json", "w"))
        print(f"wrote cc_gene_order.json ({len(go)} genes)")

    names = sys.argv[1:] or sorted(n for n in os.listdir(SRC)
                                   if os.path.exists(f"{SRC}/{n}/counts.npz"))
    rows = [export(n) for n in names]
    df = pd.DataFrame(rows)
    df.to_csv(f"{CR}/phase_sce_export.csv", index=False)
    with open(f"{OUT}/CC_FAIR_IDS.txt", "w") as fh:
        fh.write("\n".join(df.contrast) + "\n")
    print("\n=== SCE EXPORT SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
