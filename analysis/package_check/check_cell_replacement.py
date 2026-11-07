"""Supplementary Fig. 1b,c: replacing a subtype's top genes in single cells, recomputed with the package.

subtype_benchmark/cell_replacement/run_replacement.py took the 61 benchmark subtypes, replaced each method's top K genes
(K = 5, 10, 20, 50) in up to 2,000 cells by a sibling cell's values (or by zero), re-embedded the cells with
SCimilarity and recorded identity loss (15-nearest-neighbour vote among the unedited cells), the shift along the
subtype axis and the orthogonal displacement; 100 expression-matched random gene sets per method gave the null, and
zeroing a subtype's genes in sibling cells gave the specificity control. It read the RECAST genes from the
benchmark panels and re-embedded through an update of the encoder's first-layer output.

Here the RECAST genes are the package's (recast.cluster_attribution, written by check_benchmark_panels.py), the
unedited cells are embedded with SCimilarityEncoder.embed on raw counts, and every edited cell is re-embedded whole
with SCimilarityEncoder.torch_encode on its log-normalized profile. Cells, sibling partners, null gene sets, the
neighbour vote and the read-outs are replacement_lib's. Compared with celldel_main.csv, celldel_null.csv and
celldel_sibling_ctrl.csv.

    python check_cell_replacement.py [lineage ...]
Writes cell_replacement_<main|null|sibling>.csv."""
import os
import sys

import numpy as np
import pandas as pd
import torch

import common as C

CD = os.path.join(C.FSB, "cell_replacement")
sys.path.insert(0, CD)
import replacement_lib as L  # noqa: E402

MODES = ("replace", "zero")
KEYS = {"main": ["unit", "method", "mode", "K"], "null": ["unit", "method", "draw"], "sibling": ["unit", "method"]}
STORED = {"main": "celldel_main.csv", "null": "celldel_null.csv", "sibling": "celldel_sibling_ctrl.csv"}
VALS = {"main": ["n_genes", "n_anchored", "identity_loss", "axis_shift", "orth_shift", "to_partner_state", "mean_abs_dx"],
        "null": ["identity_loss", "axis_shift", "orth_shift", "mean_abs_dx"],
        "sibling": ["n_anchored", "identity_loss", "to_target"]}


def run_lineage(lineage, states, enc, out):
    lin = L.Lineage(lineage)
    dev = torch.device("cuda")
    gl = L.gene_lists(lineage)
    gl["RECAST"] = C.panel_lists(os.path.join(C.OUT, "panels", f"topN200_{lineage}_recast.csv"))
    code = {s: i for i, s in enumerate(lin.states)}
    labels = lin.labels
    lab_code = np.array([code[s] for s in labels])
    Z0 = torch.as_tensor(enc.embed(lin.Craw), dtype=torch.float32, device=dev)
    knn = L.KNN(Z0, labels, lin.states)
    Xc = lin.X.tocsr()

    def encode(Xd):
        with torch.inference_mode():
            return enc.torch_encode(Xd)

    for T in states:
        unit = f"{lineage}|{T}"
        all_t = np.flatnonzero(labels == T); sib = np.flatnonzero(labels != T)
        rng = np.random.default_rng(L.seed_of("targets", unit))
        tgt = np.sort(rng.choice(all_t, L.CAP, replace=False)) if len(all_t) > L.CAP else all_t
        part, _ = L.sibling_partners(lin, tgt, sib, unit)
        excl = [tgt, part]
        Xt = torch.as_tensor(Xc[tgt].toarray(), dtype=torch.float32, device=dev)
        Xp = torch.as_tensor(Xc[part].toarray(), dtype=torch.float32, device=dev)
        Zt = encode(Xt)
        anchored = knn.predict(Zt, excl) == code[T]
        dz = Z0[torch.as_tensor(labels == T, device=dev)].mean(0) - Z0[torch.as_tensor(labels != T, device=dev)].mean(0)
        u = torch.nn.functional.normalize(dz, dim=0)
        dist = float((dz @ u).item())

        def evaluate(cols, mode):
            c = torch.as_tensor(np.asarray(cols, dtype=np.int64), device=dev)
            X1 = Xt.clone()
            X1[:, c] = Xp[:, c] if mode == "replace" else 0.0
            Z1 = encode(X1)
            pred1 = knn.predict(Z1, excl)
            r = L.readouts(Zt, Z1, pred1, code[T], anchored, u, dist)
            dX = (X1[:, c] - Xt[:, c]).abs().sum(1).mean().item()
            r["to_partner_state"] = (float((pred1[anchored] == lab_code[part][anchored]).mean())
                                     if (mode == "replace" and anchored.any()) else np.nan)
            r["mean_abs_dx"] = float(dX)
            return r

        cols_of = {}
        for m in L.METHODS:
            genes = L.pick(gl[m], T)
            for K in L.KS:
                cols, sk = L.top_in_vocab(genes, lin.gi, K)
                cols_of[(m, K)] = cols
                for mode in MODES:
                    out["main"].append(dict(unit=unit, method=m, mode=mode, K=K, n_genes=len(cols), **evaluate(cols, mode)))
        excl_genes = set().union(*[set(cols_of[(m, 50)].tolist()) for m in L.METHODS])
        for m in L.METHODS:
            for d, cols in enumerate(L.null_gene_sets(lin, tgt, cols_of[(m, L.K_MAIN)], excl_genes, unit, m)):
                r = evaluate(cols, "replace")
                out["null"].append(dict(unit=unit, method=m, draw=d, **{k: r[k] for k in VALS["null"]}))
        rng = np.random.default_rng(L.seed_of("sibctrl", unit))
        sc = np.sort(rng.choice(sib, L.CAP, replace=False)) if len(sib) > L.CAP else sib
        Xs = torch.as_tensor(Xc[sc].toarray(), dtype=torch.float32, device=dev)
        anc = knn.predict(encode(Xs), [sc]) == lab_code[sc]
        for m in L.METHODS:
            X1 = Xs.clone()
            X1[:, torch.as_tensor(cols_of[(m, L.K_MAIN)], device=dev)] = 0.0
            p1 = knn.predict(encode(X1), [sc])
            out["sibling"].append(dict(unit=unit, method=m, n_anchored=int(anc.sum()),
                                       identity_loss=float((p1[anc] != lab_code[sc][anc]).mean()),
                                       to_target=float((p1[anc] == code[T]).mean())))
        print(f"  {unit}: {len(tgt)} cells, anchored {anchored.mean():.2f}", flush=True)


def main():
    enc = C.scimilarity()
    by_lin = {}
    for lin, st in L.units61():
        by_lin.setdefault(lin, []).append(st)
    names = sys.argv[1:] or sorted(by_lin)
    out = {k: [] for k in KEYS}
    for lin in names:
        run_lineage(lin, by_lin[lin], enc, out)
        C.assert_package_only()
    tag = "" if not sys.argv[1:] else "_" + "_".join(names)
    for name, keys in KEYS.items():
        P = pd.DataFrame(out[name])
        stored = pd.read_csv(os.path.join(CD, "results", STORED[name]))
        M = P.merge(stored[keys + VALS[name]], on=keys, how="left", suffixes=("", "_stored"), validate="one_to_one")
        assert M[f"{VALS[name][-1]}_stored"].notna().any(), name
        cols = VALS[name]
        same = np.ones(len(M), dtype=bool)
        msg = []
        for c in cols:
            a, b = M[c].to_numpy(float), M[f"{c}_stored"].to_numpy(float)
            eq = np.isclose(a, b, rtol=0, atol=1e-12) | (np.isnan(a) & np.isnan(b))
            same &= eq if c in ("n_genes", "n_anchored", "identity_loss", "to_partner_state", "to_target") else True
            msg.append(f"{c} equal {int(eq.sum())}/{len(M)}, max |d| {np.nanmax(np.abs(a - b)) if len(M) else 0:.3g}")
        M["counts_equal"] = same
        print(f"[{name}] {len(M)} rows; " + "; ".join(msg), flush=True)
        C.write(M, f"cell_replacement_{name}{tag}.csv")


if __name__ == "__main__":
    main()
