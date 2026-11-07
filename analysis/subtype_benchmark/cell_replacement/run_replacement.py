"""GPU stage of the cell-replacement experiment (Supplementary Fig. 1b, c).

For each of the 61 units (lineage | subtype T):
  * target cells = T's cells (<= CAP, seeded); each gets one sibling partner (same donor when possible), shared by all
    methods and all K;
  * for RECAST / COSG / Wilcoxon and K in {5,10,20,50}: replace the K genes by the partner's values (primary) or set
    them to zero (sensitivity), re-embed with the SCimilarity encoder, and record identity loss (k-NN majority among
    the original lineage embeddings, the cell itself and its partner excluded), shift along the target -> sibling axis
    and orthogonal displacement;
  * expression-matched random null (replace, K = 20, 100 draws per method);
  * sibling control: zero T's K = 20 genes in sibling cells and check they keep their own identity.
Writes results/*.csv and emb/<lineage>.npz.  Env SC, CUDA_VISIBLE_DEVICES=1.
usage: python run_replacement.py [--only LINEAGE ...] [--out results] [--emb emb] [--all_k]
  --all_k also stores Z1 / pred for every K (keys ...|K5, used by the example dose grid)"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replacement_lib as L

MODES = ("replace", "zero")


def run_lineage(lineage, states, enc, emb_dir=None, log=print, all_k=False):
    t0 = time.time()
    lin = L.Lineage(lineage)
    gl = L.gene_lists(lineage)
    labels = lin.labels; code = {s: i for i, s in enumerate(lin.states)}
    lab_code = np.array([code[s] for s in labels])
    H0 = enc.H0(lin.X)                                           # [n, 1024] first-Linear output, all lineage cells
    Z0 = enc.head(H0)                                            # original embeddings (unit norm)
    knn = L.KNN(Z0, labels, lin.states)
    Xc = lin.X.tocsc()
    log(f"[{lineage}] n={lin.n} states={lin.states} encode {time.time() - t0:.1f}s")

    main, null, sibc, units, emb = [], [], [], [], {"Z0": Z0.half().cpu().numpy(), "labels": labels,
                                                    "cells": lin.cells, "states": np.array(lin.states)}
    for T in states:
        unit = f"{lineage}|{T}"
        all_t = np.flatnonzero(labels == T); sib = np.flatnonzero(labels != T)
        assert len(all_t) >= 50 and len(sib) > 0, unit
        rng = np.random.default_rng(L.seed_of("targets", unit))
        tgt = np.sort(rng.choice(all_t, L.CAP, replace=False)) if len(all_t) > L.CAP else all_t
        part, same = L.sibling_partners(lin, tgt, sib, unit)
        excl = [tgt, part]
        Ht = H0[tgt]
        Zt = enc.head(Ht)                                        # same batch shape as every edited call: a zero edit is exactly zero
        pred0 = knn.predict(Zt, excl)
        anchored = pred0 == code[T]
        u = torch.nn.functional.normalize(Z0[labels == T].mean(0) - Z0[labels != T].mean(0), dim=0)
        dist = float(((Z0[labels == T].mean(0) - Z0[labels != T].mean(0)) @ u).item())
        Xt = Xc[tgt].tocsc(); Xp = Xc[part].tocsc()

        def evaluate(cols, mode):
            cols = np.asarray(cols, dtype=np.int64)
            xt = Xt[:, cols].toarray().astype(np.float32)
            dX = (Xp[:, cols].toarray().astype(np.float32) - xt) if mode == "replace" else -xt
            Z1 = enc.edited(Ht, dX, cols)
            pred1 = knn.predict(Z1, excl)
            r = L.readouts(Zt, Z1, pred1, code[T], anchored, u, dist)
            a = anchored
            r["to_partner_state"] = float((pred1[a] == lab_code[part][a]).mean()) if (mode == "replace" and a.any()) else np.nan
            r["mean_abs_dx"] = float(np.abs(dX).sum(1).mean())
            return r, Z1, pred1

        skipped, cols_of = {}, {}
        for m in L.METHODS:
            genes = L.pick(gl[m], T)
            assert len(genes) > 0, (unit, m)
            for K in L.KS:
                cols, sk = L.top_in_vocab(genes, lin.gi, K)
                cols_of[(m, K)] = cols; skipped[(m, K)] = sk
                for mode in MODES:
                    r, Z1, pred1 = evaluate(cols, mode)
                    main.append(dict(lineage=lineage, state=T, unit=unit, method=m, mode=mode, K=K, n_genes=len(cols),
                                     n_skipped_oov=sk, n_target=len(tgt), **r))
                    if K == L.K_MAIN:
                        emb[f"Z1|{T}|{m}|{mode}"] = Z1.half().cpu().numpy()
                        # exact per-cell assignment (the float16 Z1 above can flip near-ties when re-voted)
                        emb[f"pred|{T}|{m}|{mode}"] = pred1.astype(np.int16)
                    if all_k:
                        emb[f"Z1|{T}|{m}|{mode}|K{K}"] = Z1.half().cpu().numpy()
                        emb[f"pred|{T}|{m}|{mode}|K{K}"] = pred1.astype(np.int16)
        # expression-matched random null, replace, K=20, separately for each method's genes
        excl_genes = set().union(*[set(cols_of[(m, 50)].tolist()) for m in L.METHODS])
        for m in L.METHODS:
            for d, cols in enumerate(L.null_gene_sets(lin, tgt, cols_of[(m, L.K_MAIN)], excl_genes, unit, m)):
                r, _, _ = evaluate(cols, "replace")
                null.append(dict(unit=unit, method=m, draw=d, identity_loss=r["identity_loss"],
                                 axis_shift=r["axis_shift"], orth_shift=r["orth_shift"], mean_abs_dx=r["mean_abs_dx"]))
        # sibling control: zero T's K=20 genes in sibling cells
        rng = np.random.default_rng(L.seed_of("sibctrl", unit))
        sc = np.sort(rng.choice(sib, L.CAP, replace=False)) if len(sib) > L.CAP else sib
        p0 = knn.predict(enc.head(H0[sc]), [sc]); anc = p0 == lab_code[sc]
        Xs = Xc[sc].tocsc()
        for m in L.METHODS:
            cols = cols_of[(m, L.K_MAIN)]
            dX = -Xs[:, cols].toarray().astype(np.float32)
            Z1 = enc.edited(H0[sc], dX, cols)
            p1 = knn.predict(Z1, [sc])
            sibc.append(dict(unit=unit, method=m, n_sib=len(sc), n_anchored=int(anc.sum()),
                             identity_loss=float((p1[anc] != lab_code[sc][anc]).mean()),
                             to_target=float((p1[anc] == code[T]).mean())))
        top20 = {m: set(cols_of[(m, L.K_MAIN)].tolist()) for m in L.METHODS}
        jac = lambda a, b: len(top20[a] & top20[b]) / len(top20[a] | top20[b])
        units.append(dict(lineage=lineage, state=T, unit=unit, n_lineage=lin.n, n_state_cells=len(all_t),
                          n_target=len(tgt), n_sibling_cells=len(sib), n_anchored=int(anchored.sum()),
                          anchored_frac=float(anchored.mean()), same_donor_frac=float(same.mean()),
                          donor_map=lin.donor[0] is not None, centroid_dist=dist,
                          jaccard_recast_cosg=jac("RECAST", "COSG"), jaccard_recast_wilcoxon=jac("RECAST", "Wilcoxon"),
                          jaccard_cosg_wilcoxon=jac("COSG", "Wilcoxon"),
                          **{f"oov_skipped_{m}_K50": skipped[(m, 50)] for m in L.METHODS}))
        emb[f"tgt|{T}"] = tgt; emb[f"part|{T}"] = part; emb[f"anchored|{T}"] = anchored
        log(f"[{unit}] target {len(tgt)} anchored {anchored.mean():.2f} same-donor {same.mean():.2f} "
            f"| K20 replace loss " + " ".join(f"{m}={next(r['identity_loss'] for r in main[::-1] if r['unit'] == unit and r['method'] == m and r['K'] == 20 and r['mode'] == 'replace'):.3f}" for m in L.METHODS))
    if emb_dir:
        os.makedirs(emb_dir, exist_ok=True)
        np.savez_compressed(f"{emb_dir}/{lineage}.npz", **{k: np.asarray(v) for k, v in emb.items()})
    log(f"[{lineage}] done {time.time() - t0:.1f}s")
    return pd.DataFrame(main), pd.DataFrame(null), pd.DataFrame(sibc), pd.DataFrame(units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=f"{L.CD}/results")
    ap.add_argument("--emb", default=f"{L.CD}/emb")
    ap.add_argument("--all_k", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    by_lin = {}
    for lin, st in L.units61():
        by_lin.setdefault(lin, []).append(st)
    if a.only:
        by_lin = {k: v for k, v in by_lin.items() if k in a.only}
    enc = L.Enc("cuda")
    os.makedirs(a.out, exist_ok=True)
    parts = {k: [] for k in ("main", "null", "sibling_ctrl", "units")}
    for lin in sorted(by_lin):
        for k, df in zip(parts, run_lineage(lin, by_lin[lin], enc, a.emb, all_k=a.all_k)):
            parts[k].append(df)
        for k in parts:                                          # rewrite after every lineage: a crash keeps progress
            pd.concat(parts[k], ignore_index=True).to_csv(f"{a.out}/celldel_{k}.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
