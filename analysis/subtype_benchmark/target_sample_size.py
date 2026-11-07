#!/usr/bin/env python
"""Target sample-size sensitivity of population RECAST (Fig. 2g), with repeated random partitions.  Env SC, GPU.
Honours RECAST_RESULT_TAG (use _v3) and N_PARTITIONS (default 20).

For every subtype of the 61-unit set and every group size n in {2, 5, 10, 25, 50, 100} (n < N), the subtype's cells are
partitioned R times with a stable seed (crc32 of "lineage|state|partition"); for each partition, per group: target
profile C_g, direction u_g = normalize(mean z_g - mean z_R), one integrated-gradients pass from C_R to C_g, the dC > 0
gate, and the ranking is the mean over groups (helpers imported from aggregation_and_encoder.py).  A ragged last group
is kept only if it has at least n / 2 cells.  The two ends (n = 1 and n = N) do not depend on the partition; they are
computed once per subtype and compared with results/mr4_dose<tag>.csv of aggregation_and_encoder.py.

Outputs (results/):
    mr4_dose_partitions<tag>.csv   one row per (subtype, partition, n): recall, n_groups, seed
    mr4_dose_endpoints<tag>.csv    per subtype: recall at n = 1 and n = N, here and in mr4_dose<tag>.csv"""
import os, sys, time, zlib
import numpy as np, pandas as pd, torch

FSB = "/path/to/research/analysis/subtype_benchmark"
if FSB not in sys.path:
    sys.path.insert(0, FSB)
import aggregation_and_encoder as M                                   # noqa: E402  helpers of the group-size curve

R = int(os.environ.get("N_PARTITIONS", "20"))
SIZES_MID = [n for n in M.SIZES if n > 1]                        # 2, 5, 10, 25, 50, 100


def seed_for(lin, c, r):
    return zlib.crc32(f"{lin}|{c}|{r}".encode("utf-8"))         # stable across processes and machines


def main():
    import contrastive_ig
    from captum.attr import IntegratedGradients
    mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    dev = contrastive_ig.dev

    def group_attr(Cg, Cr, U):                                    # as in aggregation_and_encoder.main
        out = np.zeros_like(Cg)
        for s in range(0, len(Cg), M.BATCH):
            cg, u = Cg[s:s + M.BATCH], U[s:s + M.BATCH]
            inp = torch.tensor(cg, device=dev).requires_grad_(True)
            base = torch.tensor(np.tile(Cr, (len(cg), 1)), device=dev)
            a = ig.attribute(inp, baselines=base, target=0,
                             additional_forward_args=(torch.tensor(u, device=dev),),
                             n_steps=M.NSTEPS, internal_batch_size=M.BATCH * 8).detach().cpu().numpy()
            a = np.where(cg - Cr[None] > 0, np.clip(a, 0, None), 0.0)
            ss = a.sum(1, keepdims=True); ss[ss == 0] = 1.0
            out[s:s + M.BATCH] = (a / ss * 1000.0)
        return out

    factorial = pd.read_csv(f"{M.OUT}/mr4_dose{M.VTAG}.csv")          # aggregation_and_encoder.py
    factorial["unit"] = factorial.lineage + "|" + factorial.state
    out_rows = f"{M.OUT}/mr4_dose_partitions{M.VTAG}.csv"; out_ends = f"{M.OUT}/mr4_dose_endpoints{M.VTAG}.csv"
    rows, ends = [], []
    lineages = sorted(M.C.NEW_IDS + list(M.EXT_ZHAO))
    print(f"{len(lineages)} lineages | K={M.K} | R={R} partitions | sizes {SIZES_MID} (+ ends once)", flush=True)
    for lin in lineages:
        t0 = time.time()
        Craw, lab, states, genes = M.load_lineage(lin)
        genes = list(genes)
        Xln = M.lognorm_rows(Craw)
        Z = contrastive_ig.embed(Xln)
        uniq = M.unique_gt(states, genes)
        for c in states:
            mk = uniq[c]
            if not mk:
                continue
            tgt = np.where(lab == c)[0]; ref = np.where(lab != c)[0]
            if len(tgt) < 2 or len(ref) < 2:
                continue
            unit = f"{lin}|{c}"
            C_R = M.fc.centroid(Craw, ref).astype("float32")
            zr = Z[ref].mean(0)

            def run_groups(grp):
                Cg = np.vstack([M.fc.centroid(Craw, g) for g in grp]).astype("float32")
                U = np.vstack([Z[g].mean(0) - zr for g in grp])
                U = (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)).astype("float32")
                A = group_attr(Cg, C_R, U)
                return M.recall(np.argsort(-A.mean(0)), genes, mk)

            r1 = run_groups([np.array([i]) for i in tgt])           # n = 1 (partition-invariant)
            rN = run_groups([tgt])                                    # n = N (partition-invariant)
            s1 = factorial[(factorial.unit == unit) & (factorial.n_cells == 1)].recall
            sN = factorial[(factorial.unit == unit) & (factorial.is_full == 1)].recall
            ends.append(dict(lineage=lin, state=c, unit=unit, n_target=len(tgt), recall_n1=r1, recall_full=rN,
                             factorial_n1=float(s1.iloc[0]) if len(s1) else np.nan,
                             factorial_full=float(sN.iloc[0]) if len(sN) else np.nan))
            for r in range(R):
                rng = np.random.default_rng(seed_for(lin, c, r))
                perm = rng.permutation(tgt)
                for n in [n for n in SIZES_MID if n < len(tgt)]:
                    grp = [perm[i:i + n] for i in range(0, len(perm), n)]
                    grp = [g for g in grp if len(g) >= max(1, n // 2)]       # ragged last group: kept if >= n/2 cells
                    rows.append(dict(lineage=lin, state=c, unit=unit, partition=r, seed=seed_for(lin, c, r),
                                     n_cells=n, n_groups=len(grp), n_target=len(tgt), recall=run_groups(grp),
                                     recall_n1=r1, recall_full=rN))
            print(f"  {unit}: N={len(tgt)}  n1 {r1:.3f} (factorial {ends[-1]['factorial_n1']:.3f})  "
                  f"full {rN:.3f} (factorial {ends[-1]['factorial_full']:.3f})  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out_rows, index=False); pd.DataFrame(ends).to_csv(out_ends, index=False)
        print(f"  [{lin}] done in {time.time() - t0:.0f}s; {len(rows)} rows so far", flush=True)
        del Craw, Xln, Z
        torch.cuda.empty_cache()
    D = pd.DataFrame(rows); E = pd.DataFrame(ends)
    print(f"\n[endpoints] max |n1 - factorial| {np.nanmax(np.abs(E.recall_n1 - E.factorial_n1)):.4f}   "
          f"max |full - factorial| {np.nanmax(np.abs(E.recall_full - E.factorial_full)):.4f}", flush=True)
    pm = D.groupby(["unit", "n_cells"]).recall.mean().reset_index()
    print("mean over subtypes of the partition-mean recall, by n:")
    print(pm.groupby("n_cells").recall.agg(["mean", "count"]).round(4).to_string(), flush=True)
    print(f"[wrote] {out_rows}, {out_ends}", flush=True)


if __name__ == "__main__":
    main()
