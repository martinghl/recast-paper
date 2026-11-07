"""Rankings behind Fig. 2f, g: the four pipelines and the group-size curve on the 97 benchmark units, with the top 20
genes of every ranking written out (scored by score_groups.py).

  * cells, labels and gene order: select_recast.load_lineage; for the added lineages its module globals point at
    added_lineages, as added_lineages/select_recast_added.py does;
  * transform, centroids, K, NSTEPS, BATCH, the group sizes: aggregation_and_encoder (imported);
  * the IG step: group_attr, as in aggregation_and_encoder.py and target_sample_size.py;
  * the population ranking without encoder (a21): dC = C_T - C_R, genes with dC > 0;
  * the two endpoints (n = 1, n = N) and R = 20 stably seeded partitions at n in 2..100 (n < N), with the
    ragged-tail rule of target_sample_size.py (seed_for imported).
Units: the marker panels in ground_truth/panels/marker_panels.json.

Env SC, GPU.  Usage: CUDA_VISIBLE_DEVICES=1 python -I group_rankings.py LINEAGE [LINEAGE ...]
Writes out/orders/<lineage>.csv.gz (one row per ranking: unit, kind, partition, seed, n_cells, n_groups, n_target,
top20); a lineage whose file exists is skipped.
"""
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

SB = "/path/to/research/analysis"
FSB = f"{SB}/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (f"{FULL_BENCHMARK}/added_lineages", SB, FSB):
    if p not in sys.path:
        sys.path.insert(0, p)
import aggregation_and_encoder as M          # noqa: E402  helpers and constants
import target_sample_size as P         # noqa: E402  seed_for, R, SIZES_MID
import select_recast as S             # noqa: E402  the cells the benchmark selects from
import added_datasets as D               # noqa: E402

UNIT_GT = __import__("json").load(open(f"{FULL_BENCHMARK}/ground_truth/panels/marker_panels.json"))
OUT = f"{HERE}/out/orders"


def main(lineages):
    assert P.R == 20 and P.SIZES_MID == [2, 5, 10, 25, 50, 100] and M.K == 20 and M.NSTEPS == 50
    added_lineages = [l for l in lineages if l in D.NEW_IDS]
    assert not added_lineages or len(added_lineages) == len(lineages), "run added and original lineages in separate processes"
    if added_lineages:                                      # as added_lineages/select_recast_added.py
        S.C = types.SimpleNamespace(LINEAGES=D.LINEAGES, NEW_IDS=D.NEW_IDS)
        S.ATLASES = f"{FULL_BENCHMARK}/data/atlases"
    import contrastive_ig
    from captum.attr import IntegratedGradients
    mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    dev = contrastive_ig.dev

    def group_attr(Cg, Cr, U):                    # as in aggregation_and_encoder.main / target_sample_size.main
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

    os.makedirs(OUT, exist_ok=True)
    for lin in lineages:
        path = f"{OUT}/{lin}.csv.gz"
        if os.path.exists(path):
            print(f"[{lin}] exists, skipped", flush=True)
            continue
        t0 = time.time()
        Craw, lab, states, genes = S.load_lineage(lin)
        genes = list(genes)
        Xln = M.lognorm_rows(Craw)
        Z = contrastive_ig.embed(Xln)
        units = [c for c in states if f"{lin}|{c}" in UNIT_GT]
        assert set(units) == {u.split("|", 1)[1] for u in UNIT_GT if u.split("|", 1)[0] == lin}, lin
        rows = []
        for c in units:
            unit = f"{lin}|{c}"
            tgt = np.where(lab == c)[0]; ref = np.where(lab != c)[0]
            assert len(tgt) >= 2 and len(ref) >= 2, unit

            def keep(kind, order, partition=-1, seed=-1, n_cells=0, n_groups=0):
                rows.append(dict(unit=unit, kind=kind, partition=partition, seed=seed, n_cells=n_cells,
                                 n_groups=n_groups, n_target=len(tgt),
                                 top20=";".join(genes[j] for j in order[:M.K])))

            C_R64 = M.fc.centroid(Craw, ref)
            dC = M.fc.centroid(Craw, tgt) - C_R64                     # a21, as aggregation_and_encoder.py
            keep("a21", np.argsort(-np.where(dC > 0, dC, -np.inf)), n_cells=len(tgt), n_groups=1)
            C_R = C_R64.astype("float32")                             # as target_sample_size.py from here on
            zr = Z[ref].mean(0)

            def run_groups(grp):
                Cg = np.vstack([M.fc.centroid(Craw, g) for g in grp]).astype("float32")
                U = np.vstack([Z[g].mean(0) - zr for g in grp])
                U = (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)).astype("float32")
                return np.argsort(-group_attr(Cg, C_R, U).mean(0))

            keep("n1", run_groups([np.array([i]) for i in tgt]), n_cells=1, n_groups=len(tgt))
            keep("full", run_groups([tgt]), n_cells=len(tgt), n_groups=1)
            for r in range(P.R):
                seed = P.seed_for(lin, c, r)
                perm = np.random.default_rng(seed).permutation(tgt)
                for n in [n for n in P.SIZES_MID if n < len(tgt)]:
                    grp = [perm[i:i + n] for i in range(0, len(perm), n)]
                    grp = [g for g in grp if len(g) >= max(1, n // 2)]       # ragged-tail rule of target_sample_size.py
                    keep("part", run_groups(grp), partition=r, seed=seed, n_cells=n, n_groups=len(grp))
            print(f"  {unit}: N={len(tgt)}  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip")
        os.replace(path + ".tmp", path)
        print(f"[{lin}] {len(units)} units, {len(rows)} rankings, {time.time() - t0:.0f}s", flush=True)
        del Craw, Xln, Z
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
