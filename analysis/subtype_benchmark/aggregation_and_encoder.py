"""Target aggregation x representation, and the group-size curve between them, on the 61 CellMarker-unique subtypes
(helpers of full_benchmark/recovery_sources/group_rankings.py and target_sample_size.py).

A population contrast without an encoder, C_T - C_R with the mean-lognorm centroid, equals the mean log fold change,
so the contrast is kept in all four cells and the two remaining factors are crossed:

                       | no encoder                    | encoder
    -------------------+-------------------------------+--------------------------------
    per-cell target    | per-cell DE (Wilcoxon tc)     | a12  per-cell contrastive attribution
    population target  | a21  mean-logFC (= C_T - C_R) | RECAST

The axis is the target side; the reference is aggregated in all four cells (a single reference cell is meaningless,
and Wilcoxon also uses the whole reference distribution).  This script computes the two cells that are not benchmark
rankings already (a12, a21); RECAST and scanpy_wilcoxontiecorrect_rankby_raw are read from fs_uniq_perstate<tag>.csv.

Group-size curve: vary only how many target cells are collapsed into one profile, holding contrast, reference,
encoder, gate and K fixed.  For group size n, the subtype's cells are split into ceil(N/n) groups; each group gets its
own profile C_g, direction u_g = normalize(mean z_g - mean z_R), one IG pass from C_R to C_g, the dC > 0 gate and the
positive part normalized to sum 1000; the subtype's ranking is the mean over groups.  The endpoints are the two encoder
cells of the factorial:

    n = 1  -> one IG per cell, averaged      == a12 (per-cell contrastive)
    n = N  -> one IG for the whole subtype   == RECAST's positive channel

Env SC, GPU.  Honours RECAST_RESULT_TAG.  Writes results/mr4_factorial<tag>.csv (per subtype x cell) and
results/mr4_dose<tag>.csv (per subtype x group size; the partitions of target_sample_size.py are the ones reported)."""
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

FSB = "/path/to/research/analysis/subtype_benchmark"
SB = "/path/to/research/analysis"
for p in (SB, FSB, f"{SB}/validation", f"{SB}/attribution_methods"):
    if p not in sys.path:
        sys.path.insert(0, p)
import recast_centroid as fc                                     # noqa: E402  the mean-lognorm centroid
import lineage_registry as C                                    # noqa: E402
from select_recast import EXT_ZHAO, load_lineage              # noqa: E402  the cells the benchmark uses
from scoring_benchmark.datasets import SCIMILARITY_MODEL        # noqa: E402

OUT = f"{FSB}/results"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
K = 20
NSTEPS = 50
SIZES = [1, 2, 5, 10, 25, 50, 100]        # plus "all" (= the whole state), appended per state
BATCH = 128                                # groups per IG call
GT = json.load(open(f"{OUT}/cellmarker_gt_fine.json"))
GO_SET = {l.strip() for l in open(f"{OUT}/go_genes.txt") if l.strip()}


def lognorm_rows(Craw):
    """Per-cell tp10k-lognorm of the go-space counts -- the encoder's input, and the n=1 profile.
    Same transform recast_centroid uses, so the n=N group profile equals fc.centroid exactly."""
    X = Craw.tocsr().astype("float64")
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    M = X.multiply((1e4 / tot)[:, None]).tocsr()
    M.data = np.log1p(M.data)
    return np.asarray(M.todense(), dtype="float32")


def unique_gt(states, genes):
    """The state-unique CellMarker sets, built exactly as fs_uniq / compute_prf do: intersect the GO
    universe, then drop any gene claimed by two or more siblings in this lineage."""
    pooled = {c: (set(GT.get(c, [])) & GO_SET) for c in states}
    shared = {g for g, n in Counter(g for c in states for g in pooled[c]).items() if n >= 2}
    return {c: (pooled[c] - shared) for c in states}


def recall(order, genes, mk):
    return len({genes[j] for j in order[:K]} & mk) / len(mk) if mk else np.nan


def main():
    import contrastive_ig
    from captum.attr import IntegratedGradients
    mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    dev = contrastive_ig.dev

    def group_attr(Cg, Cr, U):
        """One IG pass per group. Cg [M,G] group profiles, Cr [G] shared reference, U [M,G] directions.
        Returns the gated, positive, sum-1000-normalized attribution per group."""
        out = np.zeros_like(Cg)
        for s in range(0, len(Cg), BATCH):
            cg, u = Cg[s:s + BATCH], U[s:s + BATCH]
            inp = torch.tensor(cg, device=dev).requires_grad_(True)
            base = torch.tensor(np.tile(Cr, (len(cg), 1)), device=dev)
            a = ig.attribute(inp, baselines=base, target=0,
                             additional_forward_args=(torch.tensor(u, device=dev),),
                             n_steps=NSTEPS, internal_batch_size=BATCH * 8).detach().cpu().numpy()
            a = np.where(cg - Cr[None] > 0, np.clip(a, 0, None), 0.0)   # the RECAST dC>0 gate, per group
            ss = a.sum(1, keepdims=True); ss[ss == 0] = 1.0
            out[s:s + BATCH] = (a / ss * 1000.0)
        return out

    fac, dose = [], []
    lineages = sorted(C.NEW_IDS + list(EXT_ZHAO))
    print(f"{len(lineages)} lineages | K={K} | sizes {SIZES}+all", flush=True)
    for lin in lineages:
        Craw, lab, states, genes = load_lineage(lin)
        genes = list(genes)
        Xln = lognorm_rows(Craw)
        Z = contrastive_ig.embed(Xln)
        uniq = unique_gt(states, genes)
        for c in states:
            mk = uniq[c]
            if not mk:
                continue
            tgt = np.where(lab == c)[0]; ref = np.where(lab != c)[0]
            if len(tgt) < 2 or len(ref) < 2:
                continue
            C_R = fc.centroid(Craw, ref)
            zr = Z[ref].mean(0)

            # ---- a21: population target, NO encoder. Rank by dC = C_T - C_R (== mean-logFC here). ----
            C_T = fc.centroid(Craw, tgt)
            dC = C_T - C_R
            r_a21 = recall(np.argsort(-np.where(dC > 0, dC, -np.inf)), genes, mk)

            # ---- the dose curve; its endpoints ARE the two encoder cells of the factorial ----
            # hash() of a str tuple is salted per process (PYTHONHASHSEED), so this partition is not reproducible
            # for n = 2..100; the reported partitions are those of target_sample_size.py (stable crc32 seeds,
            # 20 partitions).  n = 1 and n = N do not depend on the partition.
            rng = np.random.default_rng(abs(hash((lin, c))) % (2**32))
            perm = rng.permutation(tgt)
            sizes = [n for n in SIZES if n < len(tgt)] + [len(tgt)]
            per_n = {}
            for n in sizes:
                grp = [perm[i:i + n] for i in range(0, len(perm), n)]
                grp = [g for g in grp if len(g) >= max(1, n // 2)]     # drop a ragged tail group
                Cg = np.vstack([fc.centroid(Craw, g) for g in grp])
                U = np.vstack([Z[g].mean(0) - zr for g in grp])
                U = (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)).astype("float32")
                A = group_attr(Cg.astype("float32"), C_R.astype("float32"), U)
                r = recall(np.argsort(-A.mean(0)), genes, mk)
                per_n[n] = r
                dose.append(dict(lineage=lin, state=c, n_cells=n, n_groups=len(grp),
                                 n_target=len(tgt), recall=r, is_percell=int(n == 1),
                                 is_full=int(n == len(tgt))))
            fac.append(dict(lineage=lin, state=c, n_gt=len(mk), n_target=len(tgt),
                            a21_pop_noenc=r_a21,
                            a12_percell_enc=per_n[1],
                            a22_pop_enc=per_n[len(tgt)]))
        print(f"  [{lin}] {sum(1 for c in states if uniq[c])} scorable states", flush=True)

    F = pd.DataFrame(fac); F["unit"] = F.lineage + "|" + F.state
    D = pd.DataFrame(dose); D["unit"] = D.lineage + "|" + D.state
    F.to_csv(f"{OUT}/mr4_factorial{VTAG}.csv", index=False)
    D.to_csv(f"{OUT}/mr4_dose{VTAG}.csv", index=False)

    print(f"\n{len(F)} states | factorial cells computed here (means over states):")
    print(f"  a21  population + no encoder (mean-logFC)      {F.a21_pop_noenc.mean():.4f}")
    print(f"  a12  per-cell   + encoder   (contrastive)      {F.a12_percell_enc.mean():.4f}")
    print(f"  a22  population + encoder   (== RECAST)         {F.a22_pop_enc.mean():.4f}")
    print("\ndose curve (mean recall over states, by group size):")
    print(D.groupby("n_cells").recall.agg(["mean", "count"]).round(4).to_string())
    print(f"\n[wrote] {OUT}/mr4_factorial{VTAG}.csv, mr4_dose{VTAG}.csv", flush=True)


if __name__ == "__main__":
    main()
