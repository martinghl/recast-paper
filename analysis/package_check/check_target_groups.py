"""Fig. 2f and 2g: RECAST on groups of target cells, for the 97 benchmark subtypes.

For a subtype with N target cells, group sizes n = 1 (every cell its own group), 2, 5, 10, 25, 50, 100 (n < N) and
n = N (the whole subtype, i.e. RECAST itself). Every group is one target of recast.attribute, with the subtype's
siblings as the reference: the call gives the group's direction u_g, its profile C_g, the reference profile C_R and
the Integrated Gradients attribution from C_R to C_g. Kept as analysis code, exactly as in the original run: the dC > 0 gate,
the positive part normalized to sum 1000 per group, the mean over groups, and the 20 random partitions per size
(seed crc32("<lineage>|<subtype>|<r>"), groups smaller than n/2 dropped). The no-encoder population arm ranks
genes by result.dC.

Compared with the top 20 genes of every ranking that recovery_sources/group_rankings.py saved (out/orders/<lineage>.csv.gz).
    python check_target_groups.py [--partitions 20] LINEAGE [LINEAGE ...]
Writes groups/<lineage>.csv.gz (one row per ranking, with the original top 20 beside the package's)."""
import argparse
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

import common as C
import recast
from check_benchmark_panels import lineages

ORDERS = os.path.join(C.FULL_BENCHMARK, "recovery_sources", "out", "orders")
UNIT_GT = json.load(open(os.path.join(C.FULL_BENCHMARK, "ground_truth", "panels", "marker_panels.json")))
SIZES = [2, 5, 10, 25, 50, 100]
K = 20
CHUNK = 256          # groups per attribute() call


def seed_for(lin, c, r):
    return zlib.crc32(f"{lin}|{c}|{r}".encode("utf-8"))


def group_ranking(enc, a, Z, c, others, groups):
    """Mean over groups of the gated, positive, sum-1000 attribution -> gene order (as in the original run)."""
    lab = a.obs["state"].to_numpy().astype(object)
    names = [f"__group{k}" for k in range(len(groups))]
    for nm, g in zip(names, groups):
        lab[g] = nm
    a.obs["group"] = pd.Categorical(lab.astype(str))
    A = np.zeros((len(groups), a.n_vars), dtype="float32")
    for s in range(0, len(names), CHUNK):
        tg = names[s:s + CHUNK]
        res = recast.attribute(enc, a, "group", target=tg, reference=others, gate="dC",
                               centroid="mean_lognorm", qc="off", embeddings=Z)
        att = res.attribution[tg].to_numpy("float32").T
        dC = res.dC[tg].to_numpy("float32").T
        att = np.where(dC > 0, np.clip(att, 0, None), 0.0)
        ss = att.sum(1, keepdims=True); ss[ss == 0] = 1.0
        A[s:s + len(tg)] = att / ss * 1000.0
    del a.obs["group"]
    return np.argsort(-A.mean(0))


def run(lin, spec, enc, n_part):
    out = os.path.join(C.OUT, "groups", f"{lin}.csv.gz")
    if os.path.exists(out):
        print(f"[{lin}] exists, skipped", flush=True)
        return
    t0 = time.time()
    labels = C.read_labels(os.path.join(spec["p2"], spec["label_file"]))
    a, states = C.lineage_adata(spec["p2"], spec["ds"], labels, spec["states"], spec["min_cells"])
    genes = np.asarray(a.var_names)
    Z = enc.embed(a.X)
    lab = a.obs["state"].to_numpy()
    stored = pd.read_csv(os.path.join(ORDERS, f"{lin}.csv.gz"))
    units = [c for c in states if f"{lin}|{c}" in UNIT_GT]
    rows = []
    for c in units:
        unit = f"{lin}|{c}"
        tgt = np.where(lab == c)[0]
        others = [s for s in states if s != c]
        P = stored[stored.unit == unit].set_index(["kind", "partition", "n_cells"])

        def keep(kind, order, partition=-1, n_cells=0, n_groups=0):
            mine = ";".join(genes[j] for j in order[:K])
            theirs = P.top20.get((kind, partition, n_cells))
            rows.append(dict(unit=unit, kind=kind, partition=partition, n_cells=n_cells, n_groups=n_groups,
                             top20=mine, top20_original=theirs, identical=mine == theirs,
                             same_set=set(mine.split(";")) == set(str(theirs).split(";"))))

        res = recast.attribute(enc, a, "state", target=c, reference="rest", gate="dC",
                               centroid="mean_lognorm", qc="off", embeddings=Z)
        dC = res.dC[c].to_numpy()
        keep("a21", np.argsort(-np.where(dC > 0, dC, -np.inf)), n_cells=len(tgt), n_groups=1)
        keep("n1", group_ranking(enc, a, Z, c, others, [np.array([i]) for i in tgt]), n_cells=1, n_groups=len(tgt))
        keep("full", group_ranking(enc, a, Z, c, others, [tgt]), n_cells=len(tgt), n_groups=1)
        for r in range(n_part):
            perm = np.random.default_rng(seed_for(lin, c, r)).permutation(tgt)
            for n in [n for n in SIZES if n < len(tgt)]:
                grp = [perm[i:i + n] for i in range(0, len(perm), n)]
                grp = [g for g in grp if len(g) >= max(1, n // 2)]
                keep("part", group_ranking(enc, a, Z, c, others, grp), partition=r, n_cells=n, n_groups=len(grp))
        done = [x for x in rows if x["unit"] == unit]
        print(f"  {unit}: N={len(tgt)} identical {sum(x['identical'] for x in done)}/{len(done)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        C.assert_package_only()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip")
    os.replace(out + ".tmp", out)
    print(f"[{lin}] {len(units)} units, {len(rows)} rankings, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, default=20)
    ap.add_argument("lineages", nargs="+")
    A = ap.parse_args()
    todo = lineages()
    enc = C.scimilarity()
    for lin in A.lineages:
        run(lin, todo[lin], enc, A.partitions)
