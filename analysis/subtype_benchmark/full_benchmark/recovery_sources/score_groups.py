"""Fig. 2f, g tables: marker recovery of the four pipelines and of the group-size curve, from the rankings that
group_rankings.py saved.

Score (as scoring/marker_recall_all.py with UNIT_PANELS): markers = the unit's marker panel & GO,
recall = |top 20 & markers| / |markers|.
  mr4_factorial_table_v3.csv   unit, a11 (per-cell tie-corrected Wilcoxon, no encoder; read from the benchmark table),
                               a12 (per-cell targets with the encoder, n = 1), a21 (population target without the
                               encoder, dC), a22 (population target with the encoder, n = N = RECAST), n_target, n_gt
  mr4_dose_partitions_v3.csv   one row per (unit, partition, n), as target_sample_size.py, with recall_n1 and
                               recall_full on every row
Check before writing: the n = N ranking is RECAST, i.e. its top 20 equals the benchmark's panel
(scoring/in/panels/topN_<lin>_recast_M0.csv) for all 97 units, so a22 equals the benchmark's RECAST recall.
Env analysis.  Usage: python -I score_groups.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
R = f"{FSB}/results"
OUT = f"{HERE}/out/tables"
K = 20
A11 = "scanpy_wilcoxontiecorrect_rankby_raw"                     # per-cell test without encoder
UNIT_GT = json.load(open(f"{FULL_BENCHMARK}/ground_truth/panels/marker_panels.json"))
GO = {l.strip() for l in open(f"{R}/go_genes.txt") if l.strip()}
MK = {u: set(g) & GO for u, g in UNIT_GT.items()}


def v4_recall(unit, top20):
    return len(set(top20.split(";")[:K]) & MK[unit]) / len(MK[unit])


def main():
    O = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{HERE}/out/orders/*.csv.gz"))], ignore_index=True)
    assert set(O.unit) == set(UNIT_GT), sorted(set(UNIT_GT) ^ set(O.unit))[:5]
    O["recall"] = [v4_recall(u, t) for u, t in zip(O.unit, O.top20)]

    # ---- check: n = N is the benchmark's RECAST panel
    full = O[O.kind == "full"].set_index("unit")
    bad = []
    for lin in sorted({u.split("|", 1)[0] for u in UNIT_GT}):
        p = pd.read_csv(f"{FULL_BENCHMARK}/scoring/in/panels/topN_{lin}_recast_M0.csv")
        for c, g in p.groupby("cluster"):
            u = f"{lin}|{c}"
            if u in full.index and full.loc[u, "top20"].split(";") != g.sort_values("rank").gene.astype(str).tolist()[:K]:
                bad.append(u)
    print(f"check: n = N top 20 == benchmark RECAST top 20 in {len(full) - len(bad)}/{len(full)} units", bad[:5])
    assert not bad

    # ---- tables
    bench = pd.read_csv(f"{FULL_BENCHMARK}/scoring/out/recall_by_subtype.csv")
    a11 = bench[bench.method == A11].set_index("unit").recall
    m0 = bench[bench.method == "recast_M0"].set_index("unit").recall
    g = lambda kind: O[O.kind == kind].set_index("unit")
    T = pd.DataFrame({"a11_percell_noenc": a11.reindex(sorted(UNIT_GT)),
                      "a12_percell_enc": g("n1").recall, "a21_pop_noenc": g("a21").recall,
                      "a22_pop_enc": g("full").recall})
    T["n_target"] = g("full").n_target
    T["n_gt"] = pd.Series({u: len(MK[u]) for u in UNIT_GT})
    T.index.name = "unit"
    assert T.notna().all().all() and len(T) == 97
    assert np.allclose(T.a22_pop_enc, m0.reindex(T.index)), "a22 != benchmark RECAST recall"
    os.makedirs(OUT, exist_ok=True)
    T.to_csv(f"{OUT}/mr4_factorial_table_v3.csv")

    P = O[O.kind == "part"].copy()
    P["lineage"], P["state"] = P.unit.str.split("|", n=1).str[0], P.unit.str.split("|", n=1).str[1]
    P["recall_n1"] = P.unit.map(g("n1").recall)
    P["recall_full"] = P.unit.map(g("full").recall)
    cols = ["lineage", "state", "unit", "partition", "seed", "n_cells", "n_groups", "n_target", "recall",
            "recall_n1", "recall_full"]
    P[cols].to_csv(f"{OUT}/mr4_dose_partitions_v3.csv", index=False)
    O.drop(columns="top20").to_csv(f"{OUT}/mr4_all_rankings_recall.csv", index=False)

    m, se = T.iloc[:, :4].mean(), T.iloc[:, :4].sem()
    print(f"\n2x2 (mean +- s.e.m. over {len(T)} units)")
    for k in T.columns[:4]:
        print(f"  {k:20s} {m[k]:.4f} +- {se[k]:.4f}")
    pm = P.groupby(["unit", "n_cells"]).recall.mean().unstack("n_cells")
    pm.insert(0, 1, g("n1").recall.reindex(pm.index))
    print("group size curve (mean over units of the partition mean):")
    print(pd.DataFrame({"mean": pm.mean(0), "count": pm.count(0)}).round(4).to_string())
    print(f"  all (n = N): {g('full').recall.mean():.4f}")


if __name__ == "__main__":
    main()
