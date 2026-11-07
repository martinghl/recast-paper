#!/usr/bin/env python
"""Program genes captured per program-population pair (Fig. 5b, Supplementary Fig. 6a), read from
validation/biology/programs/program_gene_capture.py and the robustness tables of program_enrichment.py.  Nothing is
re-estimated here.

  enrichment_pairs.csv       every expected (99) and unexpected (1,164) pair: program genes captured by the RECAST and
                             COSG top 30 (k), their difference (dk) and the binary hits
  enrichment_programs.csv    per program (order of Fig. 5a), its expected pairs: mean k for each method, counts dk > / = / < 0
  enrichment_summary.csv     mean dk with the cluster-bootstrap CI and the cluster sign-flip permutation P (expected,
                             unexpected); the permutation P of the enrichment comparison rows
  enrichment_robustness.csv  leave-one-dataset-out (9) and leave-one-program-out (18) recovery differences (lodo / lopo.csv)

  python enrichment_pairs.py      (analysis env)  ->  ../enrichment_tables/data/"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D2 = "/path/to/research/analysis/validation/biology/programs"
OUT = os.path.join(HERE, "..", "enrichment_tables", "data")

P = pd.read_csv(f"{D2}/program_capture_pairs.csv")
S = pd.read_csv(f"{D2}/program_capture_stats.csv")
PB = pd.read_csv(f"{D2}/enrichment_comparison_p.csv")
PROG = pd.read_csv(f"{D2}/programs_frozen.csv")

keep = ["program", "collection", "term", "cls", "lineage", "state", "comp", "subtype", "n_program_genes",
        "k_recast", "k_cosg", "dk", "hit_recast", "hit_cosg", "hit_recast_specific", "hit_cosg_specific"]
PP = P[keep].copy()
PP["program_order"] = PP.program.map({p: i + 1 for i, p in enumerate(PROG.display)})
assert (PP.cls == "E").sum() == 99 and (PP.cls == "U").sum() == 1164
PP.to_csv(f"{OUT}/enrichment_pairs.csv", index=False)

E = PP[PP.cls == "E"]
G = E.groupby("program", sort=False).agg(n_pairs=("dk", "size"), mean_k_recast=("k_recast", "mean"),
                                         mean_k_cosg=("k_cosg", "mean"), n_pos=("dk", lambda x: int((x > 0).sum())),
                                         n_zero=("dk", lambda x: int((x == 0).sum())),
                                         n_neg=("dk", lambda x: int((x < 0).sum()))).reset_index()
G["program_order"] = G.program.map({p: i + 1 for i, p in enumerate(PROG.display)})
G = G.sort_values("program_order")
assert G.n_pairs.sum() == 99 and len(G) == 18                       # every program has >= 1 expected pair
G.to_csv(f"{OUT}/enrichment_programs.csv", index=False)

dk = S[S.column == "dk"].set_index("pairs")
rows = [dict(kind="capture", pairs=k, n_pairs=int(r.n_pairs), mean_k_recast=r.mean_recast, mean_k_cosg=r.mean_cosg,
             diff=r["mean"], lo=r.ci_lo, hi=r.ci_hi, p=r.p_perm, n_pos=int(r.n_pos), n_zero=int(r.n_zero),
             n_neg=int(r.n_neg)) for k, r in dk.iterrows()]
rows += [dict(kind="b_" + r.group, pairs=r.metric, diff=r.diff, lo=r.d_lo, hi=r.d_hi, p=r.p_perm)
         for r in PB.itertuples()]
SU = pd.DataFrame(rows)
SU.to_csv(f"{OUT}/enrichment_summary.csv", index=False)

LO = pd.read_csv(f"{D2}/lodo.csv")
LP = pd.read_csv(f"{D2}/lopo.csv")
RB = pd.concat([LO.assign(kind="dataset", label=LO.study)[["kind", "label", "recovery_diff", "recovery_diff_lo",
                                                           "recovery_diff_hi", "n_states_removed"]],
                LP.assign(kind="program", label=LP.omitted, n_states_removed=np.nan)[
                    ["kind", "label", "recovery_diff", "recovery_diff_lo", "recovery_diff_hi", "n_states_removed"]]])
RB.to_csv(f"{OUT}/enrichment_robustness.csv", index=False)

e = SU[(SU.kind == "capture") & (SU.pairs == "expected")].iloc[0]
u = SU[(SU.kind == "capture") & (SU.pairs == "unexpected")].iloc[0]
print(f"[expected]   {e.n_pairs} pairs: RECAST more {e.n_pos}, equal {e.n_zero}, COSG more {e.n_neg}; "
      f"mean k {e.mean_k_recast:.2f} vs {e.mean_k_cosg:.2f}; diff {e['diff']:+.2f} [{e.lo:.2f}, {e.hi:.2f}], P = {e.p:.3f}")
print(f"[unexpected] {u.n_pairs} pairs: RECAST more {u.n_pos}, equal {u.n_zero}, COSG more {u.n_neg}; "
      f"mean k {u.mean_k_recast:.2f} vs {u.mean_k_cosg:.2f}; diff {u['diff']:+.3f} [{u.lo:.3f}, {u.hi:.3f}], P = {u.p:.2f}")
print(f"[wrote] {OUT}: enrichment_pairs.csv ({len(PP)}), enrichment_programs.csv ({len(G)}), enrichment_summary.csv ({len(SU)}), "
      f"enrichment_robustness.csv ({len(RB)})")
