#!/usr/bin/env python
"""Method-specific program genes (Fig. 5c, Supplementary Fig. 6b), read from the tables of
validation/biology/programs and enrichment_tables.py; nothing is re-estimated.

  program_genes_method_specific.csv  the method-specific rows of the enrichment comparison (expected-program recovery,
                                     unexpected enrichment): percentage for each method with its compartment-stratified
                                     subtype-cluster bootstrap 95% interval, the RECAST - COSG difference with its
                                     interval and the subtype-cluster sign-flip permutation P (method_specific_metrics.csv,
                                     enrichment_comparison_p.csv), checked against enrichment_comparison.csv; the full
                                     top-30 rows are included (set = "full") for the text
  program_genes_by_comparison.csv    expected pairs per compartment and how many each method-specific set recovers
  program_genes_examples.csv         the six expected pairs enriched only in the RECAST top 30 and also recovered by the
                                     RECAST-specific genes alone, plus the WNT / colonic stem cell pair.  One row per gene
                                     in either top 30: rank in each list (top 50 ranked; NA beyond), shared / RECAST only /
                                     COSG only, and whether it is in the program's gene set (same lists, background and
                                     term filter as program_gene_capture.py); k per method checked against
                                     enrichment_pairs.csv

  python program_gene_tables.py      (analysis env)  ->  ../enrichment_tables/data/program_genes_*.csv"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BV = Path("/path/to/research/analysis/validation/biology")
D2 = BV / "programs"
FSB = Path("/path/to/research/analysis/subtype_benchmark")
GSDIR = FSB / "fs_genesets_v2"
OUT = Path(HERE).parent / "enrichment_tables" / "data"
TOPK = 30

# ------------------------------------------------------------------------------------------------ method-specific rows
SP = pd.read_csv(D2 / "method_specific_metrics.csv")
PB = pd.read_csv(D2 / "enrichment_comparison_p.csv")
TB = pd.read_csv(OUT / "enrichment_comparison.csv")
SET = {"RECAST vs COSG": "full", "RECAST-specific vs COSG-specific": "specific"}
C = SP.assign(set=SP.comparison.map(SET))
assert C.set.notna().all()
C = C[C.metric.isin(["recovery", "unexpected"])].copy()
C["n"] = np.where(C.metric == "recovery", 99, 1164)
assert (C.n_pairs.where(C.metric == "recovery", 1164) == C.n).all()
C["p_perm"] = [float(PB[(PB.group == ("all" if s == "full" else "specific")) & (PB.metric == m)].p_perm.iloc[0])
               for s, m in zip(C.set, C.metric)]
for r in C.itertuples():                                       # same numbers as enrichment_comparison.csv
    t = TB[(TB.group == ("all" if r.set == "full" else "specific")) & (TB.metric == r.metric)].iloc[0]
    assert (int(r.recast_hits) if r.metric == "recovery" else int(t.recast_n)) == int(t.recast_n)
    assert abs(r.recast - t.recast_pct) < 1e-9 and abs(r.cosg - t.cosg_pct) < 1e-9
    assert abs(r.diff - t["diff"]) < 1e-9 and abs(r.d_lo - t.d_lo) < 1e-9 and abs(r.d_hi - t.d_hi) < 1e-9
C["recast_n"] = [int(TB[(TB.group == ("all" if s == "full" else "specific")) & (TB.metric == m)].recast_n.iloc[0])
                 for s, m in zip(C.set, C.metric)]
C["cosg_n"] = [int(TB[(TB.group == ("all" if s == "full" else "specific")) & (TB.metric == m)].cosg_n.iloc[0])
               for s, m in zip(C.set, C.metric)]
C = C[["set", "metric", "n", "recast_n", "recast", "r_lo", "r_hi", "cosg_n", "cosg", "c_lo", "c_hi", "diff", "d_lo",
       "d_hi", "p_perm"]]
C.to_csv(OUT / "program_genes_method_specific.csv", index=False)

P = pd.read_csv(OUT / "enrichment_pairs.csv")
E = P[P.cls == "E"]
COMPS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]
BC = (E.groupby("comp").agg(n=("dk", "size"), recast_specific=("hit_recast_specific", "sum"),
                            cosg_specific=("hit_cosg_specific", "sum"), recast_full=("hit_recast", "sum"),
                            cosg_full=("hit_cosg", "sum")).reindex(COMPS).reset_index())
assert BC.n.sum() == 99
BC.to_csv(OUT / "program_genes_by_comparison.csv", index=False)

# ------------------------------------------------------------------------------------------------ worked examples
PROG = pd.read_csv(D2 / "programs_frozen.csv")
BG = set(x.strip() for x in open(FSB / "results/go_genes.txt") if x.strip())
CSV = {"CollecTRI": "collectri_human.csv", "Hallmark": "hallmark_human.csv", "Reactome": "reactome_human.csv",
       "GO-BP": "gobp_human.csv"}


def load_terms(csv, smin=8, smax=500):                         # enrichment_matrix_both.load_terms
    df = pd.read_csv(csv)
    src, tgt = df.columns[0], df.columns[1]
    d = {}
    for s, g in df.groupby(src):
        genes = set(g[tgt]) & BG
        if smin <= len(genes) <= smax:
            d[s] = genes
    return d


TERMS = {c: load_terms(BV / f) for c, f in CSV.items()}
cand = E[E.hit_recast & ~E.hit_cosg & E.hit_recast_specific]
assert len(cand) == 6
cand = pd.concat([cand, E[(E.program == "WNT signaling") & (E.subtype == "Stem")]])
rows = []
for pid, r in enumerate(cand.itertuples(), 1):
    pr = PROG[PROG.display == r.program].iloc[0]
    PG = TERMS[pr.collection][pr.term]
    foc = pd.read_csv(FSB / f"panels/topN_{r.lineage}_recast_M0.csv")
    de = pd.read_csv(GSDIR / f"{r.lineage}.csv")
    de.columns = [c.strip().strip('"') for c in de.columns]
    for c in ["method", "cluster", "gene"]:
        de[c] = de[c].astype(str).str.strip().str.strip('"')
    de = de[de.method == "cosg"]
    f = foc[foc.cluster.astype(str) == r.state].sort_values("rank")
    d = de[de.cluster == r.state].sort_values("rank")
    rf = {g: i + 1 for i, g in enumerate(f.gene)}             # position in each ranked list (top 50 kept)
    rd = {g: i + 1 for i, g in enumerate(d.gene)}
    F, D = list(f.gene[:TOPK]), list(d.gene[:TOPK])
    assert len(F) == len(D) == TOPK
    kR = len(set(F) & BG & PG); kC = len(set(D) & BG & PG)
    assert (kR, kC) == (r.k_recast, r.k_cosg), (r.program, r.subtype, kR, kC)
    for g in dict.fromkeys(F + D):
        rows.append(dict(pair=pid, program=r.program, collection=pr.collection, comp=r.comp, subtype=r.subtype,
                         lineage=r.lineage, state=r.state, k_recast=r.k_recast, k_cosg=r.k_cosg,
                         hit_recast_specific=r.hit_recast_specific, gene=g,
                         rank_recast=rf.get(g, np.nan), rank_cosg=rd.get(g, np.nan),
                         set=("shared" if g in F and g in D else "recast_only" if g in F else "cosg_only"),
                         in_background=g in BG, program_gene=(g in PG) and (g in BG)))
G = pd.DataFrame(rows)
for pid, g in G.groupby("pair"):
    n = g.set.value_counts()
    assert n.get("recast_only", 0) == n.get("cosg_only", 0)    # equal-sized method-specific sets
    pg = g[g.program_gene]
    dk = (pg.set == "recast_only").sum() - (pg.set == "cosg_only").sum()
    assert dk == g.k_recast.iloc[0] - g.k_cosg.iloc[0]         # shared genes cancel
G.to_csv(OUT / "program_genes_examples.csv", index=False)

print(C.round(2).to_string(index=False))
print(BC.to_string(index=False))
for pid, g in G.groupby("pair"):
    n = g.set.value_counts()
    pg = g[g.program_gene].groupby("set").gene.apply(lambda s: ", ".join(s))
    print(f"[d {pid}] {g.program.iloc[0]} | {g.comp.iloc[0]} {g.subtype.iloc[0]} ({g.lineage.iloc[0]}): "
          f"shared {n.get('shared', 0)}, RECAST only {n.get('recast_only', 0)}; program genes {dict(pg)}")
print(f"[wrote] {OUT}: program_genes_method_specific.csv ({len(C)}), program_genes_by_comparison.csv ({len(BC)}), program_genes_examples.csv ({len(G)})")
