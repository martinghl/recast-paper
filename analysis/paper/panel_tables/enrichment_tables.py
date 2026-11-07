#!/usr/bin/env python
"""Program-enrichment comparison tables (Fig. 5), read from the outputs of validation/biology/programs; nothing is
re-estimated, and the script stops when two tables disagree.

  enrichment_comparison.csv      six rows = {all top-30 genes, method-specific genes} x {recovery, precision, unexpected}:
                                 the counts (enrichment_counts.csv), the two percentages and the RECAST - COSG difference
                                 with its 95% interval (enrichment_metrics.csv primary bootstrap;
                                 method_specific_metrics.csv)
  expected_pair_categories.csv   the 99 expected program x population pairs, one row each: found by the RECAST / COSG
                                 top 30, and by the RECAST- / COSG-specific genes alone (FDR < 0.05, as in
                                 method_specific_genes.py); its category counts equal method_specific_pair_categories.csv
  example_program_genes.csv      four example programs (E2F targets / NK proliferating, PAX5 / MZB, leukocyte
                                 cytotoxicity / CD4 Tem/Temra in Ren and Yoshida, cilium movement / ciliated) with every
                                 RECAST-specific gene behind each enrichment (method_specific_hits.csv) across the target
                                 and its sibling subtypes: mean log(1 + CP10K) and fraction of cells expressing
                                 (example_gene_expression.csv)

  python enrichment_tables.py      (analysis env)  ->  ../enrichment_tables/data/"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/path/to/research/analysis/validation/biology/programs")
import enrichment_cohort as cohort

BV = "/path/to/research/analysis/validation/biology"
D2 = f"{BV}/programs"
OUT = os.path.join(HERE, "..", "enrichment_tables", "data")
os.makedirs(OUT, exist_ok=True)
ALPHA = 0.05                                                               # method_specific_genes.py's threshold

# ------------------------------------------------------------------------------------------------ comparison table
PB = pd.read_csv(f"{D2}/enrichment_metrics.csv")
PB = PB[PB.bootstrap.str.startswith("primary")].set_index("metric")
SP = pd.read_csv(f"{D2}/method_specific_metrics.csv")
CN = pd.read_csv(f"{D2}/enrichment_counts.csv")
METRICS = ["recovery", "precision", "unexpected"]
rows = []
for grp, comp, est in [("all", "RECAST vs COSG", PB),
                       ("specific", "RECAST-specific vs COSG-specific",
                        SP[SP.comparison == "RECAST-specific vs COSG-specific"].set_index("metric"))]:
    cn = CN[CN.comparison == comp].set_index("metric")
    for m in METRICS:
        r = cn.loc[m]
        for col, nn, dd in (("recast", "recast_n", "recast_d"), ("cosg", "cosg_n", "cosg_d")):
            assert abs(100.0 * r[nn] / r[dd] - est.loc[m, col]) < 1e-9, f"{grp}.{m}.{col}: counts disagree"
        assert abs(est.loc[m, "recast"] - est.loc[m, "cosg"] - est.loc[m, "diff"]) < 1e-9
        rows.append(dict(group=grp, metric=m,
                         recast_n=int(r.recast_n), recast_d=int(r.recast_d), recast_pct=est.loc[m, "recast"],
                         cosg_n=int(r.cosg_n), cosg_d=int(r.cosg_d), cosg_pct=est.loc[m, "cosg"],
                         diff=est.loc[m, "diff"], d_lo=est.loc[m, "d_lo"], d_hi=est.loc[m, "d_hi"]))
TB = pd.DataFrame(rows)
FU = SP[SP.comparison == "RECAST vs COSG"].set_index("metric")             # the two tables share one bootstrap
for m in METRICS:
    for col in ["recast", "cosg", "diff", "d_lo", "d_hi"]:
        assert abs(FU.loc[m, col] - PB.loc[m, col]) < 1e-9, f"tables differ on {m}.{col}"
TB.to_csv(f"{OUT}/enrichment_comparison.csv", index=False)

# ------------------------------------------------------------------------------- the 99 expected pairs, one row each
coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment)).reset_index(drop=True)
coh["sid"] = np.arange(len(coh))
CIDX = {(c, s): i for i, (c, s, lab) in enumerate(cohort.COLS)}
coh["cid"] = [CIDX[(r.comp, r.state)] for r in coh.itertuples()]
NS = len(coh)
assert NS == 74, NS
R = pd.read_csv(f"{BV}/enrichment_full_matrix_v3sym.csv").merge(coh[["lineage", "state", "sid"]],
                                                                  on=["lineage", "state"])
prog = pd.read_csv(f"{D2}/programs_frozen.csv")
EM = pd.read_csv(f"{D2}/expectation_matrix_frozen.csv", index_col=0)
CID = coh.cid.values
CLS = np.array([[EM.iloc[i, CID[s]] for s in range(NS)] for i in range(len(prog))])
ARMS = {"recast": "RECAST", "cosg": "DE_cosg", "recast_specific": "RECAST_unique", "cosg_specific": "DE_cosg_unique"}
H = {}
for nm, gs in ARMS.items():
    A = np.zeros((len(prog), NS), bool)
    for i, p in prog.iterrows():
        d = R[(R.geneset == gs) & (R.collection == p.collection) & (R.term == p.term) & (R.fdr < ALPHA)]
        A[i, d.sid.values] = True
    H[nm] = A
E = CLS == "E"
ii, ss = np.where(E)
PR = pd.DataFrame(dict(program=prog.display.values[ii], collection=prog.collection.values[ii],
                       compartment=coh.comp.values[ss], lineage=coh.lineage.values[ss], subtype=coh.state.values[ss],
                       **{nm: H[nm][ii, ss] for nm in ARMS}))
assert len(PR) == 99, len(PR)
cat = np.select([PR.recast & ~PR.cosg, PR.recast & PR.cosg, ~PR.recast & PR.cosg],
                ["RECAST only", "both", "COSG only"], "neither")
PR["category"] = cat
CAT = pd.read_csv(f"{D2}/method_specific_pair_categories.csv").set_index("category")
for c, r in CAT.iterrows():
    sub = PR[PR.category == c]
    got = (len(sub), int(sub.recast_specific.sum()), int(sub.cosg_specific.sum()))
    want = (r.n_expected_pairs, r.n_also_by_recast_specific, r.n_also_by_cosg_specific)
    assert got == want, f"pair category {c}: {got} != method_specific_pair_categories.csv {want}"
PR.to_csv(f"{OUT}/expected_pair_categories.csv", index=False)
both_spec = PR[PR.recast_specific & PR.cosg_specific]
print("pairs found by both specific sets:", len(both_spec), "->",
      both_spec.groupby("category").size().to_dict())

# ------------------------------------------------------------------- every RECAST-specific gene of four example programs
DS_NAME = {"fs_nk_pbmcbmn": "pbmcbmn", "fs_asc_zhao": "Zhao", "tf_cd4_ren": "Ren", "tf_cd4_yosh": "Yoshida",
           "fs_lung_hlca": "HLCA"}
BLOCKS = [("E2F targets", "NK Proliferating"), ("PAX5", "MZB"),
          ("leukocyte cytotoxicity", "Tem/Temra helper T cells"), ("cilium movement", "ciliated")]
HIT = pd.read_csv(f"{D2}/method_specific_hits.csv")
EX = pd.read_csv(f"{D2}/example_gene_expression.csv")
MG = pd.read_csv(f"{D2}/example_gene_margins.csv")
LAB = {(c, s): lab for c, s, lab in cohort.COLS}
ORDER = {(c, s): i for i, (c, s, lab) in enumerate(cohort.COLS)}
out = []
for bi, (pg, tgt) in enumerate(BLOCKS):
    h = HIT[(HIT.program == pg) & (HIT.subtype == tgt)].sort_values(["recast_rank", "gene"])
    assert len(h) and h.gene.is_unique, (pg, tgt, len(h))
    lins = sorted(set(";".join(h.lineages).split(";")))
    for _, g in h.iterrows():                                  # every gene is a hit in every dataset of the block
        assert sorted(g.lineages.split(";")) == lins, (pg, g.gene, g.lineages, lins)
    for lin in lins:
        comp = cohort.lineage_compartment(lin)
        for _, g in h.iterrows():
            mg = MG[(MG.gene == g.gene) & (MG.state == tgt) & (MG.lineage == lin) & (MG.program == pg)]
            assert len(mg) == 1 and mg.margin_mean.iloc[0] > 0, (pg, lin, g.gene)   # target above every sibling
            ex = EX[(EX.lineage == lin) & (EX.gene == g.gene)]
            states = sorted(ex.state, key=lambda s: ORDER[(comp, s)])
            assert tgt in states and len(states) == ex.state.nunique() >= 2, (lin, g.gene, states)
            for s in states:
                e = ex[ex.state == s].iloc[0]
                out.append(dict(block=bi + 1, program=pg, collection=g.collection, target=tgt, dataset=DS_NAME[lin],
                                lineage=lin, gene=g.gene, recast_rank=int(g.recast_rank), subtype=s,
                                label=LAB[(comp, s)], col_order=ORDER[(comp, s)], is_target=s == tgt,
                                n_cells=int(e.n_cells), mean_lognorm=e.mean_lognorm,
                                frac_expressing=e.frac_expressing))
DD = pd.DataFrame(out)
assert DD.groupby("block").gene.nunique().tolist() == [8, 2, 4, 5], DD.groupby("block").gene.nunique()
assert DD.drop_duplicates(["lineage", "gene"]).shape[0] == 8 + 2 + 4 * 2 + 5
DD.to_csv(f"{OUT}/example_program_genes.csv", index=False)
print(f"[wrote] {OUT}: enrichment_comparison.csv ({len(TB)} rows), expected_pair_categories.csv ({len(PR)}), "
      f"example_program_genes.csv ({len(DD)}; {DD.gene.nunique()} genes, max mean {DD.mean_lognorm.max():.2f})")
