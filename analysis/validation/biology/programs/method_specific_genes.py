"""Method-specific gene sets scored on the 18-program expectation matrix of program_enrichment.py
(Supplementary Fig. 6).

Of the genes each method finds and the other does not, which recover expected programs (the question Squair et al.,
Nat Commun 2021, ask of two differential-expression methods)?  The two specific sets have the same size in every state
(enrichment_full_matrix_v3sym.csv: RECAST top30 \\ COSG top30 and COSG top30 \\ RECAST top30), so recovery is
comparable without a size correction.

Writes:
  method_specific_metrics.csv   the four gene sets x three metrics, with the compartment-stratified subtype-cluster
                                 bootstrap of program_enrichment.py (same number of draws and seed)
  method_specific_pair_categories.csv    the 99 expected pairs split RECAST-only / both / COSG-only / neither by the full top-30
                                 panels, and how many of each the RECAST-specific genes recover on their own
  method_specific_hits.csv       every expected pair the RECAST-specific genes recover on their own, with the
                                 overlapping curated-set genes and their RECAST rank"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/research/analysis/validation/biology/programs")
import enrichment_cohort as cohort

BV = "/path/to/research/analysis/validation/biology"
OUT = f"{BV}/programs"
PANELS = "/path/to/research/analysis/subtype_benchmark/panels"
ALPHA, B, SEED = 0.05, 2000, 0        # program_enrichment's constants: the cluster indices below are drawn
                                      # from default_rng(SEED + 1) in the same order, so the RECAST-vs-COSG
                                      # rows reproduce its intervals (checked below)

# ---------------------------------------------------------------- cohort, mask, hit tensors
coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment)).reset_index(drop=True)
coh["sid"] = np.arange(len(coh))
CIDX = {(c, s): i for i, (c, s, lab) in enumerate(cohort.COLS)}
coh["cid"] = [CIDX[(r.comp, r.state)] for r in coh.itertuples()]
NS = len(coh)
assert NS == 74, NS

R = pd.read_csv(f"{BV}/enrichment_full_matrix_v3sym.csv").merge(
    coh[["lineage", "state", "sid", "cid"]], on=["lineage", "state"])
prog = pd.read_csv(f"{OUT}/programs_frozen.csv")
EM = pd.read_csv(f"{OUT}/expectation_matrix_frozen.csv", index_col=0)
CID = coh.cid.values
CLS = np.array([[EM.iloc[i, CID[s]] for s in range(NS)] for i in range(len(prog))])

ARMS = {"RECAST": "RECAST", "RECAST-specific": "RECAST_unique",
        "COSG": "DE_cosg", "COSG-specific": "DE_cosg_unique"}
H, ROWS = {}, {}
for nm, gs in ARMS.items():
    A = np.zeros((len(prog), NS), bool)
    keep = []
    for i, p in prog.iterrows():
        d = R[(R.geneset == gs) & (R.collection == p.collection) & (R.term == p.term) & (R.fdr < ALPHA)]
        for r in d.itertuples():
            A[i, r.sid] = True
            keep.append((i, r.sid, r.overlap_genes, r.k_overlap, r.fdr))
    H[nm], ROWS[nm] = A, keep

# ---------------------------------------------------------------- metrics + the paper's primary bootstrap
def per_state(A):
    o = {}
    for k in "EU":
        M = (CLS == k)
        o[k + "h"] = (A & M).sum(0).astype(float)
        o[k + "t"] = M.sum(0).astype(float)
    return o


def agg(ps, idx):
    E, Et, U, Ut = (ps[k][idx].sum() for k in ["Eh", "Et", "Uh", "Ut"])
    return np.array([100 * E / Et, 100 * E / (E + U) if (E + U) else np.nan, 100 * U / Ut])


PS = {nm: per_state(A) for nm, A in H.items()}
SUB_STATES = [np.where(CID == cid)[0] for cid in range(len(cohort.COLS))]
COMPS = list(dict.fromkeys(c for c, s, lab in cohort.COLS))
COMP_SUBS = {c: [cid for cid, (cc, s, lab) in enumerate(cohort.COLS) if cc == c] for c in COMPS}

rng = np.random.default_rng(SEED + 1)          # program_enrichment's rng2, the cluster-bootstrap stream
IDX = []
for _ in range(B):
    pick = []
    for c in COMPS:
        ids = COMP_SUBS[c]
        pick += list(rng.choice(ids, size=len(ids), replace=True))
    IDX.append(np.concatenate([SUB_STATES[c] for c in pick]))

METRICS = ["recovery", "precision", "unexpected"]
pt = {nm: agg(PS[nm], np.arange(NS)) for nm in H}
draws = {nm: np.array([agg(PS[nm], ix) for ix in IDX]) for nm in H}


def ci(v):
    v = v[~np.isnan(v)]
    return np.percentile(v, [2.5, 97.5])


rows = []
for a, b_ in [("RECAST", "COSG"), ("RECAST-specific", "COSG-specific")]:
    dif = draws[a] - draws[b_]
    for j, m in enumerate(METRICS):
        lo1, hi1 = ci(draws[a][:, j])
        lo2, hi2 = ci(draws[b_][:, j])
        lo3, hi3 = ci(dif[:, j])
        rows.append(dict(comparison=f"{a} vs {b_}", metric=m,
                         recast=pt[a][j], r_lo=lo1, r_hi=hi1,
                         cosg=pt[b_][j], c_lo=lo2, c_hi=hi2,
                         diff=pt[a][j] - pt[b_][j], d_lo=lo3, d_hi=hi3,
                         recast_hits=int((H[a] & (CLS == ("E" if m != "unexpected" else "U"))).sum()),
                         cosg_hits=int((H[b_] & (CLS == ("E" if m != "unexpected" else "U"))).sum()),
                         n_pairs=int((CLS == ("E" if m != "unexpected" else "U")).sum())))
M1 = pd.DataFrame(rows)

# the full-panel rows reproduce enrichment_metrics.csv of program_enrichment.py (same bootstrap indices)
PUB = pd.read_csv(f"{OUT}/enrichment_metrics.csv")
PUB = PUB[PUB.bootstrap == "primary: compartment-stratified subtype-cluster"].set_index("metric")
for m in METRICS:
    got = M1[(M1.comparison == "RECAST vs COSG") & (M1.metric == m)].iloc[0]
    for col in ["recast", "r_lo", "r_hi", "cosg", "c_lo", "c_hi", "diff", "d_lo", "d_hi"]:
        assert abs(got[col] - PUB.loc[m, col]) < 1e-9, \
            f"differs from enrichment_metrics.csv on {m}.{col}: {got[col]} vs {PUB.loc[m, col]}"
print("[check] the RECAST-vs-COSG rows reproduce enrichment_metrics.csv exactly (9 fields x 3 metrics)")

M1.to_csv(f"{OUT}/method_specific_metrics.csv", index=False)

# ---------------------------------------------------------------- pair categories
E = (CLS == "E")
cat = {"RECAST only": H["RECAST"] & ~H["COSG"] & E,
       "both": H["RECAST"] & H["COSG"] & E,
       "COSG only": ~H["RECAST"] & H["COSG"] & E,
       "neither": ~H["RECAST"] & ~H["COSG"] & E}
assert sum(int(v.sum()) for v in cat.values()) == int(E.sum())
cc = [dict(category=k, n_expected_pairs=int(v.sum()),
           n_also_by_recast_specific=int((v & H["RECAST-specific"]).sum()),
           n_also_by_cosg_specific=int((v & H["COSG-specific"]).sum())) for k, v in cat.items()]
C2 = pd.DataFrame(cc)
C2.to_csv(f"{OUT}/method_specific_pair_categories.csv", index=False)

# ---------------------------------------------------------------- panel d: the genes themselves
rank = {}
for lin in sorted(coh.lineage.unique()):
    try:
        d = pd.read_csv(f"{PANELS}/topN200_{lin}_recast_M0.csv")
    except FileNotFoundError:
        continue
    for (st,), g in d.groupby(["cluster"]):
        rank[(lin, st)] = dict(zip(g.gene, g["rank"]))

hits = []
for i, sid, genes, k, fdr in ROWS["RECAST-specific"]:
    if CLS[i, sid] != "E":
        continue
    r = coh.iloc[sid]
    for gene in str(genes).split(";"):
        hits.append(dict(program=prog.display[i], collection=prog.collection[i], term=prog.term[i],
                         comp=r.comp, subtype=r.state, lineage=r.lineage, gene=gene,
                         recast_rank=rank.get((r.lineage, r.state), {}).get(gene, np.nan), fdr=fdr))
Hd = pd.DataFrame(hits)
miss = Hd.recast_rank.isna().sum()
assert miss == 0, f"{miss} of {len(Hd)} genes have no RECAST rank; the panel must not print a blank rank"
# one row per program x subtype x gene: a gene recovered in two datasets keeps its best (lowest) rank
# and the dataset count, so the panel can show replication without listing the same gene twice
D = (Hd.groupby(["program", "collection", "comp", "subtype", "gene"], as_index=False)
       .agg(recast_rank=("recast_rank", "min"), n_datasets=("lineage", "nunique"),
            lineages=("lineage", lambda s: ";".join(sorted(set(s))))))
D = D.sort_values(["comp", "subtype", "program", "recast_rank"])
D.to_csv(f"{OUT}/method_specific_hits.csv", index=False)

print(M1.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
print()
print(C2.to_string(index=False))
print(f"\npanel d: {D.groupby(['program', 'comp', 'subtype']).ngroups} program x subtype blocks, "
      f"{len(D)} gene rows, {D.gene.nunique()} distinct genes, "
      f"RECAST rank {int(D.recast_rank.min())}-{int(D.recast_rank.max())}")
print(f"[wrote] method_specific_metrics.csv, method_specific_pair_categories.csv, method_specific_hits.csv")
