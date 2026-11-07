"""Program genes captured by the RECAST and COSG top 30 (Fig. 5b), with tests.

A. For each of the 99 expected and 1,164 unexpected program-population pairs of the expectation matrix
   (program_enrichment.py), k = number of the program's genes in the RECAST / COSG top 30 (same gene lists, background
   and term filter as enrichment_matrix_both.py) and dk = k_RECAST - k_COSG (shared genes cancel, so dk is also the
   difference of the method-specific genes).  Mean dk with the compartment-stratified subtype-cluster bootstrap
   interval; two-sided subtype-cluster sign-flip permutation P.  Secondary: the same for d(-log10 P), hypergeometric
   P of the same computation.
B. P values for the six rows of the enrichment comparison with the same sign-flip permutation (method labels swapped
   together within each of the 39 subtypes).

Writes:
  program_capture_pairs.csv   one row per expected / unexpected pair: k, P and dk for RECAST, COSG and both
                           method-specific sets
  program_capture_stats.csv   A: mean dk (and d -log10 P) with interval and P, counts of dk > 0 / = 0 / < 0
  enrichment_comparison_p.csv        B: the six rows with their permutation P
Run: /path/to/envs/analysis/bin/python program_gene_capture.py"""
import glob, os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, "/path/to/research/analysis/validation/biology/programs")
import enrichment_cohort as cohort

BV = Path("/path/to/research/analysis/validation/biology")
OUT = BV / "programs"
FSB = Path("/path/to/research/analysis/subtype_benchmark")
GSDIR = FSB / "fs_genesets_v2"                                   # GENESETS_DIR for the _v3 results
TOPK, ALPHA, B, SEED, NPERM = 30, 0.05, 2000, 0, 100_000
COMPS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]

# ------------------------------------------------------------------ programs, expectation matrix and cohort of program_enrichment.py
PROG = pd.read_csv(OUT / "programs_frozen.csv")
EM = pd.read_csv(OUT / "expectation_matrix_frozen.csv", index_col=0)
assert len(PROG) == 18 and list(EM.index) == list(PROG.display)
coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment)).reset_index(drop=True)
LAB = {(c, s): lab for c, s, lab in cohort.COLS}
SUBS = [(c, lab) for c, _, lab in cohort.COLS]
coh["sub"] = [LAB[(c, s)] for c, s in zip(coh.comp, coh.state)]
coh["sid"] = np.arange(len(coh))
coh["cid"] = [SUBS.index((c, s)) for c, s in zip(coh.comp, coh["sub"])]
assert len(coh) == 74 and coh.cid.nunique() == 39
assert list(EM.columns) == [f"{c}:{s}" for c, s in SUBS]
CID = coh.cid.values
CLS = np.array([[EM.iloc[i, CID[s]] for s in range(74)] for i in range(18)])          # 18 x 74, E / P / U
assert (CLS == "E").sum() == 99 and (CLS == "U").sum() == 1164

# ------------------------------------------------------------------ gene lists, background, program genes
BG = set(x.strip() for x in open(FSB / "results/go_genes.txt") if x.strip())
M = len(BG)
CSV = {"CollecTRI": "collectri_human.csv", "Hallmark": "hallmark_human.csv", "Reactome": "reactome_human.csv",
       "GO-BP": "gobp_human.csv"}


def load_terms(csv, smin=8, smax=500):                           # enrichment_matrix_both.load_terms
    df = pd.read_csv(csv)
    src, tgt = df.columns[0], df.columns[1]
    d = {}
    for s, g in df.groupby(src):
        genes = set(g[tgt]) & BG
        if smin <= len(genes) <= smax:
            d[s] = genes
    return d


TERMS = {c: load_terms(BV / f) for c, f in CSV.items()}
PG = [TERMS[r.collection][r.term] for r in PROG.itertuples()]    # program genes (in background), 18 sets

LISTS = {}                                                       # (lineage, state) -> (F, D, Fu, Du)
for pf in sorted(glob.glob(str(FSB / "panels/topN_*_recast_M0.csv"))):
    tag = re.search(r"topN_(.+)_recast_M0", os.path.basename(pf)).group(1)
    gf = GSDIR / f"{tag}.csv"
    if not gf.exists():
        continue
    foc = pd.read_csv(pf)
    de = pd.read_csv(gf)
    de.columns = [c.strip().strip('"') for c in de.columns]
    for c in ["method", "cluster", "gene"]:
        de[c] = de[c].astype(str).str.strip().str.strip('"')
    de = de[de.method == "cosg"]
    for state, g in foc.groupby("cluster"):
        F = list(g.sort_values("rank").gene[:TOPK])
        D = list(de[de.cluster == state].sort_values("rank").gene[:TOPK])
        if not D:
            continue
        LISTS[(tag, str(state))] = (F, D, [x for x in F if x not in set(D)], [x for x in D if x not in set(F)])
missing = [(r.lineage, r.state) for r in coh.itertuples() if (r.lineage, r.state) not in LISTS]
assert not missing, missing


def ora(genes, i):
    S = set(genes) & BG
    k = len(S & PG[i])
    p = hypergeom.sf(k - 1, M, len(PG[i]), len(S)) if k > 0 else 1.0
    return k, p


rows = []
for i, pr in enumerate(PROG.itertuples()):
    for s in range(74):
        if CLS[i, s] == "P":
            continue
        F, D, Fu, Du = LISTS[(coh.lineage[s], coh.state[s])]
        rec = dict(program=pr.display, collection=pr.collection, term=pr.term, cls=CLS[i, s], lineage=coh.lineage[s],
                   state=coh.state[s], comp=coh.comp[s], subtype=coh["sub"][s], sid=s, cid=CID[s], prog_i=i,
                   n_program_genes=len(PG[i]), n_specific=len(Fu))
        for nm, gs in [("recast", F), ("cosg", D), ("recast_specific", Fu), ("cosg_specific", Du)]:
            rec[f"k_{nm}"], rec[f"p_{nm}"] = ora(gs, i)
        rows.append(rec)
P = pd.DataFrame(rows)
P["dk"] = P.k_recast - P.k_cosg
assert (P.dk == P.k_recast_specific - P.k_cosg_specific).all(), "shared genes must cancel"
P["dlogp"] = -np.log10(P.p_recast) + np.log10(P.p_cosg)
P["dlogp_specific"] = -np.log10(P.p_recast_specific) + np.log10(P.p_cosg_specific)
assert (P.cls == "E").sum() == 99 and (P.cls == "U").sum() == 1164

# check against the paper's enrichment table: every significant row of these 18 terms has the same overlap
R = pd.read_csv(BV / "enrichment_full_matrix_v3sym.csv")
GS = {"RECAST": "recast", "DE_cosg": "cosg", "RECAST_unique": "recast_specific", "DE_cosg_unique": "cosg_specific"}
chk = R.merge(PROG[["display", "collection", "term"]], on=["collection", "term"]).merge(
    P[["program", "lineage", "state", "k_recast", "k_cosg", "k_recast_specific", "k_cosg_specific"]],
    left_on=["display", "lineage", "state"], right_on=["program", "lineage", "state"])
assert len(chk), "no overlap rows to check"
bad = chk[[r.k_overlap != getattr(r, "k_" + GS[r.geneset]) for r in chk.itertuples()]]
assert bad.empty, bad.head()
# and the binary hits behind b: 59 / 48 expected pairs, 16 / 5 method-specific (FDR < 0.05 from the paper's table)
hit = {g: set(map(tuple, R[(R.geneset == g) & (R.fdr < ALPHA)][["lineage", "state", "collection", "term"]].values))
       for g in GS}
for g, nm in GS.items():
    P[f"hit_{nm}"] = [(r.lineage, r.state, r.collection, r.term) in hit[g] for r in P.itertuples()]
E = P[P.cls == "E"]
assert (E.hit_recast.sum(), E.hit_cosg.sum(), E.hit_recast_specific.sum(), E.hit_cosg_specific.sum()) == (59, 48, 16, 5)
print(f"[check] {len(chk)} significant program rows: recomputed overlaps identical; hits 59/48 and 16/5 reproduced")

# ------------------------------------------------------------------ bootstrap (b's design) and permutation
rng2 = np.random.default_rng(SEED + 1)                           # program_enrichment.py: IDX_CLUSTER
SUB_STATES = [np.where(CID == cid)[0] for cid in range(39)]
COMP_SUBS = {c: [cid for cid, (cc, _) in enumerate(SUBS) if cc == c] for c in COMPS}
IDX_CLUSTER = []
for _ in range(B):
    picked = []
    for c in COMPS:
        ids = COMP_SUBS[c]
        picked += list(rng2.choice(ids, size=len(ids), replace=True))
    IDX_CLUSTER.append(np.concatenate([SUB_STATES[cid] for cid in picked]))
rng0 = np.random.default_rng(SEED)
SIGNS = rng0.choice([-1.0, 1.0], size=(NPERM, 39))


def mean_ci_p(df, col):
    """pair-level mean of col, its cluster-bootstrap CI and the cluster sign-flip permutation P"""
    v = df[col].to_numpy(float)
    per_state_sum = np.bincount(df.sid, weights=v, minlength=74)
    per_state_n = np.bincount(df.sid, minlength=74).astype(float)
    est = v.mean()
    bs = np.array([per_state_sum[ix].sum() / per_state_n[ix].sum() for ix in IDX_CLUSTER])
    clus = np.bincount(df.cid, weights=v, minlength=39)
    T = SIGNS @ clus / len(v)
    p = (1 + np.sum(np.abs(T) >= abs(est) - 1e-12)) / (1 + NPERM)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p, int(np.sum(clus != 0))


stats = []
for cls, lab in [("E", "expected"), ("U", "unexpected")]:
    d = P[P.cls == cls]
    for col, what in [("dk", "program genes captured, RECAST - COSG (top 30 = method-specific)"),
                      ("dlogp", "-log10 P, RECAST top 30 - COSG top 30"),
                      ("dlogp_specific", "-log10 P, RECAST-specific - COSG-specific")]:
        est, lo, hi, p, nz = mean_ci_p(d, col)
        stats.append(dict(pairs=lab, n_pairs=len(d), measure=what, column=col, mean=est, ci_lo=lo, ci_hi=hi,
                          p_perm=p, n_clusters_nonzero=nz, n_pos=int((d[col] > 1e-12).sum()),
                          n_zero=int((d[col].abs() <= 1e-12).sum()), n_neg=int((d[col] < -1e-12).sum()),
                          mean_recast=d.k_recast.mean() if col == "dk" else np.nan,
                          mean_cosg=d.k_cosg.mean() if col == "dk" else np.nan))
S = pd.DataFrame(stats)

# ------------------------------------------------------------------ B: permutation P for panel b's six rows
TB = pd.read_csv("/path/to/research/analysis/paper/enrichment_tables/data/enrichment_comparison.csv")


def counts(nm):
    """per-cluster (39) counts: expected hits, unexpected hits"""
    e = P[(P.cls == "E") & P[f"hit_{nm}"]].groupby("cid").size().reindex(range(39), fill_value=0).to_numpy(float)
    u = P[(P.cls == "U") & P[f"hit_{nm}"]].groupby("cid").size().reindex(range(39), fill_value=0).to_numpy(float)
    return e, u


Et, Ut = 99.0, 1164.0
rowsB = []
for grp, (a, b) in [("all", ("recast", "cosg")), ("specific", ("recast_specific", "cosg_specific"))]:
    eR, uR = counts(a)
    eC, uC = counts(b)
    for metric in ["recovery", "precision", "unexpected"]:
        def stat(eRx, uRx, eCx, uCx):
            if metric == "recovery":
                return 100 * (eRx.sum(-1) - eCx.sum(-1)) / Et
            if metric == "unexpected":
                return 100 * (uRx.sum(-1) - uCx.sum(-1)) / Ut
            return 100 * (eRx.sum(-1) / (eRx.sum(-1) + uRx.sum(-1)) - eCx.sum(-1) / (eCx.sum(-1) + uCx.sum(-1)))
        obs = stat(eR, uR, eC, uC)
        sw = SIGNS < 0                                           # swap RECAST and COSG in these clusters
        eRp, eCp = np.where(sw, eC, eR), np.where(sw, eR, eC)
        uRp, uCp = np.where(sw, uC, uR), np.where(sw, uR, uC)
        T = stat(eRp, uRp, eCp, uCp)
        p = (1 + np.sum(np.abs(T) >= abs(obs) - 1e-9)) / (1 + NPERM)
        tb = TB[(TB.group == grp) & (TB.metric == metric)].iloc[0]
        assert abs(tb["diff"] - obs) < 1e-9, (grp, metric, tb["diff"], obs)
        rowsB.append(dict(group=grp, metric=metric, diff=obs, d_lo=tb.d_lo, d_hi=tb.d_hi, p_perm=p))
PBp = pd.DataFrame(rowsB)

P.to_csv(OUT / "program_capture_pairs.csv", index=False)
S.to_csv(OUT / "program_capture_stats.csv", index=False)
PBp.to_csv(OUT / "enrichment_comparison_p.csv", index=False)
pd.set_option("display.width", 250)
print(S.drop(columns=["measure"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(PBp.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("[wrote] program_capture_pairs.csv, program_capture_stats.csv, enrichment_comparison_p.csv")
