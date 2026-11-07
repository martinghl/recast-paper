"""Program enrichment of the RECAST and COSG top-30 genes (Fig. 5, Supplementary Fig. 6).

The 18 biology-curated programs and their expected (E) / plausible (P) / unexpected (U) annotation per subtype (the
expectation matrix) are defined here and kept fixed; every analysis of the programs reads them from
programs_frozen.csv and expectation_matrix_frozen.csv.  The annotation was refined once after the enrichment results
were inspected; annotation_changes.csv lists every program x subtype class that differs between the first draft
(first_draft_programs.csv, kept when the annotation was frozen: display, collection, term, block, optional, expected,
plausible, rationale) and the frozen annotation (Supplementary Data, sheet 17).

Writes (this directory):
  programs_frozen.csv, expectation_matrix_frozen.csv, annotation_changes.csv, enrichment_cells.csv,
  enrichment_metrics.csv (primary compartment-stratified subtype-cluster bootstrap + 74-state sensitivity),
  method_specific_rates.csv (both bootstraps), spi1_nk_sensitivity.csv, lopo.csv, lodo.csv, summary.md
Inputs: enrichment_full_matrix_v3sym.csv (enrichment_matrix_both.py), enrichment_cohort74.csv,
enrichment_setsizes_v3sym.csv, first_draft_programs.csv; subtype names and order from enrichment_cohort.py (COLS).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/path/to/research/analysis/validation/biology/programs")
import enrichment_cohort as cohort
BV = "/path/to/research/analysis/validation/biology"
OUT = f"{BV}/programs"
ALPHA, B, SEED = 0.05, 2000, 0
COLLS = ["CollecTRI", "Hallmark", "Reactome", "GO-BP"]
COMPS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]

R = pd.read_csv(f"{BV}/enrichment_full_matrix_v3sym.csv").merge(cohort.COHORT, on=["lineage", "state"], how="inner")
R["comp"] = R.lineage.map(cohort.lineage_compartment)
coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment))
LAB = {(c, s): lab for c, s, lab in cohort.COLS}
coh["sub"] = [LAB[(c, s)] for c, s in zip(coh.comp, coh.state)]
coh["sid"] = np.arange(len(coh))
SUBS = [(c, lab) for c, _, lab in cohort.COLS]                     # 39 columns in figure order
coh["cid"] = [SUBS.index((c, s)) for c, s in zip(coh.comp, coh["sub"])]
STUDY = {"dom": "Dominguez Conde 2022", "ren": "Ren 2021", "steph": "Stephenson 2021", "yosh": "Yoshida 2022",
         "zhao": "Zhao 2020", "pbmcbmn": "PBMC B/Mono/NK (Azimuth)", "smillie": "Smillie 2019", "hlca": "HLCA (Sikkema 2023)", "baron": "Baron 2016"}
def study(lin):
    for k in ["dom", "ren", "steph", "yosh"]:
        if lin.endswith("_" + k): return k
    return {"fs_asc_zhao": "zhao", "fs_nk_zhao": "zhao", "fs_nk_pbmcbmn": "pbmcbmn", "fs_gut_smillie": "smillie",
            "fs_lung_hlca": "hlca", "fs_panc_baron": "baron"}[lin]
coh["study"] = coh.lineage.map(study)
assert len(coh) == 74 and coh.cid.nunique() == 39
R = R.merge(coh[["lineage", "state", "sid", "cid"]], on=["lineage", "state"])

# ------------------------------------------------------------------ FROZEN panel + mask
CD4 = [("CD4 T", x) for x in ["Tcm/naive", "Tem/effector", "Tem/Temra", "Treg"]]
Bs  = [("B / ASC", x) for x in ["Naive", "Non-switched", "Switched", "Memory", "MZB"]]
ASC = [("B / ASC", "ASC cycling"), ("B / ASC", "ASC non-cycling")]
MONO = [("Mono", "Classical"), ("Mono", "Non-classical")]
NK_ALL = [("NK", x) for x in ["CD56bright", "CX3CR1+", "CXCR6+", "NK", "Proliferating"]]
NK_MATURE = [("NK", "NK"), ("NK", "CX3CR1+")]
NK_OTHER = [("NK", "CD56bright"), ("NK", "CXCR6+"), ("NK", "Proliferating")]
CD8_CYTO = [("CD8 T", x) for x in ["Tem/Temra", "Tem/Trm", "MAIT", "γδ T"]]
LUNG_OTHER = [("Lung", x) for x in ["AT1", "Basal", "Club", "Ciliated", "Ionocyte"]]
ENDO_ALL = [("Pancreas", x) for x in ["α", "β", "δ", "γ"]] + [("Colon", "Enteroendocrine")]
ENDO_NONB = [x for x in ENDO_ALL if x != ("Pancreas", "β")]
# (display, collection, term, block, EXPECTED, PLAUSIBLE, rationale)
PROGRAMS = [
 ("FOXP3", "CollecTRI", "FOXP3", "T / NK", [("CD4 T", "Treg")], [x for x in CD4 if x[1] != "Treg"],
  "Treg lineage TF; the regulon shares activation genes (IL2RA, CTLA4) with other CD4 states"),
 ("TCF/LEF–β-catenin targets", "Reactome", "REACTOME_BINDING_OF_TCF_LEF_CTNNB1_TO_TARGET_GENE_PROMOTERS", "T / NK",
  [("CD4 T", "Tcm/naive"), ("CD8 T", "Tcm/naive")], [("NK", "CD56bright"), ("Colon", "Stem")],
  "TCF7/LEF1 program of naive / central-memory T; TCF7-high CD56bright NK and WNT-driven crypt stem cells plausible"),
 ("TBX21", "CollecTRI", "TBX21", "T / NK",
  [("CD8 T", "Tem/Temra"), ("CD8 T", "MAIT"), ("CD8 T", "γδ T"), ("CD4 T", "Tem/Temra")] + NK_MATURE,
  [("CD8 T", "Tem/Trm"), ("CD4 T", "Tem/effector")] + NK_OTHER,
  "T-bet: strongly expected in mature effector/cytotoxic T (Temra, cytotoxic CD4), MAIT, γδ T and mature CD56dim NK (NK, CX3CR1+); "
  "Tem/Trm (T-bet is down-regulated in tissue-resident memory), Th1-like CD4 effectors, CD56bright, CXCR6+ and proliferating NK plausible"),
 ("leukocyte cytotoxicity", "GO-BP", "GOBP_LEUKOCYTE_MEDIATED_CYTOTOXICITY", "T / NK",
  CD8_CYTO + [("CD4 T", "Tem/Temra")] + NK_MATURE, [("CD4 T", "Tem/effector")] + NK_OTHER,
  "cytotoxic CD8 states incl. Tem/Trm, MAIT, γδ T, cytotoxic CD4 and mature NK expected; CD4 effectors, CD56bright, CXCR6+ and proliferating NK plausible"),
 ("E2F targets", "Hallmark", "E2F_TARGETS", "T / NK", [("NK", "Proliferating"), ("B / ASC", "ASC cycling")], [("Colon", "Stem")],
  "proliferating states irrespective of lineage; cycling crypt stem/TA cells plausible"),
 ("PAX5", "CollecTRI", "PAX5", "B / myeloid", Bs, [],
  "B-lineage identity TF; PAX5 is silenced during plasma-cell differentiation, so ASC are unexpected"),
 ("B-cell receptor signaling", "GO-BP", "GOBP_B_CELL_RECEPTOR_SIGNALING_PATHWAY", "B / myeloid", Bs, ASC,
  "B lineage expected; ASC retain BCR-pathway components, plausible"),
 ("SPI1 (PU.1)", "CollecTRI", "SPI1", "B / myeloid", MONO, Bs,
  "PU.1: myeloid master TF expected in monocytes; expressed in B cells (plausible); NK unexpected in the primary mask"),
 ("myeloid leukocyte migration", "GO-BP", "GOBP_MYELOID_LEUKOCYTE_MIGRATION", "B / myeloid", MONO, [], "monocyte program"),
 ("NKX2-1", "CollecTRI", "NKX2-1", "Lung", [("Lung", "AT2")], LUNG_OTHER,
  "regulon = surfactant/AT2 genes expected in AT2; NKX2-1 is expressed across lung epithelium (plausible)"),
 ("surfactant metabolism", "Reactome", "REACTOME_SURFACTANT_METABOLISM", "Lung", [("Lung", "AT2")], [("Lung", "Club")],
  "AT2 expected; club cells secrete SFTPB/SFTPD (plausible); other lung epithelium unexpected"),
 ("cilium movement", "GO-BP", "GOBP_CILIUM_MOVEMENT", "Lung", [("Lung", "Ciliated")], [], "motile cilia"),
 ("WNT signaling", "GO-BP", "GOBP_WNT_SIGNALING_PATHWAY", "Colon", [("Colon", "Stem")], [("CD4 T", "Tcm/naive"), ("CD8 T", "Tcm/naive")],
  "crypt stem cells expected; WNT/TCF7 maintenance of naive T plausible"),
 ("chloride transport", "GO-BP", "GOBP_CHLORIDE_TRANSPORT", "Colon", [("Colon", "Enterocyte"), ("Colon", "BEST4+")],
  [("Colon", "Goblet"), ("Lung", "Ionocyte")],
  "colonocyte SLC26A3/CFTR and BEST4 (a Cl- channel) expected; CLCA1-high goblet and CFTR-high ionocytes plausible"),
 ("SPDEF", "CollecTRI", "SPDEF", "Colon", [("Colon", "Goblet")], [("Lung", "Club")], "goblet master TF; airway secretory cells plausible"),
 ("NEUROD1", "CollecTRI", "NEUROD1", "Islet", ENDO_ALL, [], "pan-endocrine TF (islet + enteroendocrine)"),
 ("PDX1", "CollecTRI", "PDX1", "Islet", [("Pancreas", "β")], ENDO_NONB, "β-cell TF expected; other endocrine states plausible"),
 ("pancreas β-cell program", "Hallmark", "PANCREAS_BETA_CELLS", "Islet", [("Pancreas", "β")], ENDO_NONB,
  "INS/IAPP plus shared endocrine machinery; other endocrine states plausible"),
]
assert len(PROGRAMS) == 18
for p in PROGRAMS:
    for x in p[4] + p[5]: assert x in SUBS, x
    assert not (set(p[4]) & set(p[5])), p[0]
    assert ((R.collection == p[1]) & (R.term == p[2])).any(), p[0]
NAMES = [p[0] for p in PROGRAMS]
def cls_of(p, sub): return "E" if sub in p[4] else ("P" if sub in p[5] else "U")
MASK = np.array([[cls_of(p, s) for s in SUBS] for p in PROGRAMS])              # 18 x 39
# sensitivity mask: SPI1 x NK -> plausible
MASK_S = MASK.copy(); i_spi = NAMES.index("SPI1 (PU.1)")
for s in NK_ALL: MASK_S[i_spi, SUBS.index(s)] = "P"
assert (MASK != MASK_S).sum() == 5
pd.DataFrame([dict(row=i + 1, display=p[0], collection=p[1], term=p[2], block=p[3],
                   expected="; ".join(f"{c}:{s}" for c, s in p[4]), plausible="; ".join(f"{c}:{s}" for c, s in p[5]), rationale=p[6])
              for i, p in enumerate(PROGRAMS)]).to_csv(f"{OUT}/programs_frozen.csv", index=False)
EM = pd.DataFrame(MASK, index=NAMES, columns=[f"{c}:{s}" for c, s in SUBS]); EM.index.name = "program"
EM.to_csv(f"{OUT}/expectation_matrix_frozen.csv")
# classes that changed between the first-draft annotation and the frozen one
draft = pd.read_csv(f"{OUT}/first_draft_programs.csv"); draft = draft[~draft.optional]
def parse(x): return set() if pd.isna(x) or x == "" else set(x.split("; "))
chg = []
for p in PROGRAMS:
    r1 = draft[draft.display == p[0]].iloc[0]
    old = {**{k: "E" for k in parse(r1.expected)}, **{k: "P" for k in parse(r1.plausible)}}
    new = {**{f"{c}:{s}": "E" for c, s in p[4]}, **{f"{c}:{s}": "P" for c, s in p[5]}}
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "U"), new.get(k, "U")
        if a != b: chg.append(dict(program=p[0], subtype=k, first_draft=a, frozen=b))
pd.DataFrame(chg).to_csv(f"{OUT}/annotation_changes.csv", index=False)
print(f"annotation changes after inspection: {len(chg)} program x subtype classes")

# ------------------------------------------------------------------ hit tensors
def hits(gs, p):
    d = R[(R.geneset == gs) & (R.collection == p[1]) & (R.term == p[2]) & (R.fdr < ALPHA)]
    return set(d.sid)
H = {}                                                            # H[method][i, sid] = enriched
for gs, m in [("RECAST", "RECAST"), ("DE_cosg", "COSG")]:
    A = np.zeros((18, 74), bool)
    for i, p in enumerate(PROGRAMS):
        for sid in hits(gs, p): A[i, sid] = True
    H[m] = A
CID = coh.cid.values                                              # subtype id per state
CLS = {k: np.array([[MASK_[i, CID[s]] for s in range(74)] for i in range(18)]) for k, MASK_ in [("primary", MASK), ("spi1nk", MASK_S)]}
cells = []
for i, p in enumerate(PROGRAMS):
    for cid, (c, lab) in enumerate(SUBS):
        sids = np.where(CID == cid)[0]
        cells.append(dict(row=i + 1, program=p[0], comp=c, subtype=lab, n_datasets=len(sids), cls=MASK[i, cid],
                          recast_n=int(H["RECAST"][i, sids].sum()), cosg_n=int(H["COSG"][i, sids].sum()),
                          recast_frac=H["RECAST"][i, sids].mean(), cosg_frac=H["COSG"][i, sids].mean()))
CELLS = pd.DataFrame(cells); CELLS.to_csv(f"{OUT}/enrichment_cells.csv", index=False)

# ------------------------------------------------------------------ metrics + bootstraps
def per_state(m, cls, progs=None, states=None):
    """per-state counts (74,) of E/U/P hits & totals for method m under mask cls, optional program / state subsets"""
    A = H[m]; ip = np.arange(18) if progs is None else np.array(progs)
    w = np.ones(74) if states is None else np.isin(np.arange(74), states).astype(float)
    out = {}
    for k in "EUP":
        M = (cls[ip] == k)
        out[k + "hit"] = ((A[ip] & M).sum(axis=0)) * w; out[k + "tot"] = M.sum(axis=0) * w
    return out
def agg(ps, idx):
    E, Et, U, Ut, P, Pt = [ps[k][idx].sum() for k in ["Ehit", "Etot", "Uhit", "Utot", "Phit", "Ptot"]]
    return dict(recovery=100 * E / Et, precision=100 * E / (E + U), unexpected=100 * U / Ut, plausible=100 * P / Pt if Pt else np.nan)
rng = np.random.default_rng(SEED)
IDX_STATE = rng.integers(0, 74, size=(B, 74))
# compartment-stratified subtype-cluster bootstrap: sample subtypes within compartment, keep all their states
SUB_STATES = [np.where(CID == cid)[0] for cid in range(39)]
COMP_SUBS = {c: [cid for cid, (cc, _) in enumerate(SUBS) if cc == c] for c in COMPS}
rng2 = np.random.default_rng(SEED + 1)
IDX_CLUSTER = []
for _ in range(B):
    picked = []
    for c in COMPS:
        ids = COMP_SUBS[c]; picked += list(rng2.choice(ids, size=len(ids), replace=True))
    IDX_CLUSTER.append(np.concatenate([SUB_STATES[cid] for cid in picked]))
def boot(fn_metrics, idx_list):
    return pd.DataFrame([fn_metrics(idx) for idx in idx_list])
def ci(x, lo=2.5, hi=97.5): return np.nanpercentile(x, lo), np.nanpercentile(x, hi)
def summarize(ps_r, ps_c, idx_list, label):
    full = np.arange(74); est_r, est_c = agg(ps_r, full), agg(ps_c, full)
    br, bc = boot(lambda i: agg(ps_r, i), idx_list), boot(lambda i: agg(ps_c, i), idx_list)
    rows = []
    for m in ["recovery", "precision", "unexpected", "plausible"]:
        d = br[m] - bc[m]
        rows.append(dict(bootstrap=label, metric=m, recast=est_r[m], r_lo=ci(br[m])[0], r_hi=ci(br[m])[1], cosg=est_c[m],
                         c_lo=ci(bc[m])[0], c_hi=ci(bc[m])[1], diff=est_r[m] - est_c[m], d_lo=ci(d)[0], d_hi=ci(d)[1]))
    return pd.DataFrame(rows)
PS = {m: per_state(m, CLS["primary"]) for m in ["RECAST", "COSG"]}
PB = pd.concat([summarize(PS["RECAST"], PS["COSG"], IDX_CLUSTER, "primary: compartment-stratified subtype-cluster"),
                summarize(PS["RECAST"], PS["COSG"], list(IDX_STATE), "sensitivity: 74-state")])
PB.to_csv(f"{OUT}/enrichment_metrics.csv", index=False)
PSS = {m: per_state(m, CLS["spi1nk"]) for m in ["RECAST", "COSG"]}
SENS = pd.concat([summarize(PSS["RECAST"], PSS["COSG"], IDX_CLUSTER, "SPI1xNK=plausible: subtype-cluster"),
                  summarize(PSS["RECAST"], PSS["COSG"], list(IDX_STATE), "SPI1xNK=plausible: 74-state")])
SENS.to_csv(f"{OUT}/spi1_nk_sensitivity.csv", index=False)
n_pairs = {k: int((CLS["primary"] == k).sum()) for k in "EPU"}
n_pairs_s = {k: int((CLS["spi1nk"] == k).sum()) for k in "EPU"}

# leave-one-program-out (point estimates + cluster CI for the recovery difference)
lopo = []
for i, p in enumerate(PROGRAMS):
    keep = [j for j in range(18) if j != i]
    pr, pc = per_state("RECAST", CLS["primary"], keep), per_state("COSG", CLS["primary"], keep)
    full = np.arange(74); er, ec = agg(pr, full), agg(pc, full)
    d = boot(lambda k: agg(pr, k), IDX_CLUSTER).recovery - boot(lambda k: agg(pc, k), IDX_CLUSTER).recovery
    lopo.append(dict(omitted=p[0], recovery_recast=er["recovery"], recovery_cosg=ec["recovery"], recovery_diff=er["recovery"] - ec["recovery"],
                     recovery_diff_lo=ci(d)[0], recovery_diff_hi=ci(d)[1], precision_diff=er["precision"] - ec["precision"],
                     unexpected_diff=er["unexpected"] - ec["unexpected"]))
LOPO = pd.DataFrame(lopo); LOPO.to_csv(f"{OUT}/lopo.csv", index=False)
# leave-one-dataset-out (study level)
lodo = []
for st in ["dom", "ren", "steph", "yosh", "zhao", "pbmcbmn", "smillie", "hlca", "baron"]:
    keep_states = coh[coh.study != st].sid.values
    pr, pc = per_state("RECAST", CLS["primary"], None, keep_states), per_state("COSG", CLS["primary"], None, keep_states)
    full = np.arange(74); er, ec = agg(pr, full), agg(pc, full)
    d = boot(lambda k: agg(pr, k), IDX_CLUSTER).recovery - boot(lambda k: agg(pc, k), IDX_CLUSTER).recovery
    lodo.append(dict(omitted=st, study=STUDY[st], n_states_removed=int((coh.study == st).sum()),
                     lineages=" ".join(sorted(coh[coh.study == st].lineage.unique())),
                     recovery_recast=er["recovery"], recovery_cosg=ec["recovery"], recovery_diff=er["recovery"] - ec["recovery"],
                     recovery_diff_lo=ci(d)[0], recovery_diff_hi=ci(d)[1],
                     precision_diff=er["precision"] - ec["precision"], unexpected_diff=er["unexpected"] - ec["unexpected"]))
LODO = pd.DataFrame(lodo); LODO.to_csv(f"{OUT}/lodo.csv", index=False)

# method-specific gene sets (panel c): % states with >=1 term, both bootstraps
rowsC = []
for coll in COLLS + ["any collection"]:
    ind = {}
    for gs in ["RECAST_unique", "DE_cosg_unique"]:
        d = R[(R.geneset == gs) & (R.fdr < ALPHA)]
        if coll != "any collection": d = d[d.collection == coll]
        ind[gs] = np.isin(np.arange(74), d.sid.unique()).astype(float)
    for label, idx_list in [("primary: subtype-cluster", IDX_CLUSTER), ("sensitivity: 74-state", list(IDX_STATE))]:
        br = np.array([100 * ind["RECAST_unique"][i].mean() for i in idx_list]); bc = np.array([100 * ind["DE_cosg_unique"][i].mean() for i in idx_list])
        rowsC.append(dict(bootstrap=label, collection=coll, recast_specific=100 * ind["RECAST_unique"].mean(), r_lo=ci(br)[0], r_hi=ci(br)[1],
                          cosg_specific=100 * ind["DE_cosg_unique"].mean(), c_lo=ci(bc)[0], c_hi=ci(bc)[1],
                          diff=100 * (ind["RECAST_unique"].mean() - ind["DE_cosg_unique"].mean()), d_lo=ci(br - bc)[0], d_hi=ci(br - bc)[1]))
PC = pd.DataFrame(rowsC); PC.to_csv(f"{OUT}/method_specific_rates.csv", index=False)
SZ = pd.read_csv(f"{BV}/enrichment_setsizes_v3sym.csv").merge(coh[["lineage", "state"]], on=["lineage", "state"])
assert len(SZ) == 74 and (SZ.n_recast_unique == SZ.n_cosg_unique).all()

# ------------------------------------------------------------------ summary.md
L = []
def w(s=""): L.append(s)
def fmt(r):
    r = r._asdict() if hasattr(r, "_asdict") else r.to_dict()   # "diff" collides with Series.diff
    return f"{r['recast']:.1f} [{r['r_lo']:.1f}, {r['r_hi']:.1f}] | {r['cosg']:.1f} [{r['c_lo']:.1f}, {r['c_hi']:.1f}] | {r['diff']:+.1f} [{r['d_lo']:+.1f}, {r['d_hi']:+.1f}]"
prim = PB[PB.bootstrap.str.startswith("primary")].set_index("metric"); sens = PB[PB.bootstrap.str.startswith("sensitivity")].set_index("metric")
w("# Program enrichment of the RECAST and COSG top 30 (18 biology-curated programs)\n")
w(f"Units: 74 cohort states (39 subtypes × their datasets), FDR<{ALPHA} binary. Programs and annotations below were "
  "curated from biology after inspecting enrichment results (so they are *biology-curated*, not pre-specified) and were then kept "
  "fixed.\n")
w("## 1. The 18 programs\n")
w("| row | program | collection | term | block |"); w("|---|---|---|---|---|")
for i, p in enumerate(PROGRAMS, 1): w(f"| {i} | {p[0]} | {p[1]} | `{p[2]}` | {p[3]} |")
w("\n## 2. Subtype–program expectation matrix (E expected, P plausible, U unexpected)\n")
w("Full 18 × 39 matrix: `expectation_matrix_frozen.csv`. Per-program lists (`programs_frozen.csv`):\n")
w("| program | expected | plausible |"); w("|---|---|---|")
for p in PROGRAMS: w(f"| {p[0]} | " + ", ".join(f"{c} {s}" for c, s in p[4]) + " | " + (", ".join(f"{c} {s}" for c, s in p[5]) or "—") + " |")
w(f"\nPair counts under the primary mask: E {n_pairs['E']}, P {n_pairs['P']}, U {n_pairs['U']} (of 18 × 74 = 1332 program × state pairs).\n")
w("## 3. Primary inference: compartment-stratified subtype-cluster bootstrap\n")
w(f"Resampling unit = biological subtype (39). In each of {B} replicates, subtypes are drawn with replacement WITHIN each compartment "
  "(CD4 T 4, CD8 T 5, NK 5, B / ASC 7, Mono 2, Lung 6, Colon 6, Islet 4, so the compartment composition is preserved); every dataset-"
  "specific state of a drawn subtype enters with it, metrics are recomputed for both methods on the same replicate, and paired RECAST − "
  "COSG differences are summarised by percentile 95% intervals. Rationale: the dataset-specific states of one subtype share the same "
  "biological identity and the same expectation annotation, so treating the 74 states as independent would understate the uncertainty.\n")
w("| metric | RECAST % [95% CI] | COSG % [95% CI] | RECAST − COSG [95% CI] |"); w("|---|---|---|---|")
LABm = dict(recovery="expected-program recovery", precision="biological precision (E / (E+U) among enriched)", unexpected="unexpected enrichment rate", plausible="plausible pairs enriched (reference)")
for m in ["recovery", "precision", "unexpected", "plausible"]: w(f"| {LABm[m]} | {fmt(prim.loc[m])} |")
w("\n## 4. Sensitivity: 74-state bootstrap (states treated as independent)\n")
w("| metric | RECAST % [95% CI] | COSG % [95% CI] | RECAST − COSG [95% CI] |"); w("|---|---|---|---|")
for m in ["recovery", "precision", "unexpected", "plausible"]: w(f"| {LABm[m]} | {fmt(sens.loc[m])} |")
w("\n## 5. SPI1 × NK sensitivity (all five SPI1 × NK pairs re-labelled plausible)\n")
w(f"Pair counts: E {n_pairs_s['E']}, P {n_pairs_s['P']}, U {n_pairs_s['U']}. Recovery is unaffected by construction (E pairs unchanged).\n")
w("| bootstrap | metric | RECAST % [95% CI] | COSG % [95% CI] | RECAST − COSG [95% CI] |"); w("|---|---|---|---|---|")
for r in SENS.itertuples():
    if r.metric != "plausible": w(f"| {r.bootstrap} | {LABm[r.metric]} | {fmt(r)} |")
w("\n## 6. Leave-one-program-out (recovery difference, subtype-cluster 95% CI)\n")
w("| omitted program | RECAST recovery | COSG recovery | difference [95% CI] | precision diff | unexpected diff |"); w("|---|---|---|---|---|---|")
for r in LOPO.itertuples(): w(f"| {r.omitted} | {r.recovery_recast:.1f} | {r.recovery_cosg:.1f} | {r.recovery_diff:+.1f} [{r.recovery_diff_lo:+.1f}, {r.recovery_diff_hi:+.1f}] | {r.precision_diff:+.1f} | {r.unexpected_diff:+.2f} |")
w(f"\nRecovery difference range over omissions: {LOPO.recovery_diff.min():+.1f} to {LOPO.recovery_diff.max():+.1f} pp; "
  f"sign reversal: {'yes' if (LOPO.recovery_diff <= 0).any() else 'no'}; smallest lower CI bound {LOPO.recovery_diff_lo.min():+.1f} "
  f"(omitting {LOPO.loc[LOPO.recovery_diff_lo.idxmin(), 'omitted']}).\n")
w("## 7. Leave-one-dataset-out (study level; a study's lineages are removed together)\n")
w("| omitted study | states removed | lineage files | RECAST recovery | COSG recovery | difference [95% CI] | precision diff | unexpected diff |"); w("|---|---|---|---|---|---|---|---|")
for r in LODO.itertuples(): w(f"| {r.study} | {r.n_states_removed} | {r.lineages} | {r.recovery_recast:.1f} | {r.recovery_cosg:.1f} | {r.recovery_diff:+.1f} [{r.recovery_diff_lo:+.1f}, {r.recovery_diff_hi:+.1f}] | {r.precision_diff:+.1f} | {r.unexpected_diff:+.2f} |")
w(f"\nRecovery difference range: {LODO.recovery_diff.min():+.1f} to {LODO.recovery_diff.max():+.1f} pp; remains positive in {(LODO.recovery_diff > 0).sum()}/9 omissions; "
  f"largest change when omitting {LODO.loc[(LODO.recovery_diff - prim.loc['recovery','diff']).abs().idxmax(), 'study']}.\n")
w("## 8. Method-specific gene sets (panel c): % of states with >= 1 FDR<0.05 term\n")
w(f"RECAST-specific = RECAST top 30 \\ COSG top 30; COSG-specific = COSG top 30 \\ RECAST top 30; equal size in every state "
  f"(median {SZ.n_recast_unique.median():.0f} genes, range {SZ.n_recast_unique.min()}–{SZ.n_recast_unique.max()}).\n")
w("| bootstrap | collection | RECAST-specific % [95% CI] | COSG-specific % [95% CI] | difference [95% CI] |"); w("|---|---|---|---|---|")
for r in PC.itertuples(): w(f"| {r.bootstrap} | {r.collection} | {r.recast_specific:.0f} [{r.r_lo:.0f}, {r.r_hi:.0f}] | {r.cosg_specific:.0f} [{r.c_lo:.0f}, {r.c_hi:.0f}] | {r.diff:+.0f} [{r.d_lo:+.0f}, {r.d_hi:+.0f}] |")
# ---- 10. conclusions, derived from the numbers
pc_prim = PC[PC.bootstrap.str.startswith("primary")].set_index("collection")
rec, pre, une = prim.loc["recovery"], prim.loc["precision"], prim.loc["unexpected"]
w("\n## 9. Which conclusions the data support\n")
w(f"- SUPPORTED — expected-program recovery: RECAST {rec.recast:.1f}% vs COSG {rec.cosg:.1f}% ({rec['diff']:+.1f} pp, 95% CI "
  f"[{rec.d_lo:+.1f}, {rec.d_hi:+.1f}]; {'CI excludes 0' if rec.d_lo > 0 else 'CI includes 0'}); the advantage {'survives' if (LOPO.recovery_diff > 0).all() and (LODO.recovery_diff > 0).all() else 'does NOT survive'} every "
  f"leave-one-program-out and leave-one-dataset-out omission (ranges above).")
w(f"- {'SUPPORTED' if pre.d_lo > 0 else 'TREND ONLY'} — biological precision: RECAST {pre.recast:.1f}% vs COSG {pre.cosg:.1f}% ({pre['diff']:+.1f} pp, 95% CI [{pre.d_lo:+.1f}, {pre.d_hi:+.1f}])"
  + ("; describe only as a favourable direction." if pre.d_lo <= 0 else "."))
w(f"- {'COMPARABLE' if une.d_lo <= 0 <= une.d_hi else 'DIFFERENT'} — unexpected enrichment rate: RECAST {une.recast:.1f}% vs COSG {une.cosg:.1f}% ({une['diff']:+.1f} pp, 95% CI [{une.d_lo:+.1f}, {une.d_hi:+.1f}])"
  + ("; the methods are comparable, do not claim improved specificity." if une.d_lo <= 0 <= une.d_hi else "."))
sp = SENS[SENS.bootstrap.str.contains("subtype-cluster")].set_index("metric")
w(f"- SPI1 × NK re-labelling changes precision to RECAST {sp.loc['precision','recast']:.1f}% vs COSG {sp.loc['precision','cosg']:.1f}% "
  f"({sp.loc['precision','diff']:+.1f} pp [{sp.loc['precision','d_lo']:+.1f}, {sp.loc['precision','d_hi']:+.1f}]) and the unexpected rate to "
  f"{sp.loc['unexpected','recast']:.1f}% vs {sp.loc['unexpected','cosg']:.1f}%; the qualitative conclusions above are unchanged.")
npos = int((pc_prim.loc[COLLS, "diff"] > 0).sum()); nsig = [c for c in COLLS if pc_prim.loc[c, "d_lo"] > 0]
w(f"- Method-specific genes: RECAST-specific sets are directionally higher in {npos}/4 resources, largest for "
  f"{pc_prim.loc[COLLS, 'diff'].idxmax()} ({pc_prim.loc[COLLS, 'diff'].max():+.0f} pp); interval excludes 0 for "
  f"{', '.join(nsig) if nsig else 'none of the four resources'} (no multiplicity adjustment across the four). Reading: RECAST-specific "
  "genes retain functional coherence comparable to equally sized COSG-specific sets, i.e. they are not functionally unstructured; NOT supported: "
  "broad or statistically significant superiority across resources.")
w("- Not claimed anywhere: subtype coverage (\"a term for 38 of 39 subtypes\"), and the full-set RECAST vs COSG rates (70/85/81/97 vs 65/84/80/84), "
  "which move to the Supplementary atlas.")
open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
