"""Sensitivity of the program-enrichment metrics (Fig. 5) to the 'plausible' class of the expectation matrix.

The primary metrics leave the 69 plausible pairs out of numerator and denominator.  Here all three metrics are
recomputed with every plausible pair counted as unexpected (strict) and as expected (lenient), with the
compartment-stratified subtype-cluster bootstrap of program_enrichment.py (same number of draws and seed).

Writes: enrichment_plausible_sensitivity.csv"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/path/to/research/analysis/validation/biology/programs")
import enrichment_cohort as cohort

BV = "/path/to/research/analysis/validation/biology"
OUT = f"{BV}/programs"
ALPHA, B, SEED = 0.05, 2000, 0

coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment)).reset_index(drop=True)
coh["sid"] = np.arange(len(coh))
CIDX = {(c, s): i for i, (c, s, lab) in enumerate(cohort.COLS)}
coh["cid"] = [CIDX[(r.comp, r.state)] for r in coh.itertuples()]
NS = len(coh)
R = pd.read_csv(f"{BV}/enrichment_full_matrix_v3sym.csv").merge(
    coh[["lineage", "state", "sid", "cid"]], on=["lineage", "state"])
prog = pd.read_csv(f"{OUT}/programs_frozen.csv")
EM = pd.read_csv(f"{OUT}/expectation_matrix_frozen.csv", index_col=0)
CID = coh.cid.values
BASE = np.array([[EM.iloc[i, CID[s]] for s in range(NS)] for i in range(len(prog))])

ARMS = {"RECAST": "RECAST", "COSG": "DE_cosg",
        "RECAST-specific": "RECAST_unique", "COSG-specific": "DE_cosg_unique"}
H = {}
for nm, gs in ARMS.items():
    A = np.zeros((len(prog), NS), bool)
    for i, p in prog.iterrows():
        d = R[(R.geneset == gs) & (R.collection == p.collection) & (R.term == p.term) & (R.fdr < ALPHA)]
        for r in d.itertuples():
            A[i, r.sid] = True
    H[nm] = A

SUB_STATES = [np.where(CID == c)[0] for c in range(len(cohort.COLS))]
COMPS = list(dict.fromkeys(c for c, s, lab in cohort.COLS))
COMP_SUBS = {c: [k for k, (cc, s, lab) in enumerate(cohort.COLS) if cc == c] for c in COMPS}
rng = np.random.default_rng(SEED + 1)          # program_enrichment's rng2, the cluster-bootstrap stream
IDX = []
for _ in range(B):
    pick = []
    for c in COMPS:
        pick += list(rng.choice(COMP_SUBS[c], size=len(COMP_SUBS[c]), replace=True))
    IDX.append(np.concatenate([SUB_STATES[c] for c in pick]))


def metrics(CLS, A, idx):
    E, U = (CLS == "E"), (CLS == "U")
    eh, et = (A & E)[:, idx].sum(), E[:, idx].sum()
    uh, ut = (A & U)[:, idx].sum(), U[:, idx].sum()
    return np.array([100 * eh / et, 100 * eh / (eh + uh) if (eh + uh) else np.nan, 100 * uh / ut])


LABELLINGS = [("primary", "plausible excluded from every metric", BASE),
              ("strict", "plausible counted as unexpected", np.where(BASE == "P", "U", BASE)),
              ("lenient", "plausible counted as expected", np.where(BASE == "P", "E", BASE))]
METRICS = ["recovery", "precision", "unexpected"]
rows = []
for tag, note, CLS in LABELLINGS:
    pt = {nm: metrics(CLS, H[nm], np.arange(NS)) for nm in H}
    dr = {nm: np.array([metrics(CLS, H[nm], ix) for ix in IDX]) for nm in H}
    E, U = (CLS == "E"), (CLS == "U")
    print(f"\n=== {tag}: {note}   E={E.sum()}  U={U.sum()}  P={(CLS == 'P').sum()}")
    for a, b_ in [("RECAST", "COSG"), ("RECAST-specific", "COSG-specific")]:
        d = dr[a] - dr[b_]
        for j, m in enumerate(METRICS):
            v = d[~np.isnan(d[:, j]), j]
            lo, hi = np.percentile(v, [2.5, 97.5])
            star = " *" if (lo > 0 or hi < 0) else "  "
            print(f"   {a:16s} vs {b_:16s} {m:11s} {pt[a][j]:6.1f} vs {pt[b_][j]:6.1f}  "
                  f"D {pt[a][j] - pt[b_][j]:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]{star}")
            rows.append(dict(labelling=tag, labelling_note=note, comparison=f"{a} vs {b_}", metric=m,
                             n_expected=int(E.sum()), n_unexpected=int(U.sum()),
                             recast=pt[a][j], cosg=pt[b_][j], diff=pt[a][j] - pt[b_][j],
                             d_lo=lo, d_hi=hi, ci_excludes_zero=bool(lo > 0 or hi < 0)))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/enrichment_plausible_sensitivity.csv", index=False)
sur = (D[D.metric == "recovery"].groupby("comparison").ci_excludes_zero.all())
print(f"\nrecovery survives every labelling: {dict(sur)}")
print("[wrote] enrichment_plausible_sensitivity.csv   (* = 95 % CI excludes 0)")
