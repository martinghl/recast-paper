"""Supplementary Data, sheet 2 (enrichment sensitivity): the rows 'Other analysis choice', 'Omit one program' and
'Omit one study', for RECAST and COSG top-30 genes.

For each row: expected and unexpected (program, population) pairs, populations removed, expected pairs recovered
and recovery (%) for each method, the recovery difference, the unexpected-pair rate (%) for each method and its
difference, and 95% intervals of both differences from 2,000 paired cluster-bootstrap draws (subtypes resampled
within each compartment, seed 1), with the frozen 18 programs and expectation matrix of program_enrichment.py.
  Individual-population bootstrap   full panels (its interval is the one of program_enrichment.py)
  SPI1-NK pairs counted as plausible  the five SPI1 x NK pairs moved from unexpected to plausible
  Omit one program                  each of the 18 programs left out in turn
  Omit one study                    the populations of each of the nine studies left out in turn

Writes enrichment_sensitivity_rows.csv next to this script's inputs (BV/results).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("RESEARCH_ANALYSIS", "/path/to/research/analysis")
BV = f"{ROOT}/validation/biology"
PROGRAMS = f"{BV}/programs"
sys.path.insert(0, PROGRAMS)
import enrichment_cohort as cohort   # noqa: E402

ALPHA, N_BOOT, SEED = 0.05, 2000, 0
COMPS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]
STUDIES = {"Domínguez Conde": "dom", "Ren": "ren", "Stephenson": "steph", "Yoshida": "yosh", "Zhao": "zhao",
           "Hao PBMC subset": "pbmcbmn", "Smillie": "smillie", "Human Lung Cell Atlas": "hlca", "Baron": "baron"}


def study(lin):
    for k in ["dom", "ren", "steph", "yosh"]:
        if lin.endswith("_" + k):
            return k
    return {"fs_asc_zhao": "zhao", "fs_nk_zhao": "zhao", "fs_nk_pbmcbmn": "pbmcbmn", "fs_gut_smillie": "smillie",
            "fs_lung_hlca": "hlca", "fs_panc_baron": "baron"}[lin]


def main():
    R = pd.read_csv(f"{BV}/enrichment_full_matrix_v3sym.csv").merge(cohort.COHORT, on=["lineage", "state"], how="inner")
    coh = cohort.COHORT.assign(comp=cohort.COHORT.lineage.map(cohort.lineage_compartment))
    lab_map = {(c, s): lab for c, s, lab in cohort.COLS}
    coh["sub"] = [lab_map[(c, s)] for c, s in zip(coh.comp, coh.state)]
    coh["sid"] = np.arange(len(coh))
    subs = [(c, lab) for c, _, lab in cohort.COLS]
    coh["cid"] = [subs.index((c, s)) for c, s in zip(coh.comp, coh["sub"])]
    coh["study"] = coh.lineage.map(study)
    assert len(coh) == 74 and coh.cid.nunique() == 39
    R = R.merge(coh[["lineage", "state", "sid", "cid"]], on=["lineage", "state"])

    progs = pd.read_csv(f"{PROGRAMS}/programs_frozen.csv")
    assert len(progs) == 18
    prog_order = progs["display"].tolist()
    mask = pd.read_csv(f"{PROGRAMS}/expectation_matrix_frozen.csv", index_col=0).to_numpy()
    assert mask.shape == (18, 39)
    mask_spi1nk = mask.copy()
    i_spi = prog_order.index("SPI1 (PU.1)")
    for s in [("NK", x) for x in ["CD56bright", "CX3CR1+", "CXCR6+", "NK", "Proliferating"]]:
        mask_spi1nk[i_spi, subs.index(s)] = "P"
    assert (mask != mask_spi1nk).sum() == 5

    H = {}
    for gs, m in [("RECAST", "RECAST"), ("DE_cosg", "COSG")]:
        A = np.zeros((18, 74), bool)
        for i, row in progs.iterrows():
            d = R[(R.geneset == gs) & (R.collection == row["collection"]) & (R.term == row["term"]) & (R.fdr < ALPHA)]
            A[i, sorted(set(d.sid))] = True
        H[m] = A

    cid = coh.cid.values
    cls = {k: np.array([[m_[i, cid[s]] for s in range(74)] for i in range(18)])
           for k, m_ in [("primary", mask), ("spi1nk", mask_spi1nk)]}

    def per_state(m, cl, keep_progs=None, keep_states=None):
        A = H[m]
        ip = np.arange(18) if keep_progs is None else np.array(keep_progs)
        w = np.ones(74) if keep_states is None else np.isin(np.arange(74), keep_states).astype(float)
        out = {}
        for k in "EUP":
            M = (cl[ip] == k)
            out[k + "hit"] = ((A[ip] & M).sum(axis=0)) * w
            out[k + "tot"] = M.sum(axis=0) * w
        return out

    def agg(ps, idx=np.arange(74)):
        E, Et, U, Ut = [ps[k][idx].sum() for k in ["Ehit", "Etot", "Uhit", "Utot"]]
        return dict(recovery=100 * E / Et, unexpected=100 * U / Ut)

    sub_states = [np.where(cid == c_)[0] for c_ in range(39)]
    comp_subs = {c: [c_ for c_, (cc, _) in enumerate(subs) if cc == c] for c in COMPS}
    rng = np.random.default_rng(SEED + 1)
    idx_cluster = []
    for _ in range(N_BOOT):
        picked = []
        for c in COMPS:
            ids = comp_subs[c]
            picked += list(rng.choice(ids, size=len(ids), replace=True))
        idx_cluster.append(np.concatenate([sub_states[c_] for c_ in picked]))

    def intervals(pr, pc):
        br = pd.DataFrame([agg(pr, idx) for idx in idx_cluster])
        bc = pd.DataFrame([agg(pc, idx) for idx in idx_cluster])
        dr, du = br.recovery - bc.recovery, br.unexpected - bc.unexpected
        return [float(np.nanpercentile(dr, q)) for q in (2.5, 97.5)] + [float(np.nanpercentile(du, q)) for q in (2.5, 97.5)]

    def row(analysis, variant, pr, pc, removed, comps_removed, with_ci=True):
        er, ec = agg(pr), agg(pc)
        r = dict(analysis=analysis, variant=variant, expected_pairs=int(pr["Etot"].sum()),
                 unexpected_pairs=int(pr["Utot"].sum()), populations_removed=removed,
                 compartments_removed=comps_removed, recast_recovery=er["recovery"], recast_hits=int(pr["Ehit"].sum()),
                 cosg_recovery=ec["recovery"], cosg_hits=int(pc["Ehit"].sum()),
                 recovery_difference=er["recovery"] - ec["recovery"], recast_unexpected=er["unexpected"],
                 cosg_unexpected=ec["unexpected"], unexpected_difference=er["unexpected"] - ec["unexpected"])
        if with_ci:
            r.update(zip(["recovery_difference_low", "recovery_difference_high",
                          "unexpected_difference_low", "unexpected_difference_high"], intervals(pr, pc)))
        return r

    rows = [row("Other analysis choice", "Individual-population bootstrap", per_state("RECAST", cls["primary"]),
                per_state("COSG", cls["primary"]), 0, "None", with_ci=False),
            row("Other analysis choice", "SPI1-NK pairs counted as plausible", per_state("RECAST", cls["spi1nk"]),
                per_state("COSG", cls["spi1nk"]), 0, "None")]
    for i, prog in enumerate(prog_order):
        keep = [j for j in range(18) if j != i]
        rows.append(row("Omit one program", prog, per_state("RECAST", cls["primary"], keep_progs=keep),
                        per_state("COSG", cls["primary"], keep_progs=keep), 0, "None"))
    for name, code in STUDIES.items():
        keep = coh[coh.study != code]["sid"].values
        gone = coh[coh.study == code]
        rows.append(row("Omit one study", name, per_state("RECAST", cls["primary"], keep_states=keep),
                        per_state("COSG", cls["primary"], keep_states=keep), len(gone),
                        "; ".join(sorted(set(gone.comp)))))
    out = pd.DataFrame(rows)
    os.makedirs(f"{BV}/results", exist_ok=True)
    out.to_csv(f"{BV}/results/enrichment_sensitivity_rows.csv", index=False)
    print(out.to_string(max_cols=8))


if __name__ == "__main__":
    main()
