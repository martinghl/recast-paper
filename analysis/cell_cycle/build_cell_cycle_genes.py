"""Cell-cycle ground truth for the cell-cycle comparisons (Supplementary Fig. 4c, d): a curated pathway
library, independent of the gene lists used to define cell-cycle phases.

  cc_gt_full.txt        Reactome CELL_CYCLE (+ _MITOTIC) union GO-BP cell-cycle terms
  cc_gt_tirosh_out.txt  the same without the Tirosh S/G2M genes (a sensitivity variant)

Both are intersected with SCimilarity's 28,231 genes (a gene no method can select is not counted).
Env: SC.  Inputs: validation/biology/{reactome,gobp}_human.csv."""
import json
import re
import sys

import pandas as pd

sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")

ROOT = "/path/to/research/analysis"
BV = f"{ROOT}/validation/biology"
CR = f"{ROOT}/cell_cycle"

# Tirosh S / G2M genes (left out of the sensitivity variant).
TIROSH = set("""MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 CENPU MLF1IP
HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6
EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8
HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 FAM64A PIMREG
SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 HN1 JPT1 CDC20
TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5
CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA""".split())

# Reactome: the two roots that mean "the cell cycle", not the many regulatory side-branches.
REACT_TERMS = {"REACTOME_CELL_CYCLE", "REACTOME_CELL_CYCLE_MITOTIC"}
# GO-BP: any term whose name is about cell-cycle / mitotic-division process. Kept broad on purpose --
# a broad, curated denominator is harder to game than a hand-picked one.
GOBP_PAT = re.compile(r"CELL_CYCLE|MITOTIC_(CELL_CYCLE|NUCLEAR_DIVISION|SISTER_CHROMATID)|"
                      r"^GOBP_DNA_REPLICATION$|CHROMOSOME_SEGREGATION")


def load(f):
    """Read a decoupler/OmniPath geneset table and drop its non-gene rows.

    These tables encode protein COMPLEXES as pseudo-symbols
    (`COMPLEX:AAAS_AHCTF1_NDC1_...`). Left in, they inflate the apparent size of a term several-fold
    while never matching a real gene, which would silently deflate every recall denominator.
    """
    d = pd.read_csv(f)
    d = d.rename(columns={d.columns[0]: "geneset", d.columns[1]: "gene"})
    d["gene"] = d.gene.astype(str)
    return d[~d.gene.str.startswith("COMPLEX:")]


def main():
    from sibling_targets import go  # noqa: E402  -- only for the 28,231-gene universe
    universe = set(map(str, go))

    react = load(f"{BV}/reactome_human.csv")
    gobp = load(f"{BV}/gobp_human.csv")
    r_hit = react[react.geneset.isin(REACT_TERMS)]
    g_terms = sorted({t for t in gobp.geneset.astype(str).unique() if GOBP_PAT.search(t)})
    g_hit = gobp[gobp.geneset.isin(g_terms)]

    full = sorted({str(g) for g in pd.concat([r_hit.gene, g_hit.gene]).astype(str)} & universe)
    tout = sorted(set(full) - TIROSH)
    # Sensitivity variant: the core mitotic machinery only, a much tighter denominator. Reported so a
    # reader can see whether any conclusion depends on how broadly "cell cycle" is drawn.
    core_terms = {"REACTOME_CELL_CYCLE_MITOTIC", "GOBP_MITOTIC_CELL_CYCLE"}
    core = sorted(({str(g) for g in pd.concat([react, gobp]).query("geneset in @core_terms").gene}
                   & universe) - TIROSH)

    open(f"{CR}/cc_gt_full.txt", "w").write("\n".join(full) + "\n")
    open(f"{CR}/cc_gt_tirosh_out.txt", "w").write("\n".join(tout) + "\n")
    open(f"{CR}/cc_gt_core.txt", "w").write("\n".join(core) + "\n")
    meta = dict(reactome_terms=sorted(REACT_TERMS), n_gobp_terms=len(g_terms),
                gobp_terms_sample=g_terms[:15],
                n_reactome_genes=int(r_hit.gene.nunique()), n_gobp_genes=int(g_hit.gene.nunique()),
                n_reactome_in_universe=len(set(r_hit.gene) & universe),
                n_gobp_in_universe=len(set(g_hit.gene) & universe),
                n_union_in_universe=len(full), n_tirosh=len(TIROSH),
                n_tirosh_in_gt=len(set(full) & TIROSH), n_after_tirosh_removed=len(tout),
                n_core_tirosh_out=len(core), universe=len(universe),
                note=("MKI67 is absent from REACTOME_CELL_CYCLE -- Reactome curates cell-cycle "
                      "machinery, not proliferation markers; GO-BP supplies it."),
                mki67_in_full=("MKI67" in set(full)),
                scored_on="cc_gt_tirosh_out.txt", sensitivity_on="cc_gt_core.txt")
    json.dump(meta, open(f"{CR}/cc_gt_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
