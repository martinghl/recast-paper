"""Data of Supplementary Fig. 6c: program genes that only RECAST selected, and their expression across siblings.

Each example gene has the largest margin over its sibling subtypes in its compartment (data/example_gene_margins.csv
lists all candidates and their margins); its RECAST-specific program hit is in data/method_specific_program_genes.csv
and the mean expression per subtype and dataset in data/example_gene_expression.csv.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (gene, subtype, program, datasets)
EXEMPLARS = [
    ("S100A12", "Classical monocytes",      "myeloid leukocyte migration", ["tf_mono_dom"]),
    ("CSF3R",   "Classical monocytes",      "SPI1 (PU.1)",                 ["tf_mono_dom", "tf_mono_yosh"]),
    ("HMGB2",   "NK Proliferating",         "E2F targets",                 ["fs_nk_pbmcbmn"]),
    ("CD19",    "MZB",                      "PAX5",                        ["fs_asc_zhao"]),
    ("GZMM",    "Tem/Temra helper T cells", "leukocyte cytotoxicity",      ["tf_cd4_ren", "tf_cd4_yosh"]),
    ("DNAH5",   "ciliated",                 "cilium movement",             ["fs_lung_hlca"]),
]

# the four genes drawn, as a 2 x 2 grid read across rows
D_GRID = [["HMGB2", "CD19"], ["GZMM", "DNAH5"]]
D_ORDER = [g for row in D_GRID for g in row]
D_XMAX = 3.0                                        # x-axis limit, log(1 + CP10K)
D_TICKS = [0, 1, 2, 3]


def example_blocks():
    """One block per gene of D_GRID: (gene, subtype, program, datasets, hit row, expression rows, subtype order)."""
    D = pd.read_csv(f"{HERE}/data/method_specific_program_genes.csv")
    E = pd.read_csv(f"{HERE}/data/example_gene_expression.csv")
    G = pd.read_csv(f"{HERE}/data/example_gene_margins.csv")
    by = {g: (g, st, p, l) for g, st, p, l in EXEMPLARS}
    assert set(D_ORDER) <= set(by), f"D_ORDER names genes that are not exemplars: {set(D_ORDER) - set(by)}"
    blocks = []
    for gene in D_ORDER:
        gene, state, prg, lins = by[gene]
        hit = D[(D.gene == gene) & (D.subtype == state) & (D.program == prg)]
        assert len(hit) == 1, f"{gene}/{state}/{prg} is not one RECAST-specific hit ({len(hit)} rows)"
        hit = hit.iloc[0]
        assert sorted(str(hit.lineages).split(";")) == sorted(lins), \
            f"{gene}: cohorts {hit.lineages} do not match the panel's {lins}"
        for lin in lins:                      # the gene must exceed its siblings in every dataset drawn
            g = G[(G.gene == gene) & (G.state == state) & (G.lineage == lin) & (G.program == prg)]
            assert len(g) == 1 and g.margin_mean.iloc[0] > 0, \
                f"{gene} does not clear its siblings in {lin}; it must not be shown as an exemplar"
        sub = E[(E.gene == gene) & (E.lineage.isin(lins))]
        sibs = (sub[sub.state != state].groupby("state").mean_lognorm.max()
                   .sort_values(ascending=False).index.tolist())
        blocks.append((gene, state, prg, lins, hit, sub, [state] + sibs))
    return blocks
