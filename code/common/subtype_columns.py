"""The 39 subtypes of the 74-population enrichment cohort, as columns of Fig. 5a, and their two-tier header.

COLS lists (compartment, subtype as named in shared_data/enrichment_populations.csv, printed label) in reading order
inside each compartment; the printed label drops what the compartment header already says ('Tem/Temra cytotoxic T
cells' prints as 'Tem/Temra' under 'CD8 T').  The header has two tiers: the tissue (Blood, Lung, Colon, Pancreas) over
the lineage inside it.  The five Zhao subtypes come from an object that pools blood, liver and spleen samples, and
their labels carry a dagger.
"""
import os

import pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the code/ folder
POPULATIONS = pd.read_csv(f"{CODE}/shared_data/enrichment_populations.csv")  # 74 populations of 39 subtypes

COMPARTMENTS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]


def compartment_of(lineage):
    """The compartment of a dataset, from its lineage key in the enrichment tables."""
    if lineage.startswith("tf_cd4"): return "CD4 T"
    if lineage.startswith("tf_cd8"): return "CD8 T"
    if lineage.startswith("tf_b") or lineage == "fs_asc_zhao": return "B / ASC"
    if lineage.startswith("tf_mono"): return "Mono"
    if lineage.startswith("fs_nk"): return "NK"
    if lineage == "fs_lung_hlca": return "Lung"
    if lineage == "fs_gut_smillie": return "Colon"
    if lineage == "fs_panc_baron": return "Pancreas"
    return "?"


COLS = [
    ("CD4 T", "Tcm/Naive helper T cells", "Tcm/naive"), ("CD4 T", "Tem/Effector helper T cells", "Tem/effector"),
    ("CD4 T", "Tem/Temra helper T cells", "Tem/Temra"), ("CD4 T", "Regulatory T cells", "Treg"),
    ("CD8 T", "Tcm/Naive cytotoxic T cells", "Tcm/naive"), ("CD8 T", "Tem/Temra cytotoxic T cells", "Tem/Temra"),
    ("CD8 T", "Tem/Trm cytotoxic T cells", "Tem/Trm"), ("CD8 T", "MAIT cells", "MAIT"), ("CD8 T", "gdT", "γδ T"),
    ("NK", "NK_CD56bright", "CD56bright"), ("NK", "CX3CR1+ NK", "CX3CR1+"), ("NK", "CXCR6+ NK", "CXCR6+"),
    ("NK", "NK", "NK"), ("NK", "NK Proliferating", "Proliferating"),
    ("B / ASC", "Naive B cells", "Naive"), ("B / ASC", "Non-switched memory B cells", "Non-switched"),
    ("B / ASC", "Switched memory B cells", "Switched"), ("B / ASC", "Memory B cells", "Memory"),
    ("B / ASC", "MZB", "MZB"), ("B / ASC", "cycling ASC", "ASC cycling"), ("B / ASC", "noncycling ASC", "ASC non-cycling"),
    ("Mono", "Classical monocytes", "Classical"), ("Mono", "Non-classical monocytes", "Non-classical"),
    ("Lung", "AT1", "AT1"), ("Lung", "AT2", "AT2"), ("Lung", "basal", "Basal"), ("Lung", "club", "Club"),
    ("Lung", "ciliated", "Ciliated"), ("Lung", "ionocyte", "Ionocyte"),
    ("Colon", "stem", "Stem"), ("Colon", "enterocyte", "Enterocyte"), ("Colon", "BEST4", "BEST4+"),
    ("Colon", "goblet", "Goblet"), ("Colon", "tuft", "Tuft"), ("Colon", "enteroendocrine", "Enteroendocrine"),
    ("Pancreas", "alpha", "α"), ("Pancreas", "beta", "β"), ("Pancreas", "delta", "δ"), ("Pancreas", "gamma", "γ"),
]
assert len({(c, st) for c, st, _ in COLS}) == len(COLS) == 39
assert [c for c, _, _ in COLS] == sorted([c for c, _, _ in COLS], key=COMPARTMENTS.index)

TISSUES = [("Blood", ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono"]),
           ("Lung", ["Lung"]), ("Colon", ["Colon"]), ("Pancreas", ["Pancreas"])]
LINEAGE_LABEL = {"CD4 T": "CD4 T", "CD8 T": "CD8 T", "NK": "NK", "B / ASC": "B / ASC", "Mono": "Mono",
                 "Lung": "Epithelium", "Colon": "Epithelium", "Pancreas": "Islet"}
ZHAO_SUBTYPES = {("NK", "CX3CR1+ NK"), ("NK", "CXCR6+ NK"), ("B / ASC", "MZB"),
                 ("B / ASC", "cycling ASC"), ("B / ASC", "noncycling ASC")}
assert [c for _, cs in TISSUES for c in cs] == COMPARTMENTS and set(LINEAGE_LABEL) == set(COMPARTMENTS)
assert ZHAO_SUBTYPES == {(compartment_of(l), st) for l, st in zip(POPULATIONS.lineage, POPULATIONS.state)
                         if l.endswith("_zhao")}


def column_label(ci):
    """The printed label of column ci, with a dagger for the Zhao subtypes."""
    c, st, lab = COLS[ci]
    return lab + ("†" if (c, st) in ZHAO_SUBTYPES else "")
