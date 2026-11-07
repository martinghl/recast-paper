"""The 74 populations of the program-enrichment analysis (Fig. 5, Supplementary Fig. 6) and their 39 subtype columns.

COHORT   the 74 (lineage, state) pairs, from enrichment_cohort74.csv
COLS     the 39 subtype columns in figure order: (compartment, state as named in COHORT, printed label)
lineage_compartment(lineage) -> the compartment of a lineage id
"""
import pandas as pd

ROOT = "/path/to/research/analysis"
BV = f"{ROOT}/validation/biology"

COHORT = pd.read_csv(f"{BV}/enrichment_cohort74.csv")

COMPARTMENTS = ["CD4 T", "CD8 T", "NK", "B / ASC", "Mono", "Lung", "Colon", "Pancreas"]


def lineage_compartment(lin):
    if lin.startswith("tf_cd4"): return "CD4 T"
    if lin.startswith("tf_cd8"): return "CD8 T"
    if lin.startswith("tf_b") or lin == "fs_asc_zhao": return "B / ASC"
    if lin.startswith("tf_mono"): return "Mono"
    if lin.startswith("fs_nk"): return "NK"
    if lin == "fs_lung_hlca": return "Lung"
    if lin == "fs_gut_smillie": return "Colon"
    if lin == "fs_panc_baron": return "Pancreas"
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
assert len({(c, st) for c, st, _ in COLS}) == len(COLS) == 39, len(COLS)
assert [c for c, _, _ in COLS] == sorted([c for c, _, _ in COLS], key=COMPARTMENTS.index), "COLS must follow COMPARTMENTS"
