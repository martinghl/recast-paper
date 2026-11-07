# copied from validation/reference_choice/reference_choice.py (importing that module runs the whole experiment)
BROAD = {
    "T":    ["CCR7+ CD4", "CXCR3+ CD4", "Treg", "CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8", "MAIT"],
    "B":    ["CD11c+ mem B", "Classical mem B", "MZB", "Naive B", "TCL1A+ naive B", "ZBTB32+ mem B"],
    "Mono": ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo", "Macrophage", "CD1c+ DCs"],
}
# lineage tag -> (sibling states, own broad, WRONG broad)
LIN = {
    "zhao_Mono": (["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"], "Mono", "T"),
    "zhao_CD8":  (["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"], "T", "B"),
    "zhao_CD4":  (["CCR7+ CD4", "CXCR3+ CD4", "Treg"], "T", "B"),
    "zhao_B":    (["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"], "B", "Mono"),
}

