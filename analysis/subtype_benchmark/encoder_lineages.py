"""Lineages of the cross-encoder comparison (Supplementary Fig. 2e): 6 sibling-subtype lineages (helper T, cytotoxic T,
B maturation) in 4 datasets (zhao, dominguez_conde, pbmcbmn, yoshida), and the curated panels of the program-scoring
lineages.  Plain dicts, no heavy imports.  Each entry: stem (atlases/{stem}_attribution_scimilarity.h5ad), lab (label
file in atlases/, or None -> obs['label']), curated {state: [markers]}."""

# canonical per-state marker panels (from e4_bootstrap MARKERS + run_benchmark)
_naiveT = ["CCR7", "SELL", "TCF7", "LEF1", "IL7R", "TXK", "NOSIP", "MAL", "TRABD2A", "ADTRP", "PLAC8", "LDHB"]
_eff_helper = ["GZMK", "GZMA", "CCL5", "KLRB1", "ANXA1", "S100A4", "IL32", "AHNAK", "CCR6", "GPR183", "CRTAM", "CCR4"]
_treg = ["FOXP3", "IL2RA", "IKZF2", "CTLA4", "TNFRSF4", "TNFRSF18", "TIGIT", "RTKN2", "TBC1D4", "LAYN", "IL10RA"]
_eff_cyto = ["GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "FGFBP2", "FCGR3A", "KLRD1", "CX3CR1", "SPON2", "KLRG1", "CMC1"]
_trm = ["ITGAE", "ITGA1", "ZNF683", "CXCR6", "XCL1", "XCL2", "CD69", "GZMK", "DUSP2", "CD8A", "CD160", "KLRC1"]
_gdt = ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "KLRC1", "KLRD1", "TRGV9", "TRDV3", "KLRG1", "NKG7"]
_naiveB = ["TCL1A", "IGHD", "FCER2", "IL4R", "YBX3", "TSPAN13", "PLPP5", "BACH2", "SELL", "VPREB3", "CD72", "IGHM"]
_interB = ["TNFRSF13B", "COCH", "AIM2", "CRTAM", "MARCKS", "RALGPS2", "SSPN", "SCIMP", "CD72"]
_memB = ["CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH", "CRTAM", "CD80", "ITGB1", "TEX9", "GRIP2", "LINC01781", "BANK1"]
_nonswB = ["IGHM", "IGHD", "CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH"]
_switchB = ["IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHGP", "CD27", "TNFRSF13B", "AIM2", "S100A10"]

CFG = {
    "zhao_CD8": dict(stem="zhao", lab=None, curated={
        "CCR7+ CD8": ["CCR7", "SELL", "TCF7", "LEF1", "MAL", "NOSIP", "IL7R", "TRABD2A"],
        "CRCR3+ CD8": ["CXCR3", "GZMK", "IL7R", "CD27", "GPR183", "LTB", "CD28", "SLAMF1"],
        "CX3CR1+ CD8": ["CX3CR1", "FGFBP2", "GNLY", "GZMB", "GZMH", "PRF1", "NKG7", "FCGR3A", "KLRD1", "ZNF683"],
        "CXCR6+ CD8": ["CXCR6", "ITGAE", "ZNF683", "CD69", "XCL1", "XCL2", "CD160", "KLRC1", "TRDC", "KLRF1"]}),
    "zhao_B": dict(stem="zhao", lab=None, curated={
        "Naive B": ["IGHD", "TCL1A", "FCER2", "IL4R", "YBX3", "BACH2", "IGHM"],
        "TCL1A+ naive B": ["TCL1A", "IGHD", "FCER2", "PLPP5", "CD72", "IL4R"],
        "Classical mem B": ["CD27", "TNFRSF13B", "AIM2", "CD82", "S100A10", "COCH"],
        "CD11c+ mem B": ["ITGAX", "FCRL5", "TBX21", "ZEB2", "FGR", "NKG7", "SLC11A1"],
        "ZBTB32+ mem B": ["ZBTB32", "NKG7", "GZMA", "CCL5", "MT1F"],
        "MZB": ["CR2", "CD1C", "CD24", "PLD4", "RALGPS2", "DNASE1L3"]}),
    "dom_helperT": dict(stem="dominguez_conde", lab="bench_labels_dominguez_conde.txt", curated={
        "Tcm/Naive helper T cells": _naiveT, "Tem/Effector helper T cells": _eff_helper, "Regulatory T cells": _treg}),
    "dom_cytoT": dict(stem="dominguez_conde", lab="bench_labels_dominguez_conde.txt", curated={
        "Tcm/Naive cytotoxic T cells": _naiveT, "Tem/Temra cytotoxic T cells": _eff_cyto,
        "Tem/Trm cytotoxic T cells": _trm, "gdT": _gdt}),
    "pbmcbmn_Bmat": dict(stem="pbmcbmn", lab="bench_labels_pbmcbmn_l2.txt", curated={
        "B naive": _naiveB, "B intermediate": _interB, "B memory": _memB}),
    "yoshida_Bsw": dict(stem="yoshida", lab="bench_labels_yoshida.txt", curated={
        "Naive B cells": _naiveB, "Non-switched memory B cells": _nonswB,
        "Switched memory B cells": _switchB, "Memory B cells": _memB}),
}
# The pbmcbmn kappa/lambda B subtypes are not included: they differ almost only by the Ig light-chain gene (IGKC vs
# IGLC2/3), which none of the encoders separates (subtype kNN F1 about 0.47).
# short pretty names + T/B compartment for the figure
PRETTY = {"zhao_CD8": "CD8 T (zhao)", "zhao_B": "B (zhao)", "dom_helperT": "helper T (dominguez)",
          "dom_cytoT": "cyto T (dominguez)", "pbmcbmn_Bmat": "B matur. (pbmcbmn·l2)", "yoshida_Bsw": "B switch (yoshida)"}
ORDER = ["zhao_CD8", "dom_helperT", "dom_cytoT", "zhao_B", "pbmcbmn_Bmat", "yoshida_Bsw"]

# --- curated panels of the program-scoring lineages ---
_mait = ["SLC4A10", "KLRB1", "ZBTB16", "RORC", "IL7R", "NCR3", "DPP4", "CEBPD", "GZMK", "TRAV1-2", "CXCR6", "IL18R1"]
_temra_helper = ["GZMB", "GNLY", "NKG7", "GZMH", "FGFBP2", "CX3CR1", "PRF1", "KLRG1", "FCGR3A", "SPON2"]
_mono_cl = ["CD14", "S100A8", "S100A9", "S100A12", "LYZ", "VCAN", "FCN1", "SELL", "CSF3R"]
_mono_ncl = ["FCGR3A", "CDKN1C", "LST1", "MS4A7", "RHOC", "IFITM3", "LILRB2", "SIGLEC10", "HES4"]
_nk_dim = ["NKG7", "GNLY", "PRF1", "GZMB", "FCGR3A", "KLRD1", "FGFBP2", "SPON2", "CST7"]
_nk_bright = ["NCAM1", "GZMK", "SELL", "XCL1", "XCL2", "IL7R", "GPR183", "KLRC1", "TCF7"]
_nk_prolif = ["MKI67", "TOP2A", "TYMS", "STMN1", "NKG7", "GNLY", "UBE2C", "BIRC5"]
_nk_cx3cr1 = ["CX3CR1", "FGFBP2", "FCGR3A", "GNLY", "PRF1", "NKG7", "SPON2"]
_nk_cxcr6 = ["CXCR6", "ITGAE", "CD69", "XCL1", "XCL2", "ZNF683", "KLRC1"]
_asc_noncyc = ["MZB1", "XBP1", "PRDM1", "SDC1", "DERL3", "IGHG1", "CD38", "TNFRSF17", "JCHAIN"]
_asc_cyc = ["MKI67", "TOP2A", "MZB1", "XBP1", "SDC1", "TYMS", "UBE2C", "JCHAIN"]
_mzb = ["CR2", "CD1C", "CD24", "PLD4", "RALGPS2", "DNASE1L3", "FCER2", "IGHD"]
NEW_PANELS = {
    "MAIT cells": _mait, "Tem/Temra helper T cells": _temra_helper,
    "Classical monocytes": _mono_cl, "Non-classical monocytes": _mono_ncl,
    "NK": _nk_dim, "NK_CD56bright": _nk_bright, "NK Proliferating": _nk_prolif,
    "CX3CR1+ NK": _nk_cx3cr1, "CXCR6+ NK": _nk_cxcr6,
    "noncycling ASC": _asc_noncyc, "cycling ASC": _asc_cyc, "MZB": _mzb,
    # T/B states reuse the existing module-level panels:
    "Tcm/Naive cytotoxic T cells": _naiveT, "Tem/Temra cytotoxic T cells": _eff_cyto,
    "Tem/Trm cytotoxic T cells": _trm, "gdT": _gdt,
    "Tcm/Naive helper T cells": _naiveT, "Tem/Effector helper T cells": _eff_helper, "Regulatory T cells": _treg,
    "Naive B cells": _naiveB, "Switched memory B cells": _switchB,
    "Non-switched memory B cells": _nonswB, "Memory B cells": _memB,
    # pbmcbmn Azimuth-l2 B-maturation axis (naive->intermediate->memory; distinct from the switching axis above):
    "B naive": _naiveB, "B intermediate": _interB, "B memory": _memB,
}
