"""Subtype lineages of the benchmark: for each lineage id, the dataset, the label file and the subtypes (with the
minimum cell count).  Per-cell labels come from atlases/bench_labels_<ds>.txt (the datasets' fine annotation), not
from the coarse obs['label'] of the attribution h5ad.  Read by the export, selection and scoring scripts."""
CD8 = ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT", "MAIT cells"]
CD8_DOM = CD8[:-1]                      # dominguez has no MAIT
CD4 = ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Regulatory T cells", "Tem/Temra helper T cells"]
BLIN = ["Naive B cells", "Switched memory B cells", "Non-switched memory B cells", "Memory B cells"]
MONO = ["Classical monocytes", "Non-classical monocytes"]
_BL = {"dom": "bench_labels_dominguez_conde.txt", "ren": "bench_labels_ren.txt",
       "steph": "bench_labels_stephenson.txt", "yosh": "bench_labels_yoshida.txt"}
_STEM = {"dom": "dominguez_conde", "ren": "ren", "steph": "stephenson", "yosh": "yoshida"}

def _b(ds, states):
    return dict(ds=_STEM[ds], label_file=_BL[ds], states=states, min_cells=50)

LINEAGES = {}
for ds in ("dom", "ren", "steph", "yosh"):
    LINEAGES[f"tf_cd8_{ds}"] = _b(ds, CD8_DOM if ds == "dom" else CD8)
    LINEAGES[f"tf_cd4_{ds}"] = _b(ds, CD4)
    LINEAGES[f"tf_mono_{ds}"] = _b(ds, MONO)
for ds in ("ren", "steph", "yosh"):          # dominguez has no B cells
    LINEAGES[f"tf_b_{ds}"] = _b(ds, BLIN)
LINEAGES["fs_nk_pbmcbmn"] = dict(ds="pbmcbmn", label_file="bench_labels_pbmcbmn_l2.txt",
                                 states=["NK", "NK_CD56bright", "NK Proliferating"], min_cells=50)
LINEAGES["fs_b_pbmcbmn"] = dict(ds="pbmcbmn", label_file="bench_labels_pbmcbmn_l2.txt",
                                states=["B naive", "B intermediate", "B memory"], min_cells=50)
LINEAGES["fs_nk_zhao"] = dict(ds="zhao", label_file="bench_labels_zhao.txt",
                              states=["CX3CR1+ NK", "CXCR6+ NK"], min_cells=50)
LINEAGES["fs_asc_zhao"] = dict(ds="zhao", label_file="bench_labels_zhao.txt",
                               states=["noncycling ASC", "cycling ASC", "MZB"], min_cells=50)
# --- non-immune lineages (subtypes with CellMarker markers) ---
LINEAGES["fs_panc_baron"] = dict(ds="baron", label_file="bench_labels_baron.txt",   # pancreatic endocrine (distinct-type control tier)
                                 states=["beta", "alpha", "delta", "gamma"], min_cells=50)
LINEAGES["fs_lung_hlca"] = dict(ds="hlca_lung", label_file="bench_labels_hlca_lung.txt",   # airway/alveolar epithelium
                                states=["basal", "club", "ciliated", "ionocyte", "AT1", "AT2"], min_cells=50)
LINEAGES["fs_gut_smillie"] = dict(ds="smillie_colon", label_file="bench_labels_smillie_colon.txt",   # colon epithelium (healthy)
                                  states=["enterocyte", "goblet", "tuft", "enteroendocrine", "BEST4", "stem"], min_cells=50)
NEW_IDS = list(LINEAGES)                       # 19 immune + non-immune expansion (pancreas/lung/gut)
ZHAO_REUSE = ["ext_zhao_CD8", "ext_zhao_CD4", "ext_zhao_B", "ext_zhao_Mono"]
