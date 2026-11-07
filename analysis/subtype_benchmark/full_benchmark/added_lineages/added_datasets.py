"""Added datasets and lineages of the subtype benchmark (lung, gut and brain): source, kept donors and cells, and the
mapping from each dataset's own labels to benchmark subtypes.

Conventions, as for the lung and gut datasets of the original lineages (paper/datasets/scripts/build_hlca.py,
build_smillie.py): healthy adult donors only; the datasets' labels pooled to one subtype per identity (immature /
progenitor forms pooled; transit-amplifying, cycling and unnamed clusters dropped); <= CAP cells per subtype drawn with
random_state=SEED; raw integer counts; HGNC symbols.  LINEAGES / NEW_IDS give the benchmark lineages (lineage ids
fs37_<tissue>_<dataset>) in the format of lineage_registry.LINEAGES."""
CENSUS_VERSION = "2025-11-08"
CAP, SEED = 3000, 0

LUNG = {"basal": "basal", "club": "club", "ciliated": "ciliated", "ionocyte": "ionocyte", "AT1": "AT1", "AT2": "AT2"}

# CELLxGENE datasets, read through Census (X_name='raw'); author labels come from the source h5ad obs
CXG = {
    # Rustam 2023 distal airways: normal adults; N2 (12 y) and N12 (11 y) are children; N7 / N10 may
    # be HLCA donors (age + sex match), so they are left out of an independent unit.
    "rustam2023": dict(
        dataset_ids=["b61a921b-7fa3-4b42-b455-aaaf32447920"], label_col="author_cell_type",
        donors=["N1", "N3", "N4", "N5", "N6", "N8", "N9", "N11"], where={},
        mapping={"Basal cells": "basal", "Secretory cells, major (common) subtype": "club",
                 "Ciliated cells, major (common) subtype": "ciliated", "Ionocytes": "ionocyte",
                 "Alveolar epithelial cells, type 1": "AT1", "Alveolar epithelial cells, type 2": "AT2"}),
    # Natri 2024 peripheral lung: the 13 controls not in HLCA, minus the 5 of unknown age. Its
    # secretory subsets are not mapped: HLCA files most of them under "pre-TB secretory", not club.
    "natri2024": dict(
        dataset_ids=["f14bc322-1322-4184-8d16-409557525ea5"], label_col="manual_annotation_1",
        donors=["THD0016", "THD0021", "THD0022", "THD0023", "VUHD112", "VUHD113", "VUHD115", "VUHD117"],
        where={}, mapping={"AT1": "AT1", "AT2": "AT2", "Basal": "basal", "Ciliated": "ciliated"}),
    # Kong 2023 terminal ileum epithelium: healthy controls only (Type == 'Heal'; CxG 'normal' also
    # holds non-inflamed Crohn's tissue).
    "kong2023ti": dict(
        dataset_ids=["fe4b89d5-461e-440c-a5a8-621b37b122c0"], label_col="Celltype", donors=None,
        where={"Type": ["Heal"]},
        mapping={"Enterocytes TMIGD1 MEP1A": "enterocyte", "Enterocytes TMIGD1 MEP1A GSTA1": "enterocyte",
                 "Enterocytes BEST4": "BEST4", "Stem cells OLFM4": "stem", "Stem cells OLFM4 LGR5": "stem",
                 "Stem cells OLFM4 GSTA1": "stem", "Stem cells OLFM4 PCNA": "stem",
                 "Goblet cells MUC2 TFF1": "goblet", "Goblet cells MUC2 TFF1-": "goblet",
                 "Goblet cells SPINK4": "goblet", "Paneth cells": "Paneth", "Tuft cells": "tuft",
                 "L cells": "enteroendocrine", "Enterochromaffin cells": "enteroendocrine"}),
    # Burclaff 2022: colon regions of 3 organ donors (SI_* lineages are small intestine).
    "burclaff2022colon": dict(
        dataset_ids=["019c7af2-c827-4454-9970-44d5e39ce068"], label_col="lineage", donors=None, where={},
        mapping={"C_earlyACC": "enterocyte", "C_lateACC": "enterocyte", "C_goblet": "goblet",
                 "C_tuft": "tuft", "C_EEC": "enteroendocrine", "C_BEST4": "BEST4", "C_ISC": "stem"}),
    # Rexach 2024 cortex (BA4, insula, V1): the 10 non-demented controls; cell_type is the authors'
    # cell-class annotation.
    "rexach2024": dict(
        dataset_ids=["ac0c6561-7a48-4185-af6f-af799f699172"], label_col="cell_type", donors=None,
        where={"disease": ["normal"]},
        mapping={"astrocyte": "astrocyte", "oligodendrocyte": "oligodendrocyte",
                 "oligodendrocyte precursor cell": "OPC", "glutamatergic neuron": "excitatory",
                 "inhibitory interneuron": "inhibitory"}),
    # Gittings 2023 frontal cortex: controls of the C9-FTD (7) and C9-ALS (5) cohorts; CxG cell_type
    # is only 'neuron', so the authors' cluster_layers carries the excitatory / inhibitory split.
    "gittings2023": dict(
        dataset_ids=["3b8b5de4-3aa1-4ac6-8890-8d03c8219981", "98f5d518-1e60-4307-b732-ab045cd3cbc2"],
        label_col="cluster_layers", donors=None, where={"disease": ["normal"]},
        mapping={"Astrocytes": "astrocyte", "Oligodendrocytes": "oligodendrocyte", "OPCs": "OPC",
                 "L2-L3 Intratelencephalic": "excitatory", "L3-L5 Intratelencephalic Type 1": "excitatory",
                 "L3-L5 Intratelencephalic Type 2": "excitatory", "L5 Extratelencephalic": "excitatory",
                 "L5-L6 Near Projecting": "excitatory", "L6 Corticothalamic / L6B": "excitatory",
                 "L6 Intratelencephalic - Type 1": "excitatory", "L6 Intratelencephalic - Type 2": "excitatory",
                 "Parvalbumin interneurons": "inhibitory", "SV2C LAMP5 Interneurons": "inhibitory",
                 "Somatostatin Interneurons": "inhibitory", "VIP Interneurons": "inhibitory"}),
}

# Hickey 2023 colon (HuBMAP snRNA-seq; Zenodo 7675308 per-sample Cell Ranger MEX + author CellType).
# Mirrors the Smillie mapping: immature forms pooled, TA1/TA2/CyclingTA dropped. Multiome libraries
# only (donors B006, B008-B012): the 3' v3 libraries use another chemistry and mostly another gene
# reference (GRCh38-3.0.0), see build_hickey.py.
HICKEY = dict(
    label_file="Colon_Epithelial_UMAP_CellType.tsv",
    mapping={"Enterocytes": "enterocyte", "Immature Enterocytes": "enterocyte", "Goblet": "goblet",
             "Immature Goblet": "goblet", "Tuft": "tuft", "Enteroendocrine": "enteroendocrine",
             "Best4+ Enterocytes": "BEST4", "Stem": "stem"})

# Maitra 2023 dlPFC (GEO GSE213982 combined matrix; cell name = donor.barcode.broad.cluster).
# Controls from the paper's Supplementary Data 1 (Condition == Control); M24 and M24_2 are one
# subject sequenced twice.
MAITRA = dict(
    controls=["F10", "F13", "F21", "F22", "F23", "F24", "F26", "F29", "F30", "F31", "F32", "F33", "F34",
              "F35", "F36", "F37", "F38", "F7", "M12", "M13", "M15", "M16", "M19", "M2", "M20", "M21",
              "M22", "M24", "M24_2", "M27", "M29", "M3", "M31", "M7", "M9"],
    same_subject={"M24_2": "M24"},
    mapping={"Ast": "astrocyte", "Oli": "oligodendrocyte", "OPC": "OPC", "ExN": "excitatory", "InN": "inhibitory"})

# benchmark lineages on the added datasets: dsid -> stem + subtypes (lineage_registry.LINEAGES format)
GLIA = ["astrocyte", "oligodendrocyte", "OPC"]
NEURON = ["excitatory", "inhibitory"]
GUT = ["enterocyte", "goblet", "tuft", "enteroendocrine", "BEST4", "stem"]
LINEAGES = {
    "fs37_lung_rustam":   dict(ds="rustam2023", states=list(LUNG)),
    "fs37_lung_natri":    dict(ds="natri2024", states=list(LUNG)),
    "fs37_gut_kong":      dict(ds="kong2023ti", states=GUT + ["Paneth"]),
    "fs37_gut_hickey":    dict(ds="hickey2023colon", states=GUT),
    "fs37_gut_burclaff":  dict(ds="burclaff2022colon", states=GUT),
    "fs37_glia_rexach":   dict(ds="rexach2024", states=GLIA),
    "fs37_neuron_rexach": dict(ds="rexach2024", states=NEURON),
    "fs37_glia_gittings": dict(ds="gittings2023", states=GLIA),
    "fs37_neuron_gittings": dict(ds="gittings2023", states=NEURON),
    "fs37_glia_maitra":   dict(ds="maitra2023", states=GLIA),
    "fs37_neuron_maitra": dict(ds="maitra2023", states=NEURON),
}
for _e in LINEAGES.values():
    _e.update(label_file=f"bench_labels_{_e['ds']}.txt", min_cells=50)
NEW_IDS = list(LINEAGES)
