"""CellMarker 3.0 inputs of the subtype marker panels: the file, the evidence thresholds, the subtype -> CellMarker
cell-name table (XW) and the encoder spelling of gene symbols.  Used by panel_rules.py, build_panels.py and
counts/construction_counts.py.

XW lists, for each benchmark subtype, the CellMarker cell_name values that name it (matched exactly, case ignored).
Names that denote another cell type (negations such as "Non-classical monocyte" under classical monocytes, sibling
subtypes, word-sharing types) and malignant cells are not listed; plain "Interneuron" is left out because it does not
say inhibitory.

encoder_spelling() writes a CellMarker symbol (upper-cased) in the spelling of the encoder's gene list when the list
carries it only in another letter case (C10ORF105 -> C10orf105) or under its HGNC previous symbol (CellMarker 3.0 uses
current names, e.g. CIMIP1 -> C20orf85), so the intersection with that list does not drop it.
"""
import collections
import csv
import json
import os

DB = "/path/to/research/human_cell_marker_new.txt"
OUT = os.path.dirname(os.path.abspath(__file__))
MIN_PMID = 2
TOPK = 20
MIN_GENES = 8
GO = "/path/to/research/analysis/subtype_benchmark/results/go_genes.txt"   # encoder gene list
HGNC = f"{OUT}/ref/hgnc_complete_set.txt"                                                # HGNC complete set
XW = {
    # --- immune ---
    "Tcm/Naive cytotoxic T cells": ["Naive CD8+ T cell", "Central memory CD8+ T cell", "Activated naive CD8+ T cell"],
    "Tem/Temra cytotoxic T cells": ["Cytotoxic CD8+ T cell", "Effector memory CD8+ T cell", "Effector CD8+ T cell",
                                    "Terminal effector CD8+ T cell", "Activated effector memory CD8+ T cell"],
    "Tem/Trm cytotoxic T cells":   ["Tissue resident memory T cell", "Tissue resident memory CD8+ T cell",
                                    "Activated tissue resident memory CD8+ T cell",
                                    "Non-recirculating tissue-resident memory T cell",
                                    "CD8+ tissue-resident memory T cell", "Tissue-resident memory T cell"],
    "gdT":                         ["Gamma delta T cell"],
    "MAIT cells":                  ["Mucosal-associated invariant T cell", "CD8+ mucosal-associated invariant T cell"],
    "Tcm/Naive helper T cells":    ["Naive CD4+ T cell", "Central memory CD4+ T cell"],
    "Tem/Effector helper T cells": ["T helper cell", "CD4+ T helper cell", "Effector memory CD4+ T cell",
                                    "Effector CD4+ T cell", "Peripheral T helper cell", "Pre-exhausted T helper cell 1",
                                    "Activated effector memory CD4+ T cell", "T helper cell 2"],
    "Regulatory T cells":          ["Regulatory T cell", "Naive regulatory T cell", "Effector regulatory T cell",
                                    "Natural regulatory T cell", "Induced regulatory T cell",
                                    "TolDC-induced regulatory T cell", "Memory regulatory T cell",
                                    "Activated regulatory T cell", "CD4 peripherally derived regulatory T cell",
                                    "CD4+ CD25+ regulatory T cell", "Follicular regulatory T cell",
                                    "Ovalbumin-specific regulatory T cell", "FOXP3+ natural regulatory T cell",
                                    "Activated CD25+ regulatory T cell", "Regulatory T cell 1"],
    "Tem/Temra helper T cells":    ["Cytotoxic CD4+ T cell", "Cytotoxic T helper cell"],
    "Naive B cells":               ["Naive B cell", "Resting naive B cell"],
    "Switched memory B cells":     ["Switched memory B cell", "Class-switched memory B cell"],
    "Non-switched memory B cells": ["Unswitched memory B cell", "Non-switched memory B cell",
                                    "Non class-switched memory B cell", "Non-switched B cell"],
    "Memory B cells":              ["Memory B cell", "Preplasma memory B cell", "Atypical memory B cell",
                                    "Activated memory B cell", "Double-negative memory B cell", "IgM memory B cell",
                                    "IgG memory B cell", "Early-generated memory B cell", "Effector memory B cell",
                                    "Resting memory B cell", "Natural memory B cell", "Aged memory B cell",
                                    "IgA memory B cell", "Tissue-like memory B cell", "CD27+ memory B cell",
                                    "Intermediate transition memory B cell"],
    "Classical monocytes":         ["Classical monocyte", "OLR1+ classical monocyte"],
    "Non-classical monocytes":     ["Nonclassical monocyte", "Non-classical monocyte", "Non classical monocyte"],
    "NK":                          ["CD16 bright CD56 dim natural killer cell", "CD56+ natural killer cell",
                                    "CD56dim natural killer cell", "CD56dim NK cell"],
    "NK_CD56bright":               ["Natural killer CD56 bright cell", "CD16 dim CD56 bright natural killer cell",
                                    "CD56bright natural killer cell", "CD56-bright natural killer cell"],
    "NK Proliferating":            ["Proliferative NK cell"],
    "CX3CR1+ NK":                  ["CD16 bright CD56 dim natural killer cell", "CD56+ natural killer cell",
                                    "CD56dim natural killer cell", "CD56dim NK cell"],   # circulating cytotoxic = CD56dim
    "CXCR6+ NK":                   [],          # no CellMarker name
    "noncycling ASC":              ["Plasma cell", "IgG plasma cell", "IgA plasma cell", "IgM plasma cell",
                                    "Classical plasma cell", "Circulating plasma cell"],
    "cycling ASC":                 ["Plasmablast", "Proliferative B cell", "Plasmablast B cell", "Unswitched plasmablast",
                                    "Class switched plasmablast", "Plasmablast like cell", "Proliferative plasma cell"],
    "MZB":                         ["Marginal zone B cell", "Marginal zone -like B cell"],
    # --- pancreas (Baron) ---
    "beta":                        ["Beta cell", "Stressed beta cell", "Pancreatic beta cell", "Stem-cell-derived-beta cell",
                                    "Mature pancreatic beta cell", "Type B pancreatic cell"],
    "alpha":                       ["Alpha cell", "Pancreatic alpha cell", "Stem-cell-derived-alpha cell", "Pancreatic a cell"],
    "delta":                       ["Delta cell", "Pancreatic delta cell", "Pancreatic d cell"],
    "gamma":                       ["Gamma cell", "Pancreatic polypeptide cell", "PP cell", "Pancreatic gamma cell",
                                    "Pancreatic PP cell"],
    # --- lung airway / alveolar epithelium ---
    "basal":                       ["Basal cell", "Normal keratinocyte basal cell", "Differentiating basal cell",
                                    "Proximal basal cell", "Proliferating basal cell", "Suprabasal cell",
                                    "Horizontal basal cell", "Respiratory horizontal basal cell", "Multipotent basal cell",
                                    "Repairing basal cell", "Airway basal cell", "Mature basal cell",
                                    "Proliferative basal cell", "Resident basal cell", "Upper airway basal stem cell"],
    "club":                        ["Club cell", "Clara cell"],
    "ciliated":                    ["Ciliated cell", "Proximal ciliated cell", "Multiciliated cell", "Mucous-ciliated cell",
                                    "Respiratory ciliated cell"],
    "ionocyte":                    ["Ionocyte cell", "Ionocyte", "Pre-ionocyte"],
    "AT1":                         ["Alveolar type 1 cell", "Pulmonary alveolar type 1 cell", "Type i pneumocyte"],
    "AT2":                         ["Alveolar type 2 cell", "Type II pneumocyte", "Pulmonary alveolar type 2 cell"],
    # --- intestinal epithelium ---
    "enterocyte":                  ["Enterocyte", "Enterocyte progenitor cell", "Absorptive enterocyte",
                                    "Enterocyte_precursor cell", "Immature enterocyte"],
    "goblet":                      ["Goblet cell", "Immature goblet cell"],
    "tuft":                        ["Tuft cell"],
    "enteroendocrine":             ["Enteroendocrine cell", "Enteroendocrine-like cell"],
    "BEST4":                       ["BEST4+ epithelial cell", "Best4+ enterocyte"],
    "stem":                        ["Intestinal stem cell", "Activeintestinal stem cell", "Small intestinal stem cell"],
    "Paneth":                      ["Paneth cell"],
    # --- brain ---
    "astrocyte":                   ["Astrocyte", "Fibrous astrocyte", "Mature astrocyte", "Astrocyte progenitor cell",
                                    "Protoplasmic astrocyte", "Reactive astrocyte", "Astrocyte of the cerebellum",
                                    "A1 astrocyte", "A2 astrocyte", "Scar-forming astrocyte", "Pan-reactive astrocyte",
                                    "Retinal astrocyte", "Immature astrocyte"],
    "oligodendrocyte":             ["Oligodendrocyte", "Mature oligodendrocyte", "Oligodendrocyte-like cell",
                                    "Pre-oligodendrocyte", "Oligodendrocyte−1"],
    "OPC":                         ["Oligodendrocyte progenitor cell", "Oligodendrocyte precursor cell"],
    "excitatory":                  ["Excitatory neuron", "Superficial layer excitatory neuron",
                                    "Deep layer excitatory neuron", "Broad excitatory neuron", "Mature excitatory neuron",
                                    "Glutamatergic neuron"],
    "inhibitory":                  ["Inhibitory neuron", "Inhibitory interneuron", "GABAergic neuron",
                                    "Intermediate GABAergic neuron"],
}


def encoder_spelling():
    """CellMarker symbol (upper-cased) -> encoder spelling, for symbols the encoder's list carries only in
    another case or under one HGNC previous symbol; symbols already in the list map to themselves."""
    go = [l.strip() for l in open(GO) if l.strip()]
    goset, upper, prev = set(go), collections.defaultdict(set), collections.defaultdict(set)
    for g in go:
        upper[g.upper()].add(g)
    with open(HGNC, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            for p in r["prev_symbol"].strip('"').split("|"):
                if p in goset:
                    prev[r["symbol"].upper()].add(p)

    def spell(g):
        if g in goset:
            return g
        for cand in (upper.get(g, ()), prev.get(g, ())):
            if len(cand) == 1:
                return next(iter(cand))
        return g                                   # not in the encoder list (dropped by the vocabulary rule)
    return spell
