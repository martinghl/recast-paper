"""CellMarker marker sets for the subtypes (/path/to/research/human_cell_marker_new.txt).  Filter: species Human,
disease Normal, gene supported by >= MIN_PMID distinct PMIDs; per subtype, CellMarker cell_name variants are pooled
and the TOPK genes by PMID support kept.  Subtypes with fewer than MIN_GENES genes are left out of the marker
benchmark (they remain in the marker-free endpoints).  Writes results/cellmarker_gt_fine.json  {state: [genes]}.
Env SC."""
import collections, json, os

DB = "/path/to/research/human_cell_marker_new.txt"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
MIN_PMID = 2      # gene must appear in >= 2 distinct PMIDs
TOPK = 20         # top-20 by PMID support -> the expert set
MIN_GENES = 8     # a subtype needs >= 8 CellMarker genes to enter the marker benchmark

# fine-state string (EXACT, as in lineage_registry) -> CellMarker cell_name variants (lowercased substrings).
# NKT is globally excluded so "natural killer" patterns never leak into "Natural killer T cell".
XW = {
    "Tcm/Naive cytotoxic T cells": ["naive cd8+ t cell", "central memory cd8+ t cell", "central memory cd8-positive"],
    "Tem/Temra cytotoxic T cells": ["effector memory cd8+ t cell", "effector cd8+ t cell", "cytotoxic cd8+ t cell", "terminally differentiated effector memory cd8"],
    "Tem/Trm cytotoxic T cells":   ["tissue resident memory cd8+ t cell", "cd8+ tissue-resident memory", "tissue resident memory t cell", "tissue-resident memory t cell", "cd8 tissue resident memory"],
    "gdT":                         ["gamma delta t cell"],
    "MAIT cells":                  ["mucosal-associated invariant t cell"],
    "Tcm/Naive helper T cells":    ["naive cd4+ t cell", "central memory cd4+ t cell"],
    "Tem/Effector helper T cells": ["effector cd4+ t cell", "effector memory cd4+ t cell", "t helper cell"],
    "Regulatory T cells":          ["regulatory t cell"],
    "Tem/Temra helper T cells":    ["cytotoxic cd4+ t cell", "terminally differentiated effector memory cd4"],
    "Naive B cells":               ["naive b cell"],
    "Switched memory B cells":     ["class-switched memory b cell", "switched memory b cell", "class switched memory b cell"],
    "Non-switched memory B cells": ["unswitched memory b cell", "non-switched memory b cell", "non class-switched memory b cell", "non-switched b cell"],
    "Memory B cells":              ["memory b cell"],
    "Classical monocytes":         ["classical monocyte"],
    "Non-classical monocytes":     ["nonclassical monocyte", "non-classical monocyte"],
    "NK":                          ["cd56dim natural killer", "cd16-positive cd56-dim", "cd16 bright cd56 dim", "cd56dim nk", "cd56+ natural killer"],
    "NK_CD56bright":               ["cd56bright natural killer", "natural killer cd56 bright", "cd56-bright natural killer", "cd16 dim cd56 bright"],
    "NK Proliferating":            ["proliferative nk cell", "dividing natural killer cell", "proliferative natural killer"],
    "CX3CR1+ NK":                  ["cd56dim natural killer", "cd16-positive cd56-dim", "cd16 bright cd56 dim", "cd56dim nk", "cd56+ natural killer"],  # circulating cytotoxic = CD56dim
    "CXCR6+ NK":                   ["tissue-resident natural killer", "tissue resident natural killer"],
    "noncycling ASC":              ["plasma cell"],
    "cycling ASC":                 ["plasmablast", "proliferative b cell"],
    "MZB":                         ["marginal zone b cell", "marginal zone -like b cell"],
    # --- non-immune subtypes, all >= 8 genes at MIN_PMID = 2 ---
    "beta":                        ["beta cell", "type b pancreatic", "insulin-secreting"],
    "alpha":                       ["alpha cell", "pancreatic a cell", "glucagon-secreting"],
    "delta":                       ["delta cell", "pancreatic d cell", "somatostatin-secreting"],
    "gamma":                       ["pp cell", "gamma cell", "pancreatic polypeptide"],
    # lung airway/alveolar epithelium (HLCA)
    "basal":                       ["respiratory basal", "airway basal", "basal cell"],
    "club":                        ["club cell", "clara cell", "bronchiolar secretory"],
    "ciliated":                    ["ciliated cell"],
    "ionocyte":                    ["ionocyte"],
    "AT1":                         ["type i pneumocyte", "alveolar type 1", "type 1 alveolar", "at1 cell", "pulmonary alveolar type i"],
    "AT2":                         ["type ii pneumocyte", "alveolar type 2", "type 2 alveolar", "at2 cell", "pulmonary alveolar type ii"],
    # gut colon epithelium (Smillie)
    "enterocyte":                  ["enterocyte", "absorptive cell"],
    "goblet":                      ["goblet cell"],
    "tuft":                        ["tuft cell"],
    "enteroendocrine":             ["enteroendocrine"],
    "BEST4":                       ["best4"],
    "stem":                        ["intestinal stem", "crypt stem", "intestinal crypt stem"],
}


def build():
    pmids = collections.defaultdict(lambda: collections.defaultdict(set))  # state -> gene -> {pmid}
    with open(DB, encoding="utf-8", errors="ignore") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 20 or f[0] != "Human" or f[4] != "Normal":
                continue
            cn = f[6].lower(); sym = f[9].strip(); pmid = f[16].strip()
            if not sym or sym == "NA":
                continue
            if "natural killer t" in cn or "nkt" in cn:      # never leak NKT into NK states
                continue
            for state, pats in XW.items():
                if any(p in cn for p in pats):
                    pmids[state][sym.upper()].add(pmid)
    gt, dropped = {}, {}
    for state in XW:
        ranked = sorted(((g, len(pm)) for g, pm in pmids[state].items() if len(pm) >= MIN_PMID),
                        key=lambda x: -x[1])
        genes = [g for g, _ in ranked[:TOPK]]
        if len(ranked) >= MIN_GENES:
            gt[state] = genes
        else:
            dropped[state] = len(ranked)
    os.makedirs(OUT, exist_ok=True)
    json.dump(gt, open(f"{OUT}/cellmarker_gt_fine.json", "w"), indent=1)
    print(f"[cellmarker GT] {len(gt)} states kept, {len(dropped)} dropped (< {MIN_GENES} genes)")
    for s, gs in gt.items():
        print(f"  KEEP  {s:32s} n={len(gs):2d}  {gs[:10]}")
    for s, n in dropped.items():
        print(f"  DROP  {s:32s} n={n}  (fewer than MIN_GENES)")
    return gt, dropped


if __name__ == "__main__":
    build()
