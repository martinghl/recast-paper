# validation/reference_choice/independent_panels.py
"""Recovery of two programs defined without RECAST by every reference's top 20, on the 16 Zhao subtypes of
Fig. 4:

  subtype program   the curated subtype panel used for the Recall@20 of Fig. 4 (CURATED in reference_choice.py;
                    5-10 genes per subtype), restricted to the model's genes.
  identity program  CellMarker (human_cell_marker_new.txt; Human, disease Normal, tissue class Blood) genes recorded for
                    the lineage's broad cell name -- CD8+ T cell, CD4+ T cell, B cell, Monocyte -- ranked by the number
                    of supporting publications (PMIDs), support >= 2, ribosomal (RPL/RPS) and mitochondrial (MT-) genes
                    dropped, genes in any curated subtype panel of the same lineage dropped (so the two programs do not
                    overlap), top 20.

Metric per (lineage, target, reference): recall = |top-20 n program| / |program|.  Inputs are the stored top-20 lists
ref_robust_genes_v3.json (no attribution is recomputed).  Outputs:
  ref_robust_indep_v4.csv            one row per lineage x target x reference
  ref_robust_indep_summary_v4.csv    mean / s.e.m. / n over the 16 targets per reference
  ref_robust_indep_programs_v4.json  the two programs per target with the CellMarker support counts
Env: analysis (pandas only; no GPU)."""
import ast, json, re, sys
import numpy as np, pandas as pd

RR = "/path/to/research/analysis/validation/reference_choice"
CM = "/path/to/research/human_cell_marker_new.txt"
GO = set(l.strip() for l in open("/path/to/research/analysis/subtype_benchmark/results/go_genes.txt") if l.strip())
K = 20; K_ID = 20; MIN_SUPPORT = 2
IDENTITY_NAME = {"zhao_CD8": "CD8+ T cell", "zhao_CD4": "CD4+ T cell", "zhao_B": "B cell", "zhao_Mono": "Monocyte"}
HK = re.compile(r"^(RP[LS]\d|RPLP\d|RPSA$|MT-)")

# --- curated subtype programs: parsed from reference_choice.py's source so nothing heavy is imported
src = open(f"{RR}/reference_choice.py").read()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CURATED")
CURATED = ast.literal_eval(node.value)

# --- identity programs from CellMarker
cm = pd.read_csv(CM, sep="\t", usecols=["species", "tissue_class", "disease", "cell_name", "symbol", "pmid"],
                 dtype=str, quoting=3, on_bad_lines="skip")
cm = cm[(cm.species == "Human") & (cm.disease == "Normal") & (cm.tissue_class == "Blood") & cm.symbol.notna()]
cm["symbol"] = cm.symbol.str.upper()
support = cm.groupby(["cell_name", "symbol"])["pmid"].nunique()

genes = json.load(open(f"{RR}/ref_robust_genes_v3.json"))          # "{lineage}|{target}" -> {reference: top-20}
programs, rows = {}, []
for key, per_ref in genes.items():
    lin, tgt = key.split("|")
    state = [g for g in CURATED[lin][tgt] if g in GO]
    lineage_curated = {g for t in CURATED[lin] for g in CURATED[lin][t]}
    s = support.loc[IDENTITY_NAME[lin]]
    s = s[(s >= MIN_SUPPORT)]
    s = s[[g for g in s.index if g in GO and not HK.match(g) and g not in lineage_curated]]
    s = s.sort_values(ascending=False, kind="mergesort")
    identity = list(s.index[:K_ID])
    programs[key] = dict(subtype_program=state, identity_program=identity,
                         identity_support={g: int(s[g]) for g in identity}, identity_cell_name=IDENTITY_NAME[lin])
    for ref, top in per_ref.items():
        top = set(top[:K])
        rows.append(dict(lineage=lin, target=tgt, reference=ref,
                         recall_subtype=len(top & set(state)) / len(state),
                         recall_identity=len(top & set(identity)) / len(identity),
                         n_subtype=len(state), n_identity=len(identity)))
df = pd.DataFrame(rows)
df.to_csv(f"{RR}/ref_robust_indep_v4.csv", index=False)
g = df.groupby("reference")
summ = pd.DataFrame({"recall_subtype": g.recall_subtype.mean(), "recall_identity": g.recall_identity.mean(),
                     "sem_subtype": g.recall_subtype.sem(), "sem_identity": g.recall_identity.sem(),
                     "n_targets": g.size()}).reset_index()
summ.to_csv(f"{RR}/ref_robust_indep_summary_v4.csv", index=False)
json.dump(programs, open(f"{RR}/ref_robust_indep_programs_v4.json", "w"), indent=1)
print("programs:"); [print(f"  {k}: subtype {v['subtype_program']} | identity {v['identity_program']}") for k, v in programs.items()]
print("\nsummary (mean over 16 targets):"); print(summ.round(3).to_string(index=False))
print("\nper-lineage means:"); print(df.groupby(["lineage", "reference"])[["recall_subtype", "recall_identity"]].mean().round(2).unstack("reference").to_string())
