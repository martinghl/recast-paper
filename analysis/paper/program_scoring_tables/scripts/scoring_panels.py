#!/usr/bin/env python
"""Curated gene panels of the 28 gene-program scoring tasks (Fig. 6, Supplementary Fig. 8; Supplementary Data sheet 18).

For every scored population: the panel as stored, where it comes from, and the genes actually scored.  The stored
panels are those the scoring loader reads (program_scoring/score_programs.py, DATASETS):
  fine-subtype tasks   encoder_lineages.NEW_PANELS[subtype] for the subtypes of lineage_registry.LINEAGES[task]
                       (panels compiled for this study)
  whole-atlas tasks    the task's panel JSON: atlases/broad_markers/<atlas>.json (CellMarker 3.0-derived expert panels)
                       or new_datasets/markers_score_<task>.json (compiled for this study)
The genes scored are read back from program_scoring/tasks/<task>.h5ad (uns['cand'], uns['panels']; written by
program_scoring/export_tasks.py with the same loader).  A scored panel is the stored panel without the genes outside
the 28,231-gene encoder vocabulary, in the stored order (asserted), and its size equals the task manifest's.
Tasks are in the order of reuse_by_task.csv; populations in the order of the stored panels.

-> paper/program_scoring_tables/data/scoring_panels.tsv
   [task, population, panel_source, n_genes, genes, genes_scored]
Env analysis (needs h5py).  Usage: python -I scoring_panels.py
"""
import csv
import json
import os
import sys

import h5py
import pandas as pd

SB = "/path/to/research/analysis"
sys.path.insert(0, f"{SB}/subtype_benchmark")
import encoder_lineages as EL  # noqa: E402  plain dicts
import lineage_registry as LR  # noqa: E402  plain dicts

TASKS_DIR = f"{SB}/program_scoring/tasks"
REUSE = f"{SB}/paper/program_scoring_tables/data/reuse_by_task.csv"
OUT = f"{SB}/paper/program_scoring_tables/data/scoring_panels.tsv"
GENE_ORDER = "/path/to/scimilarity/model_v1.1/gene_order.tsv"

CELLMARKER_ATLASES = ["pbmc3k", "lawlor", "ss3_pbmc", "ren", "stephenson", "dominguez_conde", "yoshida"]
PANEL_JSON = {t: f"{SB}/atlases/broad_markers/{t}.json" for t in CELLMARKER_ATLASES}
PANEL_JSON.update({t: f"{SB}/new_datasets/markers_score_{t}.json" for t in ("baron_pancreas", "pelka_crc_epi")})
SRC_CM = "CellMarker 3.0-derived expert panel"
SRC_OWN = "Compiled for this study"

tasks = pd.read_csv(REUSE)["dataset"].tolist()
assert len(tasks) == 28, len(tasks)
vocab = {ln.strip() for ln in open(GENE_ORDER) if ln.strip()}
assert len(vocab) == 28231, len(vocab)
manifest = pd.read_csv(f"{TASKS_DIR}/manifest.csv").set_index("task")

rows = []
for t in tasks:
    with h5py.File(f"{TASKS_DIR}/{t}.h5ad", "r") as f:
        cand = [c.decode() if isinstance(c, bytes) else c for c in f["uns/cand"][()]]
        pv = f["uns/panels"][()]
        scored = json.loads(pv.decode() if isinstance(pv, bytes) else pv)
    assert list(scored) == cand, t
    if t in LR.LINEAGES:
        stored = {s: EL.NEW_PANELS[s] for s in LR.LINEAGES[t]["states"] if s in EL.NEW_PANELS}
        src = SRC_OWN
    else:
        stored = json.load(open(PANEL_JSON[t]))
        src = SRC_CM if t in CELLMARKER_ATLASES else SRC_OWN
    assert set(cand) <= set(stored), t
    sizes = dict(x.rsplit(":", 1) for x in manifest.loc[t, "panel_sizes"].split(";"))
    for s in [s for s in stored if s in cand]:
        g_st, g_sc = list(stored[s]), list(scored[s])
        it = iter(g_st)
        assert all(g in it for g in g_sc), (t, s, "scored genes are not in the stored order")
        assert all(g not in vocab for g in g_st if g not in g_sc), (t, s, "a removed gene is in the vocabulary")
        assert int(sizes[s]) == len(g_sc), (t, s)
        rows.append(dict(task=t, population=s, panel_source=src, n_genes=len(g_st), genes=", ".join(g_st),
                         genes_scored=", ".join(g_sc)))

df = pd.DataFrame(rows)
df.to_csv(OUT, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
n_scored = df.genes_scored.map(lambda x: len([g for g in x.split(", ") if g]))
print(f"{len(df)} populations in {df.task.nunique()} tasks")
print("tasks by panel source:", df.groupby("panel_source").task.nunique().to_dict())
print("empty stored panels:", int((df.n_genes == 0).sum()), "| empty after the vocabulary filter:", int((n_scored == 0).sum()))
print("populations with genes removed:", int((df.n_genes != n_scored).sum()))
print("wrote", OUT)
