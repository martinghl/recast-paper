"""Shared configuration for the attribution-vs-DE marker-transfer experiment (Option A).

Option A (Gate A): source per-study expression from CELLxGENE Census; compute attributions
ourselves with SCimilarity IG (SIGnature's definition). Two envs:
  - `SC`        : SCimilarity / SIGnature / captum / scanpy  → harmonize, attribute, select, evaluate
  - `sigcensus` : cellxgene_census / tiledbsoma              → fetch per-study expression
Importable from BOTH envs (only stdlib + pathlib here).
"""
from pathlib import Path

ROOT = Path("/path/to/research/external/Signature")
DATA = ROOT / "data"
RAW = DATA / "raw"          # per-study raw-count .h5ad fetched from Census
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

# SCimilarity model (release 1.1 == PLAN's "2023_01_rep0"): encoder + gene_order.tsv (28,231 symbols)
MODEL_DIR = "/path/to/scimilarity/model_v1.1"

# interpreters (two envs)
PY_SC = "/path/to/envs/SC/bin/python"
PY_CENSUS = "/path/to/envs/sigcensus/bin/python"

# --- experiment parameters (from PLAN + D5) ---
SEED = 42
TARGET_CELL_TYPE = "CD8-positive, alpha-beta T cell"   # Tier 1a
K_VALUES = [5, 10, 20, 30, 50, 100]    # full curve
HEADLINE_K = [5, 10]                   # D5: headline on SMALL k (selective panels)
HEADLINE_METRIC = "auprc"              # D5: lead with AUPRC, not k=100 AUROC
MAX_TARGET_CELLS = 2000                # per (study x target)      — PLAN §1.3
MAX_BACKGROUND_CELLS = 4000            # per (study x background)  — PLAN §1.3

# --- Census sourcing (D4/D5) ---
CENSUS_VERSION = "2025-11-08"          # stable/LTS; pinned for reproducibility
# D5 (user): blood-only cross-disease LOSO — hold tissue & cell type constant, vary disease +
# lab/batch. Meta-atlases (HLCA, Tabula Sapiens) and LBCL/tumor are held for a Phase-4
# cross-tissue stress test (not this gate).
STUDY_DATASET_IDS = [
    "9dbab10c-118d-496b-966a-67f1763a6b7d",  # Ren — COVID-19 (blood, 10x 5'v2)
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",  # Perez — lupus (blood, 10x 3'v2)
    "19e46756-9100-4e01-8b0e-23b557558a4c",  # clonal hematopoiesis (blood, 10x 3'v3)
    "c838aec3-03ef-4398-b882-0e3912abfff0",  # healthy / normal PBMC (blood, 10x 5'v2)
    "ebc2e1ff-c8f9-466a-acf4-9d291afaf8b3",  # COVID-19 / influenza (blood, 10x 5'v1)
]
GATE_B_STUDY = "9dbab10c-118d-496b-966a-67f1763a6b7d"  # Ren COVID blood — Gate B signal check

# short labels for plots / tables
STUDY_LABELS = {
    "9dbab10c-118d-496b-966a-67f1763a6b7d": "Ren_COVID",
    "218acb0f-9f2f-4f76-b90b-15a4b7c7f629": "Perez_lupus",
    "19e46756-9100-4e01-8b0e-23b557558a4c": "CHIP",
    "c838aec3-03ef-4398-b882-0e3912abfff0": "Healthy",
    "ebc2e1ff-c8f9-466a-acf4-9d291afaf8b3": "COVID_flu",
}

# --- Gate B signal check: curated CD8+ T markers should out-rank ribosomal/mito genes ---
# (housekeeping matched by prefix at check time: ribosomal RPS*/RPL* excl. RPS6K*, mitochondrial MT-*)
CD8_MARKERS = [
    "CD8A", "CD8B", "GZMK", "GZMB", "GZMA", "GZMH", "NKG7", "CCL5", "KLRG1", "PRF1",
    "EOMES", "IFNG", "CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC2", "GNLY", "CST7",
]
