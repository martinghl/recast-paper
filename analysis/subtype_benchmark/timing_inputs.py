"""Objects of the RECAST timing (time_recast_package.py): the 20 subtype lineages and the six atlases.  Counts come from
the *_fullgene.h5ad objects in SCimilarity gene order (select_recast._load_raw); label sets and the minimum-cells rule
are those of time_recast_stages.py."""
import sys
import numpy as np, pandas as pd, anndata as ad
SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path: sys.path.insert(0, p)
import lineage_registry as C                       # noqa: E402
from select_recast import _load_raw              # noqa: E402
from scoring_benchmark.datasets import _go_gene_order, SCIMILARITY_MODEL   # noqa: E402,F401
ATLASES = f"{SB}/atlases"; OUT = f"{SB}/subtype_benchmark/results"
LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]
ATLASES = ["pbmc3k", "lawlor", "dominguez_conde", "ren", "stephenson", "yoshida"]
ATLAS_MIN_CELLS = 50
_go = None
def go():
    global _go
    if _go is None: _go = _go_gene_order()
    return _go

def load_lineage(dsid):
    """AnnData of raw counts (go order) for one fine-state lineage; obs['state'] = fine state."""
    e = C.LINEAGES[dsid]; stem = e["ds"]
    lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r")
    obs = np.asarray(att.obs_names).astype(str); att.file.close(); del att
    keepstates = sorted([s for s in e["states"] if (lab_all == s).sum() >= e["min_cells"]])
    keep = np.isin(lab_all, keepstates)
    a = ad.AnnData(_load_raw(stem, obs[keep])); a.var_names = go(); a.obs["state"] = lab_all[keep].astype(str)
    return a

def load_atlas(ds):
    """AnnData of raw counts (go order) for one whole atlas; obs['type'] = every annotated type
    with >= ATLAS_MIN_CELLS cells (program_scoring/distinct_labels)."""
    lab = np.array([l.strip() for l in open(f"{SB}/program_scoring/distinct_labels/labels_{ds}.txt")])
    att = ad.read_h5ad(f"{ATLASES}/{ds}_attribution_scimilarity.h5ad", backed="r")
    obs = np.asarray(att.obs_names).astype(str); att.file.close(); del att
    assert len(lab) == len(obs), (ds, len(lab), len(obs))
    vc = pd.Series(lab).value_counts(); keeptypes = sorted(vc[vc >= ATLAS_MIN_CELLS].index)
    keep = np.isin(lab, keeptypes)
    a = ad.AnnData(_load_raw(ds, obs[keep])); a.var_names = go(); a.obs["type"] = lab[keep].astype(str)
    return a

OBJECTS = [(d, "lineage", "state") for d in LIN20] + [(d, "atlas", "type") for d in ATLASES]
def load(name, kind):
    return load_lineage(name) if kind == "lineage" else load_atlas(name)
