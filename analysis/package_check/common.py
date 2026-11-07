"""Shared helpers for the package checks: every RECAST computation in the manuscript recomputed with
the released package (`import recast`) and compared with the tables the analysis scripts wrote.

Nothing here computes an attribution. The helpers only load data the way the analysis scripts did
(raw integer counts from <stem>_fullgene.h5ad, reindexed onto the encoder's gene order), read the
original tables, and compare gene lists.

Paths come from the environment:
  RESEARCH_ROOT      root of the analysis tree: its analysis/ folder holds code/ and analysis/ of this repository merged, and the data folders
                     and the original tables
  SCIMILARITY_MODEL  SCimilarity v1.1 model directory
  PACKAGE_CHECK_OUT  where the checks write their tables
"""
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = os.environ["RESEARCH_ROOT"]
SB = os.path.join(ROOT, "analysis")
FSB = os.path.join(SB, "subtype_benchmark")
ATLASES = os.path.join(SB, "atlases")
FULL_BENCHMARK = os.path.join(FSB, "full_benchmark")
SCIM = os.environ["SCIMILARITY_MODEL"]
OUT = os.environ["PACKAGE_CHECK_OUT"]
os.makedirs(OUT, exist_ok=True)

# Analysis-tree modules a check may import: lineage registries (plain dicts), data loaders and the
# cell-replacement read-outs. None of them computes an attribution.
HELPERS = ("lineage_registry", "encoder_lineages", "added_datasets", "replacement_lib", "select_recast", "cluster_key",
           "scoring_benchmark")


def assert_package_only():
    """Raise unless the released package is the only attribution code loaded: `recast` must be version 0.8.1 or
    later and come from outside RESEARCH_ROOT's analysis code, and no module from under RESEARCH_ROOT may be loaded
    other than the helpers above and these checks."""
    import recast
    if tuple(int(x) for x in recast.__version__.split(".")[:3]) < (0, 8, 1):
        raise RuntimeError(f"recast {recast.__version__} is loaded; the checks need 0.8.1 or later")
    root = os.path.realpath(ROOT) + os.sep
    allowed = (os.path.dirname(os.path.realpath(recast.__file__)) + os.sep,
               os.path.dirname(os.path.realpath(__file__)) + os.sep)
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        f = os.path.realpath(f)
        if f.startswith(root) and not f.startswith(allowed) and name.split(".")[0] not in HELPERS:
            bad.append(name)
    if bad:
        raise RuntimeError(f"analysis code other than the listed helpers was imported: {sorted(bad)}")


def config(name, sub=FSB):
    """Import a lineage registry (plain dicts) from the analysis tree."""
    if sub not in sys.path:
        sys.path.insert(0, sub)
    import importlib
    return importlib.import_module(name)


def scimilarity_genes():
    with open(os.path.join(SCIM, "gene_order.tsv")) as fh:
        return [l.strip() for l in fh if l.strip()]


_ENC = {}


def scimilarity():
    import recast
    if "scim" not in _ENC:
        _ENC["scim"] = recast.SCimilarityEncoder(SCIM, device="cuda")
    return _ENC["scim"]


def align(X, rawvar, genes):
    """(cells x rawvar) counts reindexed onto `genes`, zero-filling genes the data lack; duplicated
    symbols are summed. The same column scatter the analysis scripts used."""
    pos = {g: i for i, g in enumerate(genes)}
    src = [j for j, g in enumerate(rawvar) if g in pos]
    dst = [pos[rawvar[j]] for j in src]
    M = sp.coo_matrix((np.ones(len(src), dtype="float32"), (src, dst)), shape=(len(rawvar), len(genes))).tocsr()
    return (sp.csr_matrix(X).astype("float32") @ M).tocsr()


def counts(p2, stem, obs_names, genes, integer=True):
    """Integer counts of the given cells (rows in that order) from <stem>_fullgene.h5ad on `genes`.
    integer=False accepts a non-integer expression matrix (Lawlor's Smart-seq estimates), which the
    analysis scripts and the package log-normalize in the same way."""
    fg = ad.read_h5ad(os.path.join(p2, f"{stem}_fullgene.h5ad"))
    pos = {str(b): i for i, b in enumerate(fg.obs_names)}
    rows = [pos[str(b)] for b in obs_names]
    C = align(sp.csr_matrix(fg.X)[rows], np.asarray(fg.var_names).astype(str), genes)
    C.eliminate_zeros()
    assert C.nnz and (not integer or np.all(C.data == np.round(C.data))), f"{stem}: not integer counts"
    return C


def attribution_obs(p2, stem):
    """Cell order and obs of the attribution object the labels were written against."""
    A = ad.read_h5ad(os.path.join(p2, f"{stem}_attribution_scimilarity.h5ad"), backed="r")
    obs = A.obs.copy()
    names = np.asarray(A.obs_names).astype(str)
    A.file.close()
    return names, obs


def read_labels(path):
    return np.array([l.strip() for l in open(path) if l.strip()])


def lineage_adata(p2, stem, labels, states, min_cells=50, genes=None):
    """AnnData of one lineage: the cells whose label is one of `states` (states with fewer than
    `min_cells` cells dropped), raw counts on `genes` (SCimilarity's order by default),
    obs['state']."""
    genes = scimilarity_genes() if genes is None else genes
    names, _ = attribution_obs(p2, stem)
    assert len(labels) == len(names), f"{stem}: {len(labels)} labels for {len(names)} cells"
    keep_states = sorted(s for s in states if (labels == s).sum() >= min_cells)
    keep = np.isin(labels, keep_states)
    a = ad.AnnData(counts(p2, stem, names[keep], genes))
    a.var_names = list(genes)
    a.obs_names = names[keep]
    a.obs["state"] = labels[keep].astype(str)
    return a, keep_states


def panel_lists(path):
    """{cluster: [genes in rank order]} from a (cluster, rank, gene) panel file."""
    P = pd.read_csv(path)
    return {c: list(g.sort_values("rank")["gene"].astype(str)) for c, g in P.groupby("cluster", sort=False)}


def compare_lists(pkg, stored):
    """Agreement of two ranked lists over the original depth."""
    k = len(stored)
    pkg = list(pkg)[:k]
    same = [a == b for a, b in zip(pkg, stored)]
    first = next((i + 1 for i, s in enumerate(same) if not s), None)
    out = dict(depth=k, identical=all(same) and len(pkg) == k, first_diff_rank=first)
    for K in (20, 50, 200):
        if K <= k:
            out[f"same_set_top{K}"] = set(pkg[:K]) == set(stored[:K])
            out[f"overlap_top{K}"] = len(set(pkg[:K]) & set(stored[:K])) / K
    return out


def write(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"[written] {path}", flush=True)
    return path
