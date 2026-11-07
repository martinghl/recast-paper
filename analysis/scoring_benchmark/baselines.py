"""Per-cell gene-set scoring methods, averaged per cluster: Seurat (AddModuleScore), ANS (adjusted
neighborhood scoring), UCell and JASMINE, from the `signaturescoring` package
(https://github.com/BoevaLab/signaturescoring).  `adata_log.X` must hold log-normalized expression.

`signaturescoring.score_signature(adata, gene_list, method=..., **kwarg)` dispatches to the method:
'adjusted_neighborhood_scoring' -> ANS; 'seurat_scoring', 'seurat_ag_scoring', 'seurat_lvg_scoring',
'scanpy_scoring' -> Seurat variants; 'jasmine_scoring' -> JASMINE; 'ucell_scoring' -> UCell.  Each `score_genes`
writes `adata.obs[score_name]` in place (copy=False); the wrappers here pass an explicit `score_name`, read the column
and delete it, so the caller's AnnData is left unchanged.  "seurat" calls `scanpy.tl.score_genes` directly.

Behaviour of the package to be aware of:
  1. UCell and JASMINE default to joblib_kwargs={'n_jobs': 4}; the loky workers re-import `signaturescoring`
     without the AnyRandom shim and fail (BrokenProcessPool).  Both are run with joblib_kwargs={'n_jobs': 1}.
  2. ANS's remove_genes_with_invalid_control_set=True drops signature genes in the top or bottom ctrl_size // 2
     genes by mean expression; if none remains, ANS raises "empty signature".  Only very small synthetic panels
     reach this.
  3. score_genes(..., use_raw=None) uses `adata.raw` whenever it is set; ANS would then score stale `.raw.X` and
     UCell fails.  Every call passes use_raw=False.
  4. JASMINE selects likelihood or odds ratio with score_method='likelihood' / 'oddsratio'.
  5. JASMINE's enrichment term uses expressed / not expressed (non-zero) counts per cell; on data without zeros the
     term is constant and its min-max scaling returns NaN for every cell (no error).

AUCell needs `decoupler`, which is not installed in the SC environment; it is listed in MISSING_BASELINES.
The AnyRandom shim runs once at import, before `import signaturescoring`."""
from scoring_benchmark._env_probe import install_anyrandom_shim

install_anyrandom_shim()

import numpy as np
import scanpy as sc
import signaturescoring

_TMP_COL = "_scoring_benchmark_tmp_score"


def _present_genes(adata_log, gene_set):
    var_names = set(map(str, adata_log.var_names))
    # dict.fromkeys (not set()) to dedupe while preserving order, in case gene_set has duplicates.
    return list(dict.fromkeys(g for g in map(str, gene_set) if g in var_names))


def _read_and_clear(adata_log, col):
    scores = np.asarray(adata_log.obs[col], dtype="float64")
    del adata_log.obs[col]
    return scores


def _seurat(adata_log, genes):
    sc.tl.score_genes(adata_log, genes, score_name=_TMP_COL, use_raw=False)
    return _read_and_clear(adata_log, _TMP_COL)


def _ans(adata_log, genes):
    # use_raw=False: see note 3 of the module docstring -- without it, ANS silently scores from stale .raw.X
    # whenever the caller's AnnData has .raw set.
    signaturescoring.score_signature(
        adata_log,
        gene_list=genes,
        method="adjusted_neighborhood_scoring",
        score_name=_TMP_COL,
        use_raw=False,
    )
    return _read_and_clear(adata_log, _TMP_COL)


def _ucell(adata_log, genes):
    # n_jobs=1: forced in-process to avoid the worker failure of note 1 of the module docstring.
    # use_raw=False: see note 3 of the module docstring -- without it, UCell crashes outright whenever the
    # caller's AnnData has .raw set (a Raw object has no .obs).
    signaturescoring.score_signature(
        adata_log,
        gene_list=genes,
        method="ucell_scoring",
        score_name=_TMP_COL,
        joblib_kwargs={"n_jobs": 1},
        use_raw=False,
    )
    return _read_and_clear(adata_log, _TMP_COL)


def _jasmine(score_method):
    """Build a score_cells-compatible callable bound to JASMINE's `score_method` selector.

    `score_method` in {"likelihood", "oddsratio"} -- the keyword and its two values are those of the installed package (note 4 of the module
    docstring).
    """

    def _f(adata_log, genes):
        # use_raw=False: see note 3 of the module docstring -- same reasoning as _ans/_ucell.
        # joblib_kwargs n_jobs=1: see note 4 of the module docstring -- JASMINE's default n_jobs=4 hits the same
        # AnyRandom-shim-not-in-subprocess failure as UCell (note 1).
        signaturescoring.score_signature(
            adata_log,
            gene_list=genes,
            method="jasmine_scoring",
            score_method=score_method,
            score_name=_TMP_COL,
            use_raw=False,
            joblib_kwargs={"n_jobs": 1},
        )
        return _read_and_clear(adata_log, _TMP_COL)

    return _f


_jasmine_lh = _jasmine("likelihood")
_jasmine_or = _jasmine("oddsratio")


_METHODS = {
    "seurat": _seurat,
    "ans": _ans,
    "ucell": _ucell,
    "jasmine_lh": _jasmine_lh,
    "jasmine_or": _jasmine_or,
}


def available_methods():
    """Return the working dispatch keys (derived from `_METHODS`, never a hardcoded/stale list)."""
    return list(_METHODS.keys())


# Baselines that could not be run, with the reason; run_scoring_benchmark.py prints this list with available_methods().
MISSING_BASELINES = ["aucell"]  # decoupler is not installed in the SC environment; signaturescoring has no AUCell


def score_cells(method, adata_log, gene_set, all_genes):
    """Per-cell scores for `gene_set` under `method` in `available_methods()`
    (currently {"seurat", "ans", "ucell", "jasmine_lh", "jasmine_or"}).

    `adata_log.X` must hold log-normalized expression. `gene_set` is filtered to genes present in
    `adata_log.var_names` before scoring. `all_genes` (a background gene universe) is accepted but not
    used: each method treats the full `adata_log.var_names` as its gene pool. Does not permanently mutate
    `adata_log.obs` (the temp score column is deleted before returning).

    Note: `jasmine_lh`/`jasmine_or` return NaN (not an error) for fully dense input with no true
    zeros (see the module docstring).

    Returns an `(n_cells,)` float64 array.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown scoring method {method!r}; choose from {sorted(_METHODS)}")
    genes = _present_genes(adata_log, gene_set)
    try:
        return _METHODS[method](adata_log, genes)
    finally:
        # Belt-and-suspenders: if the underlying call raised after writing the column but before our
        # own _read_and_clear ran, don't leave adata_log.obs permanently mutated.
        if _TMP_COL in adata_log.obs.columns:
            del adata_log.obs[_TMP_COL]


def cluster_mean(scores, labels, clusters):
    """Mean per-cell score within each cluster: {cluster -> float}."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(str)
    return {c: float(scores[labels == str(c)].mean()) for c in clusters}
