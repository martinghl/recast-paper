"""The eight per-cell gene-set scoring methods of ANS (Adjusted Neighborhood Scoring), run through
`signaturescoring.score_signature` with the hyperparameters of the ANS benchmark (its constants.py SCORING_METHODS).

Extends `scoring_benchmark.baselines` (five methods with default parameters) to the full eight, reusing its
AnyRandom shim and its `_present_genes` / `_read_and_clear` / `_TMP_COL` helpers.

    display     ssc method                       params
    ANS         adjusted_neighborhood_scoring    ctrl_size=100
    Seurat      seurat_scoring                   ctrl_size=100, n_bins=25
    Seurat_AG   seurat_ag_scoring                n_bins=25
    Seurat_LVG  seurat_lvg_scoring               ctrl_size=100, n_bins=25,
                                                 lvg_computation_version='v1', lvg_computation_method='seurat'
    Scanpy      scanpy_scoring                   ctrl_size=100, n_bins=25
    Jasmine_LH  jasmine_scoring                  score_method='likelihood'
    Jasmine_OR  jasmine_scoring                  score_method='oddsratio'
    UCell       ucell_scoring                    maxRank=1500

"Scanpy" is `signaturescoring.scanpy_scoring` (the ANS reimplementation), not `scanpy.tl.score_genes`.
`use_raw=False` on every call; `joblib_kwargs={'n_jobs': 1}` for UCell and both JASMINE variants (the AnyRandom
shim does not reach loky subprocesses)."""
# Importing scoring_benchmark.baselines runs install_anyrandom_shim() at import time (needed before
# `import signaturescoring`). Also pull the low-level helpers so behaviour matches the shared module.
from scoring_benchmark.baselines import _present_genes, _read_and_clear, _TMP_COL  # noqa: F401

import numpy as np
import signaturescoring as ssc

# (display name, ssc method, params dict, run_serial) -- run_serial forces n_jobs=1 (UCell/JASMINE).
ANS_SCORING_METHODS = [
    ("ANS",        "adjusted_neighborhood_scoring", {"ctrl_size": 100}, False),
    ("Seurat",     "seurat_scoring",                {"ctrl_size": 100, "n_bins": 25}, False),
    ("Seurat_AG",  "seurat_ag_scoring",             {"n_bins": 25}, False),
    ("Seurat_LVG", "seurat_lvg_scoring",            {"ctrl_size": 100, "n_bins": 25,
                                                     "lvg_computation_version": "v1",
                                                     "lvg_computation_method": "seurat"}, False),
    ("Scanpy",     "scanpy_scoring",                {"ctrl_size": 100, "n_bins": 25}, False),
    ("Jasmine_LH", "jasmine_scoring",               {"score_method": "likelihood"}, True),
    ("Jasmine_OR", "jasmine_scoring",               {"score_method": "oddsratio"}, True),
    ("UCell",      "ucell_scoring",                 {"maxRank": 1500}, True),
]
ANS_METHOD_NAMES = [m[0] for m in ANS_SCORING_METHODS]
_BY_NAME = {m[0]: m for m in ANS_SCORING_METHODS}


def score_cells(display_name, adata_log, gene_set):
    """Per-cell scores for `gene_set` under one of the 8 ANS methods (`display_name` in
    ANS_METHOD_NAMES). `adata_log.X` must hold log-normalized expression. Returns an (n_cells,)
    float64 array. Does not permanently mutate `adata_log.obs` (temp col deleted before return).
    """
    if display_name not in _BY_NAME:
        raise ValueError(f"Unknown method {display_name!r}; choose from {ANS_METHOD_NAMES}")
    _, method, params, run_serial = _BY_NAME[display_name]
    genes = _present_genes(adata_log, gene_set)
    kw = dict(params)
    if run_serial:
        kw["joblib_kwargs"] = {"n_jobs": 1}
    try:
        ssc.score_signature(
            adata_log,
            gene_list=genes,
            method=method,
            score_name=_TMP_COL,
            use_raw=False,
            **kw,
        )
        return _read_and_clear(adata_log, _TMP_COL)
    finally:
        if _TMP_COL in adata_log.obs.columns:
            del adata_log.obs[_TMP_COL]


def score_all_methods(adata_log, signatures, verbose=True):
    """Score every (method x signature) pair, ANS-faithful per-cell.

    signatures: {signature_name: [genes]}.
    Returns {method: DataFrame(cells x signatures)} -- each column named by the signature, index ==
    adata_log.obs_names. A per-(method,signature) failure (e.g. ANS dropping every signature gene as
    an expression extreme) is caught, logged, and that column filled with NaN so one bad pair does
    not abort the whole panel; harness argmax already routes NaN to always-lose.
    """
    import pandas as pd
    out = {}
    for method in ANS_METHOD_NAMES:
        cols = {}
        for sig_name, genes in signatures.items():
            try:
                cols[sig_name] = score_cells(method, adata_log, genes)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  [WARN] {method} x {sig_name} failed: {type(e).__name__}: {e}",
                          flush=True)
                cols[sig_name] = np.full(adata_log.n_obs, np.nan, dtype="float64")
        df = pd.DataFrame(cols, index=adata_log.obs_names)
        out[method] = df
        if verbose:
            print(f"  scored {method}: {df.shape}", flush=True)
    return out
