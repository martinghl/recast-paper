"""SCimilarity Integrated-Gradients attributions — the SIGnature definition.

We reuse SIGnature's own ``SCimilarityWrapper`` so attribution values match the paper
exactly: IG with ``multiply_by_inputs=True`` (folds expression into the attribution),
zero baseline, ``|.|``, then each cell's attributions row-normalized to sum ``1000``
(``SIGnature.utils.normalize_attribution_matrix``).

Contract: the input AnnData is ALREADY aligned to the SCimilarity gene order
(``var_names == wrapper.gene_order``, 28,231 genes) and log-normalized (``X`` = lognorm,
per ``data.harmonize``). Feeding the SAME ``adata.X`` here and to the DE / neutral
evaluator guarantees attributions and expression are computed on identical cells & genes.

Validated: 64 cells → (64, 28231) in ~0.6 s on an H200; row sums == 1000; finite.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Single source of truth for the model path is config.MODEL_DIR (fallback keeps this module
# importable even when `config` is not on sys.path). On-disk SCimilarity release 1.1.
try:
    from config import MODEL_DIR as DEFAULT_MODEL_DIR
except Exception:  # pragma: no cover
    DEFAULT_MODEL_DIR = "/path/to/scimilarity/model_v1.1"

_WRAPPER_CACHE: dict = {}


def get_wrapper(model_dir: str = DEFAULT_MODEL_DIR, use_gpu: bool = True):
    """Load (and cache) the SCimilarity encoder wrapper used for attributions."""
    key = (model_dir, use_gpu)
    if key not in _WRAPPER_CACHE:
        from SIGnature.models.scimilarity import SCimilarityWrapper

        _WRAPPER_CACHE[key] = SCimilarityWrapper(model_dir, use_gpu=use_gpu)
    return _WRAPPER_CACHE[key]


def gene_order(model_dir: str = DEFAULT_MODEL_DIR) -> list:
    """The frozen SCimilarity input gene space (list of symbols, len 28,231)."""
    return list(get_wrapper(model_dir).gene_order)


def compute_attributions(
    adata_lognorm,
    model_dir: str = DEFAULT_MODEL_DIR,
    batch_size: int = 500,
    use_gpu: bool = True,
) -> "sp.csr_matrix":
    """IG attributions for an aligned, log-normalized AnnData.

    Parameters
    ----------
    adata_lognorm : AnnData
        ``X`` = log-normalized expression, ``var_names`` == the SCimilarity gene order.
    batch_size : int
        Cells per IG batch. Keep <= ~1000 (the host's NVML quirk can crash very large
        single GPU allocations).

    Returns
    -------
    scipy.sparse.csr_matrix
        (n_cells, 28231) attribution matrix; each row sums to ~1000; non-negative.
    """
    w = get_wrapper(model_dir, use_gpu)
    if list(adata_lognorm.var_names) != list(w.gene_order):
        raise ValueError(
            "adata_lognorm must be aligned to the SCimilarity gene_order "
            f"(got {adata_lognorm.n_vars} vars; expected {len(w.gene_order)} in model order). "
            "Run data.harmonize() first."
        )
    X = adata_lognorm.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    attr = w.calculate_attributions(
        X, method="ig", batch_size=batch_size, multiply_by_inputs=True,
        target_sum=1e3, disable_tqdm=True,
    )
    return attr.tocsr()


def mean_attribution_per_gene(attr_csr: "sp.csr_matrix") -> np.ndarray:
    """Mean attribution per gene across cells → 1-D array of length n_genes."""
    return np.asarray(attr_csr.mean(axis=0)).ravel()
