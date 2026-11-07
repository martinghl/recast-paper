"""Raw integer counts of an atlas, in the cell order of its attribution file.

`SIGnature.utils.lognorm_counts` makes `layers["counts"]` an alias of `X` and log-normalizes in place, so the
`layers["counts"]` of `atlases/<stem>_attribution_scimilarity.h5ad` holds log-normalized values, not counts.
The counts are read here from `<stem>_fullgene.h5ad`: rows reordered to the attribution file's `obs_names`, genes
re-indexed onto SCimilarity's 28,231-gene order with `_align_to_go` (a sparse scatter, no normalization), and the
result checked to be integer.  `att.X` (the log-normalized input of the encoder) is correct as stored.

Usage:
    import sys; sys.path.insert(0, "/path/to/research/analysis")
    import real_counts
    cnt = real_counts.load("zhao", obs_names)      # csr, cells x 28231, integer"""
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

SB = "/path/to/research/analysis"
ATLASES = f"{SB}/atlases"
for _p in (SB, f"{SB}/ans_replication"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from scoring_benchmark.datasets import _align_to_go, _go_gene_order  # noqa: E402

_CACHE = {}


def load(stem, obs_names, cache=True):
    """cells x 28,231 integer counts for `stem`, rows ordered to match `obs_names`.

    obs_names: the attribution h5ad's obs_names (or any subset of them, in the order you want).
    Raises if a barcode is missing from the fullgene file or if the aligned matrix is not integer --
    both mean the wrong source matrix was picked up, which is exactly the failure this module exists
    to make impossible to miss.
    """
    want = [str(b) for b in obs_names]
    key = (stem, len(want), want[0] if want else "", want[-1] if want else "")
    if cache and key in _CACHE:
        return _CACHE[key]

    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pos = {str(b): i for i, b in enumerate(fg.obs_names)}
    missing = [b for b in want if b not in pos]
    if missing:
        raise RuntimeError(f"{stem}: {len(missing)} barcodes absent from {stem}_fullgene.h5ad "
                           f"(first: {missing[:3]})")
    X = sp.csr_matrix(fg.X)[[pos[b] for b in want]]
    rawvar = np.asarray(fg.var_names).astype(str)
    del fg

    C = sp.csr_matrix(_align_to_go(X, rawvar, _go_gene_order()))
    C.eliminate_zeros()
    d = C.data
    if not (C.nnz and np.all(d == np.round(d))):
        raise RuntimeError(f"{stem}: aligned matrix is not integer counts "
                           f"(nnz={C.nnz}, min={d.min() if C.nnz else 'NA'}) -- wrong source matrix")
    if cache:
        _CACHE.clear()          # one dataset resident at a time; these are ~100k x 28k
        _CACHE[key] = C
    return C


def align_to(C, genes, target_genes):
    """Re-index a counts matrix onto another encoder's gene space, zero-filling genes we lack.

    `load()` handles the aliasing in the stored files; this handles it in the encoder wrappers.
    `SSLWrapper.preprocess_adata` (and scFoundation's)
    call `align_dataset` then `lognorm_counts`, and `lognorm_counts` does

        data.X = data.layers["counts"]        # an ALIAS, not a copy
        sc.pp.normalize_total(data); sc.pp.log1p(data)

    so after `preprocess_adata` the returned object's `layers["counts"]` IS its lognorm `X`. A branch that
    reads `adh.layers["counts"]` to build an IG centroid therefore double-normalises -- even when the
    counts handed *in* were real. The tell in the cross-encoder arm was that the EXPRESSION baseline moved
    between runs, a quantity with no centroid in it at all.

    Gene handling mirrors `align_dataset`: reorder onto `target_genes`, zeros where the FM names a gene the
    input does not carry. Cells are untouched, so the result stays row-aligned with the preprocessed object.
    """
    pos = {g: i for i, g in enumerate(np.asarray(genes).astype(str))}
    tgt = np.asarray(target_genes).astype(str)
    idx = np.fromiter((pos.get(g, -1) for g in tgt), dtype=np.int64, count=len(tgt))
    Cs = sp.csr_matrix(C)
    pad = sp.csr_matrix((Cs.shape[0], 1), dtype=Cs.dtype)          # column of zeros for absent genes
    out = sp.hstack([Cs, pad], format="csc")[:, np.where(idx >= 0, idx, Cs.shape[1])].tocsr()
    out.eliminate_zeros()
    if out.nnz and not np.all(out.data == np.round(out.data)):
        raise RuntimeError("align_to: source matrix was not integer counts")
    return out


def summary(C):
    """One-line sanity string, for logging next to the load."""
    tot = np.asarray(C.sum(1)).ravel()
    return (f"{C.shape[0]}x{C.shape[1]} integer counts, max={C.data.max():.0f}, "
            f"median per-cell total={np.median(tot):.0f}")
