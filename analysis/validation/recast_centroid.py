"""The centroid used by every validation script: the mean of per-cell log-normalized profiles,

    C_g = mean_i log1p(1e4 * x_ig / sum_g x_ig)          ("mean_lognorm"),

and, for comparison, pooled counts then log,

    C_g = log1p(1e4 * sum_i x_ig / sum_ig x_ig)          ("pseudobulk").

`centroid()` equals recast.centroid.mean_lognorm_centroid but stays sparse (the "global" reference is ~100k cells x
28k genes, ~11 GB dense).  log1p(0) = 0, so the zeros of a CSR matrix stay zeros and the sparse mean equals the dense
one.

    import sys; sys.path.insert(0, ".../validation")
    import recast_centroid as fc
    Cc = fc.centroid(counts, target_idx)
    Cr = fc.centroid(counts, ref_idx)"""
import numpy as np
import scipy.sparse as sp

PAPER_DEFAULT = "mean_lognorm"


def assert_counts(counts, who=""):
    """Fail loudly if `counts` is already lognorm rather than raw counts.

    Both recipes here normalize internally, so handing them a lognorm matrix double-normalizes and
    silently returns a plausible-looking profile. That is not hypothetical: several atlases
    `*_attribution_*.h5ad` files carry a `layers["counts"]` that SIGnature's lognorm_counts overwrote in
    place (hence `real_counts.load`).  Raw counts are non-negative integers, so one cheap check catches this.
    """
    X = counts
    d = (X.data[:20000] if sp.issparse(X) else np.asarray(X).ravel()[:200000])
    d = d[d != 0]
    if d.size and not np.allclose(d, np.round(d)):
        raise ValueError(
            f"{who or 'recast_centroid'}: expected RAW COUNTS, got non-integer values "
            f"(min {d.min():.4g}, max {d.max():.4g}) -- this looks lognorm. Both centroid recipes "
            f"normalize internally, so this would double-normalize. Load real counts (real_counts.load / "
            f"the *_fullgene.h5ad), not layers['counts'] of an attribution h5ad.")


def centroid(counts, idx=None, recipe=PAPER_DEFAULT):
    """Collapse a cell set into one denoised profile.

    counts : (cells, genes) RAW integer counts, sparse or dense.  Not lognorm -- passing an already
             normalized matrix silently double-normalizes (the bug class this module exists to end).
    idx    : integer or boolean index of the cell set; None = all cells.
    recipe : "mean_lognorm" (paper default) or "pseudobulk" (pool-before-log, for sensitivity only).
    """
    assert_counts(counts, "centroid()")
    X = counts if idx is None else counts[idx]
    if recipe == "pseudobulk":
        tot = np.asarray(X.sum(axis=0)).ravel().astype("float64")
        s = tot.sum()
        return np.log1p(1e4 * tot / max(s, 1.0)).astype("float32")
    if recipe != "mean_lognorm":
        raise ValueError(f"unknown centroid recipe {recipe!r}")
    if sp.issparse(X):
        X = X.tocsr().astype("float64")
        tot = np.asarray(X.sum(axis=1)).ravel()
        tot[tot == 0] = 1.0                                  # empty cell -> all-zero lognorm row
        M = X.multiply((1e4 / tot)[:, None]).tocsr()
        M.data = np.log1p(M.data)                            # log1p(0)=0: the zeros stay zeros
        return np.asarray(M.mean(axis=0)).ravel().astype("float32")
    X = np.asarray(X, dtype="float64")
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return np.log1p(1e4 * X / tot).mean(axis=0).astype("float32")


def lognorm_matrix(counts):
    """Per-cell tp10k-lognorm of the whole count matrix, once, as CSR.

    mean_lognorm normalizes EVERY cell before pooling, so unlike pseudobulk it cannot be computed
    from a column sum -- a naive `centroid()` per call re-normalizes the whole reference set each
    time.  That is invisible for a one-shot script and crippling for reference_choice.py, which draws 40
    bootstrap references per (target, reference) pair over a 60k-cell dataset.  Transform once here,
    then `Centroider` reduces each call to a row-slice mean.
    """
    X = (counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)).astype("float64")
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    M = X.multiply((1e4 / tot)[:, None]).tocsr()
    M.data = np.log1p(M.data)
    return M


class Centroider:
    """Precomputed mean_lognorm centroids: `C = Centroider(counts); C(idx)`.

    Numerically identical to `centroid(counts, idx)` -- the
    per-cell transform is the same, only its position in the loop changes.
    """

    def __init__(self, counts):
        assert_counts(counts, "Centroider()")
        self.Ln = lognorm_matrix(counts)

    def __call__(self, idx=None):
        X = self.Ln if idx is None else self.Ln[idx]
        return np.asarray(X.mean(axis=0)).ravel().astype("float32")
