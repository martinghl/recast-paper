"""RECAST against the gene-set scoring methods on one fine-subtype dataset (GPU, SCimilarity), with
ANS's protocol (per-cell label assignment, balanced accuracy, scale imbalance).

Run as a module from /path/to/research/analysis (not with /path/to/research on sys.path, which would shadow the
installed `recast` package):

    cd /path/to/research/analysis
    CUDA_VISIBLE_DEVICES=<free gpu> /path/to/envs/SC/bin/python \
        -m scoring_benchmark.run_scoring_benchmark pbmcbmn_l2

Writes results/scoring_{key}.csv and results/confusion_{key}_{method}.csv.

RECAST scores from a_raw (raw counts on SCimilarity's gene order, datasets.py); the other methods from a_log
(log-normalized values, per cell).  `_PerCellLognormEncoder` below log-normalizes the cells that `.embed()` receives."""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

import recast
from recast import Encoder

from scoring_benchmark.datasets import SCIMILARITY_MODEL, load_scoring_dataset
from scoring_benchmark import baselines as B
from scoring_benchmark import harness as H

OUT = "/path/to/research/analysis/scoring_benchmark/results"


class _PerCellLognormEncoder(Encoder):
    """Wraps a real recast.SCimilarityEncoder so `.embed()` accepts RAW counts that are already gene-
    space aligned to SCimilarity's go order (as datasets.py's a_raw now is) instead of requiring
    pre-lognormed input.

    `scimilarity.CellEmbedding.get_embeddings`'s own docstring says its input must be "Gene space
    aligned and log normalized (tp10k) gene expression matrix" -- it does NOT lognorm internally, it
    just forwards whatever it is given straight into the trained network. RECAST's own `attribute()`
    calls `enc.embed(counts[mask])` with `counts = adata.X` UNCHANGED, so whatever adata we pass in
    must already be lognormed for that call to be correct -- but `pseudobulk_centroid` (also called on
    the SAME `counts`) needs RAW counts (it does its own log1p(1e4*proportion) internally). RECAST ties
    both operations to one array; recast_subtypes.py does not -- it embeds a separately built log-normalized
    layer (Xln) while summing raw counts (Craw) with population_ig. This wrapper reconciles the two by applying that same per-cell tp10k lognorm
    (log1p(1e4*proportion), identical to scimilarity.utils.lognorm_counts's
    sc.pp.normalize_total(target_sum=1e4)+log1p) to whichever raw-count slice `.embed()` receives,
    immediately before handing it to the real encoder.

    Re-indexing a_raw's raw counts onto the gene order (datasets.py's `_align_to_go`) and applying this
    log-normalization reproduces the stored `{stem}_attribution_scimilarity.h5ad` X to float32 rounding
    (largest difference 4.8e-7, Pearson r = 1.0, 30 cells of the Hao PBMC subset).

    `torch_encode`/`centroid_module` (used for the IG attribution step itself, over
    `pseudobulk_centroid`'s output) are delegated to the real encoder UNCHANGED: once a_raw is go-
    aligned, `pseudobulk_centroid`'s log1p(1e4*proportion) output is already the right shape (28231,
    go order) and the right scale (tp10k-style lognorm of an aggregate) for the network -- confirmed by
    reading recast/attribute.py's `_attribute_one`, which feeds that output straight into
    `centroid_module()` with no further preprocessing. Only the per-cell `.embed()` path needs help.

    A side effect of the lognorm step (which densifies) is also what avoids a real, independent bug:
    the installed `scimilarity` package's `get_embeddings` does
    `isinstance(X.data, zarr.core.Array)` on its sparse-input fast path, but `zarr.core.Array` does not
    exist in the installed zarr (3.1.6) -- ANY sparse input crashes with
    `AttributeError: module 'zarr.core' has no attribute 'Array'`, independent of gene-space shape.
    """

    def __init__(self, base_encoder):
        self._base = base_encoder

    def embed(self, counts):
        X = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
        X = X.astype("float32")
        totals = X.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        lognorm = np.log1p(1e4 * X / totals).astype("float32")
        return self._base.embed(lognorm)

    def torch_encode(self, x):
        return self._base.torch_encode(x)


def build_encoder(model_path=SCIMILARITY_MODEL):
    """Real SCimilarityEncoder wrapped for a_raw's go-aligned raw-count input (see
    `_PerCellLognormEncoder`) -- call once per process, reuse. Pins the SCimilarity model CPU-resident
    via device="cpu" (recast.SCimilarityEncoder, as of recast commit b13b6c7, defaults to CUDA-if-available
    when device= is omitted -- see `_recast_matrices_all`'s docstring for why every call site below also
    passes device="cpu" explicitly to the attribution call itself)."""
    return _PerCellLognormEncoder(recast.SCimilarityEncoder(model_path, device="cpu"))


def _recast_matrices_all(enc, a_raw, signatures):
    """One RECAST attribution pass (cluster_attribution computed once internally by
    score_gene_set_panel, reused across every signature x composite-variant pair) -> {variant:
    {signature: {cluster: score_mean}}}. score_mean is used as the cross-signature-comparable matrix
    value (mean, not sum, so signatures of different marker-set sizes aren't trivially favored).

    device="cpu": as of recast commit b13b6c7, `recast.SCimilarityEncoder.__init__` accepts a `device=`
    kwarg that defaults to CUDA-if-available when omitted, moving the underlying network (and
    `scimilarity.CellEmbedding`'s own `use_gpu` flag) there -- so an encoder built with no explicit
    device would be GPU-resident on this box. `build_encoder()` above pins `device="cpu"` at
    construction to keep the network CPU-resident, matching the `device="cpu"` passed to the
    attribution call below: recast.attribute()'s own device resolution ALSO defaults to "cuda" whenever
    torch.cuda.is_available(), so if either side (encoder construction or the attribution call) were
    left on its GPU-if-available default while the other was pinned to CPU, the IG step's tensors and
    the model would land on different devices -- reproduced live: `RuntimeError: Expected all tensors to
    be on the same device, but found at least two devices, cpu and cuda:0!`. Passing device="cpu" at
    BOTH sites (RECAST's own public, supported kwarg -- no recast package change needed) is what keeps
    everything on CPU. This is cheap regardless: the IG step attributes a single per-cluster pseudobulk
    vector (not per-cell), and enc.embed() runs wherever the encoder's network lives (CPU here, via the
    same pinning), so nothing here is actually GPU-accelerated in the current recast+scimilarity version
    pairing -- CPU is not a slowdown from some faster GPU path, it is the pinned, reproducible path the
    committed results were measured on.
    """
    df = recast.score_gene_set_panel(enc, a_raw, "state", signatures, reference="rest",
                                     composites=(None, "tauE_discrRU"), device="cpu")
    out = {}
    for variant, sub_v in df.groupby("variant"):
        M = {}
        for sig, sub in sub_v.groupby("signature"):
            M[sig] = {r.cluster: float(r.score_mean) for r in sub.itertuples()}
        out[variant] = M
    return out


def run(key, recast_variants=("bare", "tauE_discrRU")):
    os.makedirs(OUT, exist_ok=True)
    a_raw, a_log, labels, clusters, signatures, truth = load_scoring_dataset(key)
    print(f"[{key}] {len(labels)} cells, clusters={clusters}, "
          f"methods={B.available_methods()}, MISSING={B.MISSING_BASELINES}", flush=True)

    rows = []
    enc = build_encoder()
    recast_mats = _recast_matrices_all(enc, a_raw, signatures)
    for v in recast_variants:
        M = recast_mats[v]
        pred = H.annotate(M, list(signatures), clusters)
        rows.append(dict(dataset=key, method=f"RECAST_{v}",
                          balanced_acc=H.balanced_accuracy(pred, truth, clusters),
                          scale_imbalance=H.scale_imbalance(M, truth, clusters)))
        H.confusion(pred, truth, list(signatures)).to_csv(f"{OUT}/confusion_{key}_RECAST_{v}.csv")

    for m in B.available_methods():
        M = {sig: B.cluster_mean(B.score_cells(m, a_log, genes, list(a_log.var_names)), labels, clusters)
             for sig, genes in signatures.items()}
        pred = H.annotate(M, list(signatures), clusters)
        rows.append(dict(dataset=key, method=m,
                          balanced_acc=H.balanced_accuracy(pred, truth, clusters),
                          scale_imbalance=H.scale_imbalance(M, truth, clusters)))
        H.confusion(pred, truth, list(signatures)).to_csv(f"{OUT}/confusion_{key}_{m}.csv")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/scoring_{key}.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["pbmcbmn_l2", "yoshida", "helperT", "cytoT"]):
        run(k)
