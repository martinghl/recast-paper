"""Profiles and integrated gradients for populations of cells.

  C_c       target profile: p_{c,g} = N_{c,g} / sum_h N_{c,h} (pooled counts),  C_{c,g} = log1p(1e4 p_{c,g})
  C_negc    reference profile, built the same way from the reference cells (proportions averaged linearly over
            the reference populations, then log1p)
  u_c       normalize(zbar_c - zbar_negc), from the mean embeddings
  pbexpr           C_c - C_negc                        (expression difference, no encoder)
  pbcig_signed     IG(C_c; baseline C_negc; target <enc(x), u_c>)
  pbcig_positive   pbcig_signed where C_c - C_negc > 0  (RECAST's positive channel)"""
import sys, numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig
from captum.attr import IntegratedGradients
embed, CentroidTarget, enc, dev, gi, go = contrastive_ig.embed, contrastive_ig.CentroidTarget, contrastive_ig.enc, contrastive_ig.dev, contrastive_ig.gi, contrastive_ig.go
ig_of = {i: g for g, i in gi.items()}


def cluster_countsums(Craw, labels, types, rawvar):
    """sum raw counts per cluster (12127), map to model go space (28231). -> CS [K, 28231]."""
    S = np.vstack([np.asarray(Craw[labels == t].sum(0)).ravel() for t in types])          # [K,12127]
    df = pd.DataFrame(S.T, index=rawvar).groupby(level=0).sum().reindex(go).fillna(0.0)     # [28231,K]
    return df.values.T                                                                      # [K,28231]


def props_and_C(CS):
    tot = CS.sum(1, keepdims=True); tot[tot == 0] = 1
    P = CS / tot                                                                            # proportions [K,28231]
    return P.astype("float64"), np.log1p(1e4 * P).astype("float32")                         # P, C_c


def cluster_C(Craw, labels, types, rawvar, recipe=None):
    """Per-cluster representative profile C [K, 28231], under EITHER centroid recipe.

    "pseudobulk": pooled counts, then log (`props_and_C(cluster_countsums(...))`).
    "mean_lognorm": per-cell log-normalization, then the mean (the manuscript's centroid,
    validation/recast_centroid.py).

    recipe=None reads $RECAST_CENTROID and defaults to mean_lognorm; the experiments on groups of target cells
    (subtype_benchmark/aggregation_and_encoder.py) compare the two.

    mean_lognorm needs per-CELL counts on the go gene space -- the cluster count-sums have already
    thrown away the per-cell information -- so it aligns first (the same column scatter
    cluster_countsums' reindex performs) and then uses validation/recast_centroid.py.
    """
    import os
    recipe = recipe or os.environ.get("RECAST_CENTROID", "mean_lognorm")
    if recipe == "pseudobulk":
        return props_and_C(cluster_countsums(Craw, labels, types, rawvar))[1]
    if recipe != "mean_lognorm":
        raise ValueError(f"unknown centroid recipe {recipe!r}")
    sys.path.insert(0, "/path/to/research/analysis")
    sys.path.insert(0, "/path/to/research/analysis/validation")
    from scoring_benchmark.datasets import _align_to_go
    import recast_centroid as fc
    Cgo = _align_to_go(Craw, list(rawvar), list(go))          # cells x 28231, counts still
    return np.vstack([fc.centroid(Cgo, labels == t) for t in types]).astype("float32")


def neg_linear(P):
    """balanced negative in proportion space THEN log (the CORRECT construction). -> Cneg [K,28231], Pbal."""
    K = P.shape[0]; Pbal = (P.sum(0) - P) / (K - 1)                                          # mean of OTHER clusters' proportions
    return np.log1p(1e4 * Pbal).astype("float32"), Pbal


def neg_meanlog(C):
    """OLD (wrong) negative: average the per-cluster LOG pseudobulks. kept only for the Q1 comparison."""
    K = C.shape[0]; return ((C.sum(0) - C) / (K - 1)).astype("float32")


def directions(Z, labels, types):
    """u_c from balanced mean-of-embedding. Z = per-cell embeddings (recompute wherever data changes)."""
    zc = np.vstack([Z[labels == t].mean(0) for t in types]); K = len(types)
    U = zc - (zc.sum(0) - zc) / (K - 1)
    return (U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)).astype("float32")


def pbcig_signed(C, Cneg, U):
    """IG(C_c; C_negc, u_c) for all K centroids at once. -> S [K,28231] signed."""
    mod = CentroidTarget(enc).to(dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
    inp = torch.tensor(C, dtype=torch.float32, device=dev).requires_grad_(True)
    base = torch.tensor(Cneg, dtype=torch.float32, device=dev); u = torch.tensor(U, dtype=torch.float32, device=dev)
    return ig.attribute(inp, baselines=base, target=0, additional_forward_args=(u,), n_steps=50).detach().cpu().numpy()


def positive_channel(S, dC):
    """keep pbcig score only where pseudobulk expr difference dC>0; push the rest to the bottom."""
    return np.where(dC > 0, S, -np.inf)


def _rank_metrics(score, dC, midx, G):
    finite = np.where(np.isneginf(score), np.nanmin(score[np.isfinite(score)]) - 1e6 if np.isfinite(score).any() else -1e9, score)
    order = np.argsort(-finite)
    rec = {k: len(set(order[:k]) & midx) / max(1, len(midx)) for k in (20, 50, 100)}
    absf = {k: float(np.mean(dC[order[:k]] < 0)) for k in (20, 50, 100)}      # AbsenceFraction@K
    y = np.zeros(G); y[list(midx)] = 1
    ap = average_precision_score(y, finite) if midx else np.nan
    return rec, absf, ap, order


def per_class_table(dataset, method_scores, dC_by_cluster, markers, types, cell_stats):
    """method_scores: {method: [K,G] score}. -> long DataFrame (one row per class x method)."""
    G = next(iter(method_scores.values())).shape[1]; rows = []
    for i, t in enumerate(types):
        midx = {gi[g] for g in markers.get(t, []) if g in gi}
        for m, sc in method_scores.items():
            rec, absf, ap, _ = _rank_metrics(sc[i], dC_by_cluster[i], midx, G)
            rows.append(dict(dataset=dataset, cell_type=t, method=m, n_markers=len(midx),
                             n_cells=cell_stats[t]["n"], total_umi=cell_stats[t]["umi"], median_umi=cell_stats[t]["med"],
                             rec20=round(rec[20], 3), rec50=round(rec[50], 3), rec100=round(rec[100], 3),
                             AP=round(ap, 3), absence20=round(absf[20], 3), absence50=round(absf[50], 3)))
    return pd.DataFrame(rows)


def macro(df):
    """class-macro means (NOT pooled). -> DataFrame per method."""
    g = df.groupby("method")
    return pd.DataFrame({"macro_rec20": g.rec20.mean(), "macro_rec50": g.rec50.mean(),
                         "macro_AP": g.AP.mean(), "macro_absence20": g.absence20.mean()}).round(3).reset_index()
