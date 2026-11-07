"""Ranking metrics of a gene list against a marker panel (Supplementary Data sheet 13).

metrics(ranked, target) -> recall@10, recall@20, recall@50, AP@20 and NDCG@20 with binary relevance, or None for an
empty panel.  Used by paper/ground_truth_tables/panel_construction.py.
"""
import numpy as np

KS = (10, 20, 50)


def metrics(ranked, target):
    out = {}
    if not target: return None
    for k in KS:
        out[f"recall@{k}"] = len(set(ranked[:k]) & target) / len(target)
    rel = np.array([1.0 if g in target else 0.0 for g in ranked[:20]]); denom = min(len(target), 20)
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    out["AP@20"] = float((prec * rel).sum() / denom)
    dcg = float((rel / np.log2(np.arange(2, len(rel) + 2))).sum()); idcg = float((1 / np.log2(np.arange(2, denom + 2))).sum())
    out["NDCG@20"] = dcg / idcg
    return out
