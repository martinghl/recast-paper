"""Data of Supplementary Fig. 4c,d: the cell-cycle fraction of the selected genes.

35 phase-defined comparisons (33 bone-marrow identity-phase comparisons, the Zhao antibody-secreting cells and a pooled
control) x 52 method configurations x 20, 50, 100 or 200 requested genes (data/cell_cycle_scores_all_methods.csv;
precision = hits / n_top, where n_top is the length of the list actually returned).
  f20, pe, rank_recast        per-comparison values at 20 genes, the mean per method, and RECAST's rank by that mean
  KS, recast_k, best_k, best_id, rand_k, rk
                              per number of requested genes: RECAST's mean, the highest mean of any configuration (an
                              envelope, not one method) and which configuration, the alphabetical control, RECAST's rank
The configuration called "random" returns genes in alphabetical order; it is drawn as "Alphabetical control".
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# display names of the methods
_SPECIAL = {"recast_M0": "RECAST", "cosg": "COSG", "logfc_raw": "log fold-change",
            "logfc_abs": "log fold-change (abs)", "lm_two_sample": "linear model", "glmgampoi": "glmGamPoi",
            "cepo": "Cepo", "random": "Random", "limma_trend": "limma (trend)", "limma_voom": "limma (voom)",
            "edger_glm": "edgeR (GLM)", "edger_ql": "edgeR (QL)"}


def pretty(m):
    if m in _SPECIAL: return _SPECIAL[m]
    s = (m.replace("scran_scoreMarkers_", "scran scoreMarkers ").replace("scran_findMarkers_", "scran findMarkers ")
         .replace("seurat_", "Seurat ").replace("scanpy_", "scanpy "))
    s = s.replace("_rankby_raw", "").replace("_rankby_abs", " (abs)")
    s = (s.replace("wilcoxontiecorrect", "Wilcoxon (tie-corr.)").replace("wilcoxon", "Wilcoxon")
         .replace("tover", "t (overest. var.)").replace("_logFC_cohen", " logFC Cohen")
         .replace("_logFC_detected", " logFC det.").replace("_AUC", " AUC").replace("_", " "))
    return s

ALL = pd.read_csv(f"{HERE}/data/cell_cycle_scores_all_methods.csv")
assert len(ALL) == 21840
cc = ALL[ALL["gt"] == "primary"]
assert len(cc) == 35 * 52 * 4
assert (np.abs(cc.precision - cc.hits / cc.n_top) < 1e-12).all() and (cc.n_top <= cc.K).all()
m8 = pd.read_csv(f"{HERE}/data/cell_cycle_scores.csv")
m8 = m8[(m8["gt"] == "primary") & (m8.K == 20) & (m8.method == "RECAST_M0")].set_index("contrast").precision
_rg = pd.read_csv(f"{HERE}/data/cell_cycle_genes_bone_marrow_fetal_asdc.csv")
_rg = _rg[_rg.method == "random"].sort_values("rank").gene.tolist()
assert _rg == sorted(_rg), "the 'random' configuration is expected to list genes alphabetically"

f20 = cc[cc.K == 20]
assert f20.contrast.nunique() == 35 and f20.method.nunique() == 52
_c = set(f20.contrast)                      # 35 = 33 bone-marrow identity-phase + Zhao ASC (original clusters) + pooled
assert len([c for c in _c if c.startswith("bmf_") and c != "bmf_global"]) == 33 and {"bmf_global", "zhao_asc_exact"} <= _c
assert not {"scmarkergene", "nsforest", "rankcorr", "scmags", "scgenefit"} & set(f20.method)   # no dedicated selectors
assert "seurat_roc" in set(f20.method)       # Seurat roc is part of these 52 configurations, unlike the 56-method benchmark
pe = f20.groupby("method").precision.mean().sort_values(ascending=False)
rank_recast = int((pe > pe["RECAST_M0"]).sum()) + 1
assert (round(pe["RECAST_M0"], 3), rank_recast, round(pe["cepo"], 3), round(pe["random"], 3)) == (0.696, 40, 0.863, 0.079)
fair = f20[f20.method == "RECAST_M0"].set_index("contrast").precision
CURES = [("GMP", "bm_gmp_cycle", "bmf_gmp_cycle"),
         ("Proliferating T", "bm_t_prolif", "bmf_t_proliferating"),
         ("Pre-pDC", "bm_prepdc_cycling", "bmf_pre_pdc")]
assert [round(m8[a], 2) for _, a, _ in CURES] == [0.15, 0.05, 0.05]
assert [round(fair[b], 2) for _, _, b in CURES] == [0.95, 0.85, 0.80]
assert round(fair["bmf_global"], 2) == 0.05 and "bmf_global" in set(f20.contrast)    # pooled control is one of the 35

KS = [20, 50, 100, 200]
recast_k, best_k, best_id, rand_k, rk = [], [], [], [], []
for K in KS:
    pk = cc[cc.K == K].groupby("method").precision.mean()
    assert pk.idxmax() != "RECAST_M0"
    recast_k.append(pk["RECAST_M0"]); best_k.append(pk.max()); best_id.append(pk.idxmax()); rand_k.append(pk["random"])
    rk.append(int((pk > pk["RECAST_M0"]).sum()) + 1)
assert rk == [40, 27, 24, 22]


def name(m):
    return "RECAST" if m == "RECAST_M0" else ("Alphabetical control" if m == "random" else pretty(m))
