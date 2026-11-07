"""Data of Supplementary Fig. 4a,b: marker recovery and within-atlas AUROC of broad populations.

  recall, npop, pooled  expert-panel marker recovery per atlas (62 populations in 7 atlases; CITE-seq has no expert
                        panel), and pooled over the 62 populations with equal weights
  auc                   within-atlas AUROC of the 20 selected genes scored as one signature, 8 atlases x the 52 methods
                        drawn (RECAST, Scanpy Wilcoxon and 50 others)
Tables: data/broad_marker_recall.csv, data/broad_specificity.csv (the methods drawn), data/broad_auroc_two_definitions.csv
(per-gene and signature AUROC).
"""
import os

import pandas as pd

from method_colours import INK, RECAST, DE, DE_L, DE_XL

HERE = os.path.dirname(os.path.abspath(__file__))

REC = pd.read_csv(f"{HERE}/data/broad_marker_recall.csv")
SPEC = pd.read_csv(f"{HERE}/data/broad_specificity.csv")
METH = ["RECAST", "COSG", "limma", "Wilcoxon"]                   # ids in the recovery table (RECAST = RECAST)
DISP = {"RECAST": "RECAST", "COSG": "COSG", "limma": "limma-trend", "Wilcoxon": "Scanpy Wilcoxon"}
CMAP = {"RECAST": RECAST, "COSG": DE, "limma": DE_L, "Wilcoxon": DE_XL}
EDGE = {"RECAST": RECAST, "COSG": DE, "limma": DE_L, "Wilcoxon": INK}
WIL = "scanpy_wilcoxon_rankby_raw"                              # Scanpy Wilcoxon in the AUROC table
# rows top to bottom: (AUROC-table id, recovery-table id or None, label)
ROWS = [("pbmc3k", "pbmc3k", "pbmc3k"), ("ss3_pbmc", "ss3_pbmc", "Smart-seq3 PBMC"), ("lawlor", "lawlor", "Lawlor"),
        ("dominguez_conde", "dominguez_conde", "Domínguez Conde"), ("ren", "ren", "Ren"),
        ("stephenson", "stephenson", "Stephenson"), ("yoshida", "yoshida", "Yoshida"), ("citeseq", None, "CITE-seq CBMC")]

assert set(REC.method) == set(METH) and REC.atlas.nunique() == 7
assert SPEC.groupby(["atlas", "method"]).size().max() == 1 and SPEC.atlas.nunique() == 8 and SPEC.method.nunique() == 52
assert SPEC.groupby("atlas").size().eq(52).all()                  # every atlas has all 52 methods
recall = REC.pivot(index="atlas", columns="method", values="mean_recall")
npop = REC.groupby("atlas").n_clusters.first()
assert npop.to_dict() == {"pbmc3k": 9, "ss3_pbmc": 8, "lawlor": 7, "dominguez_conde": 7, "ren": 9, "stephenson": 10,
                          "yoshida": 12} and int(npop.sum()) == 62
pooled = {m: float((REC[REC.method == m].mean_recall * REC[REC.method == m].n_clusters).sum() / 62) for m in METH}
assert tuple(round(pooled[m], 3) for m in METH) == (0.272, 0.324, 0.307, 0.180)
auc = SPEC.pivot(index="atlas", columns="method", values="auroc")
assert (round(auc["RECAST"].mean(), 2), round(auc[WIL].mean(), 2)) == (0.96, 0.86)        # mean single-gene AUROC
assert (auc.idxmax(1) == "RECAST").all()                            # RECAST highest in 8 of 8 atlases
OTHER = [m for m in auc.columns if m not in ("RECAST", WIL)]
assert len(OTHER) == 50

# The AUROC drawn in b is that of each population's signature (its 20 genes scored together).  data/broad_specificity.csv
# holds the mean single-gene AUROC; data/broad_auroc_two_definitions.csv has both definitions, for the same 52 methods.
R = pd.read_csv(f"{HERE}/data/broad_auroc_two_definitions.csv")
R = R.merge(SPEC[["atlas", "method"]], on=["atlas", "method"], how="inner")
assert R.groupby("atlas").size().eq(52).all() and R.atlas.nunique() == 8, R.groupby("atlas").size()
_pub = SPEC.set_index(["atlas", "method"]).auroc
_chk = R.set_index(["atlas", "method"])
_cmp = _chk.drop(index="RECAST", level="method")
assert (_cmp.auroc_per_gene - _pub.reindex(_cmp.index)).abs().max() < 1e-9    # the per-gene values agree
_sig = R.pivot(index="atlas", columns="method", values="auroc_signature")
assert set(_sig.columns) == set(auc.columns) and set(_sig.index) == set(auc.index)
auc = _sig[auc.columns].loc[auc.index]
