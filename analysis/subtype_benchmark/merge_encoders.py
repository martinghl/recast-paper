"""Merge the per-encoder tables of the cross-encoder comparison (Supplementary Fig. 2e) into one table per metric, read
by validation/encoders/encoder_summary.py.

encoders_recast.py (SCimilarity and the two SSL encoders), encoders_scvi.py and encoders_scfoundation.py each write
their own recall / subtype-kNN table; this script concatenates them in a fixed encoder and lineage order:

    subtype_benchmark/encoder_results/recast_cross_fm_recall_ALL_multi<tag>.csv   encoder,lineage,k,RECAST,expression,n_states,delta
    subtype_benchmark/results/fm_finestate_knn_ALL_multi<tag>.csv                 FM,lineage,dim,state_knn_acc,state_macroF1

Usage: python merge_encoders.py [tag]   (default _v3).  Env SC.  Reads CSVs only.
"""
import sys

import pandas as pd

SB = "/path/to/research/analysis"
FSB = f"{SB}/subtype_benchmark/results"
ENC_RESULTS = f"{SB}/subtype_benchmark/encoder_results"

RECALL_SRC = ["recast_cross_fm_recall_multi", "encoders_scvi", "encoders_scfoundation"]
KNN_SRC = ["fm_finestate_knn_multi", "recast_scvi_finestate_knn_multi", "recast_scf_finestate_knn_multi"]
ENC_ORDER = ["SCimilarity", "scTab_SSL", "PBMC_SSL", "scVI", "scFoundation"]
LIN_ORDER = ["zhao_CD8", "dom_helperT", "dom_cytoT", "zhao_B", "pbmcbmn_Bmat", "yoshida_Bsw"]

# scVI takes raw counts and log-transforms internally, so encoders_scvi.py builds its IG endpoints as mean counts
# per cell and has no centroid choice; its tables carry the _v2 tag and are read for the _v3 merge as well.
PINNED = {"_v3": {"encoders_scvi": "_v2", "recast_scvi_finestate_knn_multi": "_v2"}}


def _src(stem, tag):
    return f"{FSB}/{stem}{PINNED.get(tag, {}).get(stem, tag)}.csv"


def _ordered(df, enc_col):
    df = df.copy()
    df["_e"] = df[enc_col].map({e: i for i, e in enumerate(ENC_ORDER)})
    df["_l"] = df.lineage.map({l: i for i, l in enumerate(LIN_ORDER)})
    return df.sort_values(["_l", "_e"]).drop(columns=["_e", "_l"]).reset_index(drop=True)


def build(tag):
    rec = pd.concat([pd.read_csv(_src(s, tag)) for s in RECALL_SRC], ignore_index=True)
    rec = rec[rec.k == 20].copy()
    rec["delta"] = rec.RECAST - rec.expression
    rec = _ordered(rec[["encoder", "lineage", "k", "RECAST", "expression", "n_states", "delta"]], "encoder")

    knn = pd.concat([pd.read_csv(_src(s, tag)) for s in KNN_SRC], ignore_index=True)
    knn = _ordered(knn[["FM", "lineage", "dim", "state_knn_acc", "state_macroF1"]], "FM")
    return rec, knn


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_v3"
    rec, knn = build(tag)
    rp, kp = f"{ENC_RESULTS}/recast_cross_fm_recall_ALL_multi{tag}.csv", f"{FSB}/fm_finestate_knn_ALL_multi{tag}.csv"
    rec.to_csv(rp, index=False); knn.to_csv(kp, index=False)
    print(f"wrote {rp}\nwrote {kp}")


if __name__ == "__main__":
    main()
