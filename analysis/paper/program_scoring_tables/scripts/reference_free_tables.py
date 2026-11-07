#!/usr/bin/env python3
"""Reference-free attribution (SIGnature) in the program-scoring and transfer comparisons (Fig. 6a, Supplementary
Fig. 9, Supplementary Data).

SIGnature's formulation is integrated gradients on the SCimilarity encoder without a reference contrast.  This script
adds it to the tables of the two experiments:

  per-cell scoring over the 28 tasks  (program_scoring/results_signature/metrics_all_signature.csv)
    -> reuse_by_task_reference_free.csv   reuse_by_task.csv plus the two SIGnature rows (gated, plain)
    -> reuse_reference_free_summary.csv   means over the 28 tasks and per regime
  cross-atlas transfer of selected panels
    -> transfer_reference_free.csv        the fine-subtype tracks (transfer/arm3_transfer_pairwise_signature_v3.csv)
                                          and the broad cross-atlas track (results/transferability_currency_v3.csv,
                                          method native_direct), each with the method's rank among the methods the
                                          panel draws
The method labels in the outputs (signature_native_gated, SIGnature_native, native_direct) are the ids of the stored
result tables."""
import os, sys
import numpy as np, pandas as pd

ROOT = "/path/to/research/analysis"
D6 = f"{ROOT}/paper/program_scoring_tables/data"
SIG = f"{ROOT}/program_scoring/results_signature/metrics_all_signature.csv"
DTR = f"{ROOT}/subtype_benchmark"
V = os.environ.get("RECAST_RESULT_TAG", "_v3")
METRICS = [("ba", "balanced_acc"), ("f1", "macro_f1"), ("auc", "mean_auroc")]


def percell():
    tab = pd.read_csv(f"{D6}/reuse_by_task.csv")
    s = pd.read_csv(SIG)
    s = s[s.calibration == "z_train"]
    got = set(s.dataset.unique())
    miss = sorted(set(tab.dataset) - got)
    g = s[s.method == "signature_native_gated"].set_index("dataset")
    p = s[s.method == "signature_native_plain"].set_index("dataset")
    df = tab.copy()
    for tag, col in METRICS:
        df[f"sig_gated_{tag}"] = [g.loc[d, col] if d in g.index else np.nan for d in df.dataset]
        df[f"sig_plain_{tag}"] = [p.loc[d, col] if d in p.index else np.nan for d in df.dataset]
    df.to_csv(f"{D6}/reuse_by_task_reference_free.csv", index=False)
    print(f"wrote reuse_by_task_reference_free.csv ({len(df)} tasks; SIGnature computed for {len(got)}, missing {miss})")
    rows = []
    for name, key in (("RECAST", "recast"), ("ANS", "ans"), ("SIGnature_native_gated", "sig_gated"),
                      ("SIGnature_native_plain", "sig_plain")):
        r = dict(method=name, n_tasks=int(df[f"{key}_ba"].notna().sum()))
        for tag, _ in METRICS:
            v = df[f"{key}_{tag}"]
            r[f"mean_{tag}"] = v.mean()
            r[f"fine_{tag}"] = v[df.reg == "fine"].mean()
            r[f"distinct_{tag}"] = v[df.reg == "distinct"].mean()
            if key != "recast":
                r[f"recast_higher_{tag}"] = int((df[f"recast_{tag}"] > v).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{D6}/reuse_reference_free_summary.csv", index=False)
    print(out.round(3).to_string(index=False))
    return df


def transfer():
    rows = []
    fp = f"{DTR}/transfer/arm3_transfer_pairwise_signature{V}.csv"
    if os.path.exists(fp):
        pairs = pd.read_csv(f"{DTR}/transfer/arm3_transfer_pairwise{V}.csv")
        pairs = pairs[pairs.method != "seurat_roc"]                    # not in the method set, as in the panel
        sig = pd.read_csv(fp)
        # the panel's method set = mean over pairs per method; SIGnature is ranked within that set
        for met in ("auroc", "sum_max", "ontarget", "knn"):
            agg = pairs[pairs.metric == met].groupby("method")["transfer"].mean()
            sv = float(sig[sig.metric == met]["transfer"].mean())
            field = pd.concat([agg, pd.Series({"SIGnature_native": sv})])
            rows.append(dict(track=f"fine_subtype_{met}", method="SIGnature_native", value=sv,
                             rank=int((field.values > sv).sum() + 1), n_field=int(field.size),
                             recast_value=float(agg["RECAST_M0"]),
                             recast_rank=int((field.values > agg["RECAST_M0"]).sum() + 1),
                             n_pairs=int(sig[sig.metric == met].shape[0]),
                             source=os.path.basename(fp)))
    else:
        print(f"{fp} not present -- fine-subtype tracks not written")
    # broad cross-atlas track: native_direct (SIGnature) is one of the methods of this track
    b = pd.read_csv(f"{DTR}/results/transferability_baselines.csv").groupby("method")["transfer_auc"].mean()
    c = pd.read_csv(f"{DTR}/results/transferability_currency{V}.csv").groupby("method")["transfer_auc"].mean()
    comb = pd.concat([b, c]); comb = comb[~comb.index.duplicated()].drop("seurat_roc", errors="ignore")
    # methods other than RECAST only: the five m0_* composites are alternative RECAST configurations
    _IS_RECAST_CFG = lambda m: m.startswith("m0_")
    comb = comb[[not _IS_RECAST_CFG(m) for m in comb.index]]
    sv = float(comb["native_direct"])
    rows.append(dict(track="broad_cross_atlas_auroc", method="native_direct (SIGnature native)", value=sv,
                     rank=int((comb.values > sv).sum() + 1), n_field=int(comb.size),
                     recast_value=float(comb["M0"]), recast_rank=int((comb.values > comb["M0"]).sum() + 1),
                     n_pairs=np.nan, source=f"transferability_currency{V}.csv"))
    out = pd.DataFrame(rows)
    out.to_csv(f"{D6}/transfer_reference_free.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    if os.path.exists(SIG):
        percell()
    else:
        print(f"{SIG} not present -- per-cell tables not written")
    transfer()
