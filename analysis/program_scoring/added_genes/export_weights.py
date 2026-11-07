"""Per-fold RECAST and SIGnature gene weights of every task under both fold schemes (GPU), for scoring
the panel variants on the CPU server.

RECAST phi_c and SIGnature psi_c are vectors over all 28,231 genes, built from one fold's training cells only; C_R is
the mean of the fold's other training cells.  A panel variant changes only the gene index set, so any panel can be
scored from these files.

Fold schemes
  cell    StratifiedKFold(5, shuffle, 20220502)                  -- the folds of Fig. 6a (sensitivity)
  donor   StratifiedGroupKFold(5, shuffle, 20220502) by donor    -- primary, on tasks with >= 5 donors;
          tasks with < 5 donors or no donor unit use the cell scheme (recorded in fold_scheme)
Check (cell scheme, curated panels): RECAST z_train and SIGnature gated z_train reproduce the rows of
reuse_by_task (recast_*) and results_signature (z_train) within 2e-3.

Writes program_scoring/added_genes/weights/<task>__<scheme>.npz and weights/parity.csv.
  cd /path/to/research/analysis
  NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=0 /path/to/envs/SC/bin/python program_scoring/added_genes/export_weights.py"""
import json, os, sys, time
import numpy as np, pandas as pd, anndata as ad
SB = "/path/to/research/analysis"
sys.path.insert(0, f"{SB}/program_scoring")
import score_programs as B                      # noqa: E402  loads SCimilarity on the GPU
import reference_free_scoring as SN                # noqa: E402
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold  # noqa: E402

OUT = f"{SB}/program_scoring/added_genes/weights"
os.makedirs(OUT, exist_ok=True)
DM = f"{SB}/program_scoring/donor_maps"
RAW = {"baron_pancreas": (f"{SB}/dataset_files/baron_pancreas/baron_pancreas_raw.h5ad", "donor"),
       "pelka_crc_epi": (f"{SB}/dataset_files/pelka_crc_epi/pelka_crc_epi_raw.h5ad", "specimen")}
master = pd.read_csv(f"{SB}/paper/program_scoring_tables/data/reuse_by_task.csv").set_index("dataset")
sig = pd.read_csv(f"{SB}/program_scoring/results_signature/metrics_all_signature.csv")
sig = sig[(sig.method == "signature_native_gated") & (sig.calibration == "z_train")].set_index("dataset")


def selection(name):
    """Cell indices / names kept by load_dataset (same >=50-cell rule and RandomState(0) stratified cap)."""
    cfg = B.DATASETS[name]; a = ad.read_h5ad(cfg["h5ad"], backed="r")
    y = np.array([l.strip() for l in open(cfg["labels"])]); lmap = cfg.get("label_map", {})
    if lmap:
        y = np.array([lmap.get(v, v) for v in y])
    panels = json.load(open(cfg["sig_json"])) if cfg["regime"] == "distinct" else \
        {s: cfg["panels"][s] for s in cfg["states"] if s in cfg["panels"]}
    cand = sorted(c for c in panels if int((y == c).sum()) >= B.MIN_CELLS)
    idx = np.where(np.isin(y, cand))[0]; yk = y[idx]
    if idx.size > B.CAP:
        rng = np.random.RandomState(0); per = max(B.MIN_CELLS, B.CAP // len(cand))
        sel = [np.where(yk == c)[0] if (yk == c).sum() <= per else rng.choice(np.where(yk == c)[0], per, replace=False)
               for c in cand]
        idx = idx[np.sort(np.concatenate(sel))]
    return idx, np.array(a.obs_names, dtype=object)[idx], y[idx]


def donors_for(name, parent, idx, cells):
    if name in RAW:
        rp, col = RAW[name]; r = ad.read_h5ad(rp, backed="r"); return np.array(r.obs[col], dtype=object)[idx]
    f = os.path.join(DM, f"{parent}_donor.tsv")
    if os.path.exists(f):
        d = pd.read_csv(f, sep="\t"); m = dict(zip(d.cell.astype(str), d.donor.astype(str)))
        got = np.array([m.get(str(c)) for c in cells], dtype=object); assert all(g is not None for g in got)
        return got
    return None


def ztrain(S_tr, S_te):
    mu, sd = S_tr.mean(0, keepdims=True), S_tr.std(0, keepdims=True); sd[sd == 0] = 1.0
    return (S_te - mu) / sd


par_rows = []
for t in master.index:
    t0 = time.time()
    X, A, y, cand, sig_idx, sig_genes, regime = SN.load_dataset_with_attr(t)
    inp = ad.read_h5ad(f"{SB}/program_scoring/tasks/{t}.h5ad")
    assert (inp.obs["label"].astype(str).to_numpy() == y).all() and list(inp.uns["cand"]) == cand, t
    assert np.array_equal(inp.X.toarray().astype("float32"), X), f"{t}: X differs from the exported task"
    idx, cells, ysel = selection(t); assert (ysel == y).all(), t
    don = donors_for(t, master.loc[t, "parent"], idx, cells)
    nd = 0 if don is None else len(set(don))
    n, K, G = X.shape[0], len(cand), X.shape[1]
    schemes = {"cell": ("cell", list(StratifiedKFold(5, shuffle=True, random_state=B.CV_SEED).split(X, y)))}
    if nd >= 5:
        dsp = list(StratifiedGroupKFold(5, shuffle=True, random_state=B.CV_SEED).split(X, y, groups=don))
        ok = all(all((y[tr] == c).sum() > 0 for c in cand) for tr, _ in dsp)
        assert all(not set(don[tr]) & set(don[te]) for tr, te in dsp)
        schemes["donor"] = ("donor", dsp) if ok else ("cell_fallback_classmissing", schemes["cell"][1])
    else:
        schemes["donor"] = ("cell_fallback_" + ("nodonor" if don is None else f"{nd}donors"), schemes["cell"][1])
    for sch, (label, splits) in schemes.items():
        fold = np.full(n, -1, int)
        phi = np.zeros((5, K, G), "float32"); psi = np.zeros((5, K, G), "float32"); cref = np.zeros((5, K, G), "float32")
        zR = np.zeros((n, K)); zS = np.zeros((n, K))
        for f, (tr, te) in enumerate(splits):
            fold[te] = f
            rp = B.build_candidate_params(X[tr], y[tr], cand, sig_idx)
            sp_ = SN.reference_free_params(A[tr], X[tr], y[tr], cand)
            for k, c in enumerate(cand):
                phi[f, k], cref[f, k], psi[f, k] = rp[c]["phi"], rp[c]["C_ref"], sp_[c]["psi"]
                assert np.array_equal(rp[c]["C_ref"], sp_[c]["C_ref"])
            zR[te] = ztrain(B.m1_scores(X[tr], rp, cand, sig_idx), B.m1_scores(X[te], rp, cand, sig_idx))
            zS[te] = ztrain(SN.reference_free_scores(X[tr], sp_, cand, sig_idx), SN.reference_free_scores(X[te], sp_, cand, sig_idx))
        assert (fold >= 0).all()
        np.savez_compressed(f"{OUT}/{t}__{sch}.npz", fold=fold, phi=phi, psi=psi, cref=cref, cand=np.array(cand),
                            y=y, donor=np.array([] if don is None else don.astype(str)), scheme=label)
        rR, rS = B.evaluate("RECAST", zR, y, cand), B.evaluate("SIGnature", zS, y, cand)
        row = dict(dataset=t, scheme=sch, fold_scheme=label, n_donors=nd,
                   recast_ba=rR["balanced_acc"], recast_f1=rR["macro_f1"], recast_auc=rR["mean_auroc"],
                   sig_ba=rS["balanced_acc"], sig_f1=rS["macro_f1"], sig_auc=rS["mean_auroc"])
        if sch == "cell":
            row.update(d_recast=max(abs(rR["balanced_acc"] - master.loc[t, "recast_ba"]),
                                    abs(rR["macro_f1"] - master.loc[t, "recast_f1"]),
                                    abs(rR["mean_auroc"] - master.loc[t, "recast_auc"])),
                       d_sig=max(abs(rS["balanced_acc"] - sig.loc[t, "balanced_acc"]),
                                 abs(rS["macro_f1"] - sig.loc[t, "macro_f1"]),
                                 abs(rS["mean_auroc"] - sig.loc[t, "mean_auroc"])))
        par_rows.append(row)
        print(t, sch, label, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()
                              if k.startswith(("recast", "sig", "d_"))}, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(par_rows).to_csv(f"{OUT}/parity.csv", index=False)

P = pd.DataFrame(par_rows); c = P[P.scheme == "cell"]
print("PARITY max |d| RECAST", c.d_recast.max(), " SIGnature", c.d_sig.max())
assert c.d_recast.max() < 2e-3 and c.d_sig.max() < 2e-3, "parity gate failed"
print("parity gate passed")
