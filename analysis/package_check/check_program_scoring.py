"""Fig. 6: per-cell program scoring.

RECAST's part of every Fig. 6 panel is recast.CellScorer: on each fold's training cells, fit gives every cell
type c its attribution phi_c and reference profile C_R[c]; score then applies them to the held-out cells with
the curated panels, raw (Fig. 6b, c) or calibrated on the training cells' scores (calibrate="zscore"; Fig. 6a
and 6d). The original runs computed the same quantities with analysis code and exported, for the 28 tasks and
both fold schemes (cell-level and donor-grouped), the per-fold phi and C_R over all 28,231 genes
(program_scoring/added_genes/weights/<task>__<scheme>.npz); the 40 contamination variants of Fig. 6d were scored from
those exports. For every task, scheme and fold this script

  1. fits CellScorer on the training cells (raw counts) and compares phi and C_R with the exported ones;
  2. scores the held-out cells on the curated panels and on the 40 contamination variants;
  3. compares, per cell, the calibrated scores on the curated panels with the original formula applied to the
     exported weights, and evaluates every panel (balanced accuracy and macro-F1 of the argmax, mean one-vs-rest
     AUROC) against reuse_by_task.csv (Fig. 6a, cell-level folds) and added_genes/results/eval_B (Fig. 6d, both
     schemes, all 41 panels);
and for tf_cd8_dom compares the raw held-out scores with program_scores_tf_cd8_dom.csv (Fig. 6b, c).
Fold assignments, cell selection and panels are read from the original exports (data splits and inputs, not
RECAST computations).

    python check_program_scoring.py [--panels orig|all] [TASK ...]
Writes program_scoring_weights.csv, program_scoring_percell.csv, program_scoring_metrics.csv (with _<tasks> when tasks are given)."""
import argparse
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import common as C
import recast

PC = os.path.join(C.SB, "program_scoring")
ADDED = os.path.join(PC, "added_genes")
DATASET_FILES = os.path.join(C.SB, "dataset_files")
MASTER = os.path.join(C.SB, "paper", "program_scoring_tables", "data", "reuse_by_task.csv")
CD8_SCORES = os.path.join(C.SB, "validation", "biology", "program_scores_tf_cd8_dom.csv")
MIN_CELLS, CAP = 50, 8000          # cell-type size floor and stratified cap of the original loader
LABEL_MAP = {
    "baron_pancreas": {"activated_stellate": "stellate", "quiescent_stellate": "stellate"},
    "pelka_crc_epi": {"cE01 (Stem/TA-like)": "Stem_TA", "cE03 (Stem/TA-like prolif)": "Prolif",
                      "cE05 (Enterocyte 2)": "Enterocyte", "cE08 (Goblet)": "Goblet", "cE09 (Best4)": "Best4",
                      "cE10 (Tuft)": "Tuft", "cE11 (Enteroendocrine)": "Enteroendocrine"},
}
DISTINCT = ["pbmc3k", "lawlor", "ss3_pbmc", "ren", "stephenson", "dominguez_conde", "yoshida"]


def source(task):
    """(folder, stem, label file, label map) of a task, as the original loader defined them."""
    if task in LABEL_MAP:
        d = os.path.join(DATASET_FILES, task, task)
        return d, task, os.path.join(d, f"bench_labels_{task}.txt"), LABEL_MAP[task]
    if task in DISTINCT:
        return C.ATLASES, task, os.path.join(PC, "distinct_labels", f"labels_{task}.txt"), {}
    spec = C.config("lineage_registry").LINEAGES[task]
    return C.ATLASES, spec["ds"], os.path.join(C.ATLASES, spec["label_file"]), {}


def task_cells(task):
    """Raw counts of the task's cells (rows in the original order), labels, cell types, curated panels."""
    inp = ad.read_h5ad(os.path.join(PC, "tasks", f"{task}.h5ad"))
    cand = list(inp.uns["cand"])
    panels = json.loads(inp.uns["panels"])
    y = inp.obs["label"].astype(str).to_numpy()
    genes = list(map(str, inp.var_names))
    assert genes == C.scimilarity_genes()
    d, stem, label_file, lmap = source(task)
    names, _ = C.attribution_obs(d, stem)
    lab = np.array([l.strip() for l in open(label_file)])
    assert len(lab) == len(names), (task, len(lab), len(names))
    if lmap:
        lab = np.array([lmap.get(v, v) for v in lab])
    idx = np.where(np.isin(lab, cand))[0]
    if idx.size > CAP:
        rng = np.random.RandomState(0)
        per = max(MIN_CELLS, CAP // len(cand))
        yk = lab[idx]
        sel = [np.where(yk == c)[0] if (yk == c).sum() <= per else rng.choice(np.where(yk == c)[0], per, replace=False)
               for c in cand]
        idx = idx[np.sort(np.concatenate(sel))]
    assert (lab[idx] == y).all(), task
    a = ad.AnnData(C.counts(d, stem, names[idx], genes, integer=stem != "lawlor"))
    a.var_names = genes
    a.obs_names = names[idx]
    a.obs["label"] = pd.Categorical(y)
    X = sp.csr_matrix(a.X, dtype="float64")
    tot = np.asarray(X.sum(1)).ravel()
    L = (sp.diags(1e4 / np.where(tot == 0, 1, tot)) @ X).log1p()
    dX = float(abs(L - sp.csr_matrix(inp.X, dtype="float64")).max())     # original matrix vs lognorm(counts)
    return a, y, cand, panels, sp.csc_matrix(inp.X, dtype="float32"), dX


def original_scores(Xc, phi, cref, panel, gpos):
    """The original per-cell score, mean_g phi[g] * max(0, x_g - C_R[g]), on the exported weights; NaN for a
    panel with no gene in the data (score_variants_recast.py)."""
    gidx = np.array([gpos[g] for g in dict.fromkeys(panel) if g in gpos], dtype=int)
    if gidx.size == 0:
        return np.full(Xc.shape[0], np.nan)
    Xg = Xc[:, gidx].toarray()
    return (np.clip(Xg - cref[gidx], 0.0, None) * phi[gidx]).mean(1)


def evaluate_variants(S, y, cand, nonempty):
    """program_scoring/added_genes/evaluate.py, evaluate(): a candidate whose panel has no gene in the data is left out of the argmax, and AUROC
    is averaged over the other candidates. (The package scores such a panel 0, as score_programs did for the
    Fig. 6a table; the evaluation of the variants does not use it.)"""
    S = np.asarray(S, dtype="float64").copy()
    S[:, ~nonempty] = np.nan
    pred = pd.DataFrame(S, columns=cand).idxmax(axis=1).fillna("__none__").to_numpy()
    ba = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=cand, zero_division=0)
    auc = np.mean([roc_auc_score((y == c).astype(int), S[:, k]) for k, c in enumerate(cand) if nonempty[k]])
    return float(ba), float(f1), float(auc)


def evaluate(S, y, cand):
    pred = pd.DataFrame(S, columns=cand).idxmax(axis=1).to_numpy()
    ba = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=cand, zero_division=0)
    auc = np.mean([roc_auc_score((y == c).astype(int), S[:, k]) for k, c in enumerate(cand) if 0 < (y == c).sum() < len(y)])
    return float(ba), float(f1), float(auc)


def run(task, enc, all_panels):
    t0 = time.time()
    a, y, cand, panels, Xc, dX = task_cells(task)
    gpos = {g: i for i, g in enumerate(a.var_names)}
    V = json.load(open(os.path.join(ADDED, "variants_B", f"{task}.json")))
    assert V["orig"] == {c: panels[c] for c in V["orig"]}, task
    if not all_panels:
        V = {"orig": V["orig"]}
    nonempty = {v: np.array([any(g in gpos for g in gs[c]) for c in cand]) for v, gs in V.items()}
    Z = enc.embed(a.X)
    n, K = a.n_obs, len(cand)
    wrows, crows, mrows = [], [], []
    for scheme in ("cell", "donor"):
        w = np.load(os.path.join(ADDED, "weights", f"{task}__{scheme}.npz"), allow_pickle=True)
        assert list(w["cand"]) == cand and (w["y"] == y).all(), (task, scheme)
        fold = w["fold"]
        Zv = {v: np.zeros((n, K), "float32") for v in V}
        raw = np.zeros((n, K), "float32")
        stored = np.zeros((n, K))
        for f in range(5):
            tr, te = fold != f, fold == f
            sc = recast.CellScorer(enc).fit(a[tr], "label", states=cand, embeddings=Z[tr])
            phi = sc.weights_[cand].to_numpy("float64").T
            cref = sc.reference_profile_[cand].to_numpy("float64").T
            wrows.append(dict(task=task, scheme=scheme, fold=f, max_abs_phi=float(abs(w["phi"][f]).max()),
                              max_abs_dphi=float(abs(phi - w["phi"][f]).max()),
                              max_abs_dcref=float(abs(cref - w["cref"][f]).max())))
            for v, gs in V.items():
                Zv[v][te] = sc.score(a[te], gs, calibrate="zscore")[cand].to_numpy()
            raw[te] = sc.score(a[te], panels)[cand].to_numpy()
            for k, c in enumerate(cand):                      # the original z_train, from the exported weights
                s = original_scores(Xc, w["phi"][f, k].astype("float64"), w["cref"][f, k].astype("float64"),
                                     panels[c], gpos)
                mu, sd = s[tr].mean(), s[tr].std()
                stored[te, k] = (s[te] - mu) / (sd if sd > 0 else 1.0)
        ne = nonempty["orig"]
        same = (np.where(ne, Zv["orig"], -np.inf).argmax(1) == np.where(ne, stored, -np.inf).argmax(1))
        crows.append(dict(task=task, scheme=scheme, n_cells=n, n_empty_panels=int((~ne).sum()),
                          max_abs_dz=float(abs(Zv["orig"][:, ne] - stored[:, ne]).max()),
                          argmax_agree=int(same.sum()), dX_lognorm=dX))
        ev = pd.read_csv(os.path.join(ADDED, "results", "eval_B", f"{task}.csv"))
        ev = ev[(ev.method == "RECAST") & (ev.calibration == "matched") & (ev.scheme == scheme)].set_index("variant")
        for v in V:
            ba, f1, auc = evaluate_variants(Zv[v], y, cand, nonempty[v])
            mrows.append(dict(task=task, scheme=scheme, panel=v, ba=ba, f1=f1, auc=auc,
                              ba_stored=ev.loc[v, "balanced_acc"], f1_stored=ev.loc[v, "macro_f1"],
                              auc_stored=ev.loc[v, "mean_auroc"]))
        if scheme == "cell":
            m = pd.read_csv(MASTER).set_index("dataset").loc[task]
            ba, f1, auc = evaluate(Zv["orig"], y, cand)
            mrows.append(dict(task=task, scheme=scheme, panel="orig_master_m7", ba=ba, f1=f1, auc=auc,
                              ba_stored=m.recast_ba, f1_stored=m.recast_f1, auc_stored=m.recast_auc))
            if task == "tf_cd8_dom":
                d6 = pd.read_csv(CD8_SCORES)
                assert (d6.gt_state.to_numpy() == y).all()
                P6 = d6[[f"RECAST::{c}" for c in cand]].to_numpy()
                crows.append(dict(task=task, scheme="cell_raw_fig6bc", n_cells=n, max_abs_dz=float(abs(raw - P6).max()),
                                  argmax_agree=int((raw.argmax(1) == P6.argmax(1)).sum()), dX_lognorm=dX))
    print(f"  {task}: n={n} K={K} max|dphi|={max(r['max_abs_dphi'] for r in wrows):.2e} "
          f"max|dz|={max(r['max_abs_dz'] for r in crows):.2e} argmax "
          + ", ".join(f"{r['scheme']} {r['argmax_agree']}/{r['n_cells']}" for r in crows)
          + f" ({time.time() - t0:.0f}s)", flush=True)
    C.assert_package_only()
    return wrows, crows, mrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", choices=["orig", "all"], default="all")
    ap.add_argument("tasks", nargs="*")
    A = ap.parse_args()
    tasks = A.tasks or list(pd.read_csv(MASTER).dataset)
    enc = C.scimilarity()
    W, Cc, M = [], [], []
    for t in tasks:
        w, c, m = run(t, enc, A.panels == "all")
        W += w; Cc += c; M += m
    W, Cc, M = pd.DataFrame(W), pd.DataFrame(Cc), pd.DataFrame(M)
    for k in ("ba", "f1", "auc"):
        M[f"d_{k}"] = (M[k] - M[f"{k}_stored"]).abs()
    print(f"\nweights: max |phi_package - phi_original| = {W.max_abs_dphi.max():.2e} (max |phi| {W.max_abs_phi.max():.2e}), "
          f"max |C_R difference| = {W.max_abs_dcref.max():.2e}")
    print(f"per cell (curated panels, z): max |dz| = {Cc[Cc.scheme != 'cell_raw_fig6bc'].max_abs_dz.max():.2e}; argmax equal "
          f"{int(Cc[Cc.scheme != 'cell_raw_fig6bc'].argmax_agree.sum())}/{int(Cc[Cc.scheme != 'cell_raw_fig6bc'].n_cells.sum())}")
    print("metrics, max |package - original| by comparison:")
    M["what"] = np.where(M.panel == "orig_master_m7", "Fig 6a (master_m7, cell folds)",
                         np.where(M.panel == "orig", "curated panels (eval_B)", "40 contamination variants (eval_B)"))
    print(M.groupby(["what", "scheme"])[["d_ba", "d_f1", "d_auc"]].max().to_string(float_format=lambda v: f"{v:.2e}"))
    print(f"exactly equal balanced accuracy: {int((M.d_ba == 0).sum())}/{len(M)}, macro-F1: {int((M.d_f1 == 0).sum())}/{len(M)}")
    suffix = "" if not A.tasks else "_" + "_".join(A.tasks)
    C.write(W, f"program_scoring_weights{suffix}.csv")
    C.write(Cc, f"program_scoring_percell{suffix}.csv")
    C.write(M, f"program_scoring_metrics{suffix}.csv")


if __name__ == "__main__":
    main()
