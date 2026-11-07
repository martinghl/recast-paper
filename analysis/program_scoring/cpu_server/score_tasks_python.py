"""Python scorers on the exported per-cell tasks (CPU server), a check against Fig. 6a, and the inputs of
the R scorers.

Modes (each writes <out>/scores/<task>__<method>.csv.gz, cells x candidates, NaN for an empty panel):
  prep     write the R inputs: <out>/rin/<task>.mtx.gz (genes x cells), <task>.genes.txt, <task>.json
  parity   expr_mean (as score_programs: zeros for an empty panel), and UCell / ANS through signaturescoring with the
           settings of ans_replication/ans_baselines.py  [env scoring_python]
  scse     SCSE (Pont 2019): sum of UMI over the set / total UMI = sum_set expm1(X) / 1e4 on log1p(CP10K)
  ulm      decoupleR ULM, decoupler 2.2.0 dc.mt.ulm, tmin = 1 (no panel is dropped)  [env decoupler]
Usage: python score_tasks_python.py <mode> --root /path/to/cpu_server/program_scoring [--jobs N]"""
import argparse, glob, json, os, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings("ignore")


def load(root, task):
    a = ad.read_h5ad(f"{root}/inputs/{task}.h5ad")
    cand = list(a.uns["cand"])
    panels = json.loads(a.uns["panels"])
    return a, cand, panels


def write_scores(root, task, method, S, cand):
    os.makedirs(f"{root}/scores", exist_ok=True)
    pd.DataFrame(S, columns=cand).to_csv(f"{root}/scores/{task}__{method}.csv.gz", index=False)


def do_prep(root, task):
    import scipy.io
    a, cand, panels = load(root, task)
    os.makedirs(f"{root}/rin", exist_ok=True)
    X = sp.csr_matrix(a.X).astype(np.float64)
    with open(f"{root}/rin/{task}.mtx", "wb") as fh:
        scipy.io.mmwrite(fh, X.T.tocoo(), precision=9)          # genes x cells
    os.system(f"gzip -f {root}/rin/{task}.mtx")
    open(f"{root}/rin/{task}.genes.txt", "w").write("\n".join(a.var_names) + "\n")
    json.dump({"cand": cand, "panels": panels, "n_cells": int(a.n_obs)}, open(f"{root}/rin/{task}.json", "w"))
    return task


def do_parity(root, task):
    import signaturescoring as ssc
    a, cand, panels = load(root, task)
    X = sp.csr_matrix(a.X)
    gi = {g: i for i, g in enumerate(a.var_names)}
    # expr_mean exactly as score_programs.run(): zeros for an empty panel
    Xd = X.toarray().astype("float32")                     # dense float32, as in score_programs
    S = np.column_stack([Xd[:, [gi[g] for g in panels[c]]].mean(1) if panels[c]
                         else np.zeros(a.n_obs) for c in cand]).astype("float32")
    write_scores(root, task, "expr_mean_parity", S, cand)
    A = ad.AnnData(X=X.toarray().astype("float32"), var=pd.DataFrame(index=a.var_names))
    A.obs_names = [str(i) for i in range(a.n_obs)]
    cfg = {"UCell_parity": ("ucell_scoring", {"maxRank": 1500, "joblib_kwargs": {"n_jobs": 1}}),
           "ANS_parity": ("adjusted_neighborhood_scoring", {"ctrl_size": 100})}
    for name, (method, kw) in cfg.items():
        cols = []
        for c in cand:
            genes = [g for g in panels[c] if g in gi]
            try:
                ssc.score_signature(A, gene_list=genes, method=method, score_name="_tmp", use_raw=False, **kw)
                cols.append(A.obs["_tmp"].to_numpy(dtype="float64"))
            except Exception as e:  # noqa: BLE001 -- as ans_baselines.score_all_methods: NaN column
                print(f"[{task}] {name} x {c} failed: {type(e).__name__}: {e}", flush=True)
                cols.append(np.full(a.n_obs, np.nan))
            if "_tmp" in A.obs.columns:
                del A.obs["_tmp"]
        write_scores(root, task, name, np.column_stack(cols), cand)
    return task


def do_scse(root, task):
    a, cand, panels = load(root, task)
    X = sp.csr_matrix(a.X)
    gi = {g: i for i, g in enumerate(a.var_names)}
    E = X.copy(); E.data = np.expm1(E.data)
    S = np.column_stack([np.asarray(E[:, [gi[g] for g in panels[c]]].sum(1)).ravel() / 1e4 if panels[c]
                         else np.full(a.n_obs, np.nan) for c in cand])
    write_scores(root, task, "SCSE", S, cand)
    return task


def do_ulm(root, task):
    import decoupler as dc
    a, cand, panels = load(root, task)
    net = pd.DataFrame([(c, g, 1.0) for c in cand for g in panels[c]], columns=["source", "target", "weight"])
    A = ad.AnnData(X=sp.csr_matrix(a.X).astype("float32"), var=pd.DataFrame(index=a.var_names))
    A.obs_names = [str(i) for i in range(a.n_obs)]
    dc.mt.ulm(A, net, tmin=1, verbose=False)
    sc = A.obsm["score_ulm"]
    S = np.column_stack([sc[c].to_numpy(dtype="float64") if c in sc.columns else np.full(a.n_obs, np.nan)
                         for c in cand])
    write_scores(root, task, "decoupleR_ULM", S, cand)
    return task


MODES = {"prep": do_prep, "parity": do_parity, "scse": do_scse, "ulm": do_ulm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--root", default="/path/to/cpu_server/program_scoring")
    ap.add_argument("--jobs", type=int, default=28)
    ap.add_argument("--tasks", default="")
    A = ap.parse_args()
    tasks = A.tasks.split(",") if A.tasks else sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{A.root}/inputs/*.h5ad"))
    fn = MODES[A.mode]
    with Pool(min(A.jobs, len(tasks))) as p:
        for t in p.starmap(fn, [(A.root, t) for t in tasks]):
            print(A.mode, "done", t, flush=True)
