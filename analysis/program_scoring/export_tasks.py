"""Export the 28 per-cell scoring tasks as the comparator scorers saw them in Fig. 6a.

The comparator rows (ANS roster + expr_mean) were computed in score_programs.run() on the matrix returned by
score_programs.load_dataset(): log-normalized expression in SCimilarity gene order (28,231 genes), capped at 8,000
cells by a seed-0 stratified draw, restricted to candidate populations with >= 50 cells, with each candidate's curated
panel restricted to genes in that order.  This script calls the same function and writes one sparse h5ad per task,
so the other scorers (run on the CPU server, which has no GPU) receive the same cells, labels, candidates and gene
sets.

Output: program_scoring/tasks/<task>.h5ad
  X                 CSR float32, log-normalized, SCimilarity gene order
  obs['label']      the population label (the evaluation target)
  uns['cand']       candidate populations, sorted (the argmax order used by evaluate())
  uns['panels']     {candidate: [genes]} -- the curated panel genes present in the gene order
  uns['regime']     fine | distinct
Run (GPU, env SC; importing score_programs loads the encoder only for its gene order):
  CUDA_VISIBLE_DEVICES=0 /path/to/envs/SC/bin/python program_scoring/export_tasks.py"""
import json, os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

SB = "/path/to/research/analysis"
sys.path.insert(0, os.path.join(SB, "program_scoring"))
import score_programs as B  # noqa: E402

OUT = os.path.join(SB, "program_scoring", "tasks")
TASKS = pd.read_csv(os.path.join(SB, "paper/program_scoring_tables/data/reuse_by_task.csv"))["dataset"].tolist()
assert len(TASKS) == 28 and all(t in B.DATASETS for t in TASKS), TASKS

os.makedirs(OUT, exist_ok=True)
manifest = []
for t in TASKS:
    X, y, cand, sig_idx, sig_genes, regime = B.load_dataset(t)
    A = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"label": y}, index=[str(i) for i in range(len(y))]),
                   var=pd.DataFrame(index=B.GO))
    A.uns["cand"] = list(cand)
    A.uns["panels"] = json.dumps({c: list(sig_genes[c]) for c in cand})
    A.uns["regime"] = regime
    A.write_h5ad(os.path.join(OUT, f"{t}.h5ad"), compression="gzip")
    manifest.append(dict(task=t, regime=regime, n_cells=len(y), n_states=len(cand),
                         panel_sizes=";".join(f"{c}:{len(sig_genes[c])}" for c in cand),
                         expr_mean_check=float(np.mean([X[:, sig_idx[c]].mean() if sig_idx[c].size else 0.0
                                                        for c in cand]))))
    print(t, regime, X.shape, len(cand), flush=True)
pd.DataFrame(manifest).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
print("wrote", OUT)
