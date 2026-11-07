"""RECAST and SIGnature per-cell scores for every panel variant, from the exported per-fold weights
(CPU server).

  python score_variants_recast.py <task> <variants.json> <out.npz>
For each fold scheme (cell, donor) and each fold f, with that fold's training-cell weights (export_weights.py):
  RECAST     S_ic = mean_{g in G_c} phi_c[g] * max(0, x_ig - C_R,g)     (score_programs.m1_scores)
  SIGnature  S_ic = mean_{g in G_c} psi_c[g] * max(0, x_ig - C_R,g)     (reference_free_scoring.reference_free_scores,
                                                                          gated)
Test cells get raw scores and z_train (mean and standard deviation of the same fold's training-cell scores, a standard
deviation of 0 -> 1), as in Fig. 6a.  A candidate whose variant panel is empty -> NaN.
Output npz: <method>__<scheme>__<raw|ztrain> arrays (n_variants, n_cells, K) float32, plus variants, cand, folds."""
import json, sys, time
import numpy as np, anndata as ad
ROOT = "/path/to/cpu_server/program_scoring"; W = "/path/to/cpu_server/program_scoring/added_genes/weights"
task, vfile, out = sys.argv[1:4]
a = ad.read_h5ad(f"{ROOT}/inputs/{task}.h5ad")
X = a.X.toarray().astype("float32"); cand = list(a.uns["cand"]); n, K = X.shape[0], len(cand)
gi = {g: i for i, g in enumerate(a.var_names)}; assert len(gi) == 28231
V = json.load(open(vfile)); vids = list(V)
res = {}; t0 = time.time()
for scheme in ("cell", "donor"):
    w = np.load(f"{W}/{task}__{scheme}.npz", allow_pickle=True)
    assert list(w["cand"]) == cand and (w["y"] == a.obs["label"].astype(str).to_numpy()).all()
    fold = w["fold"]; res[f"fold__{scheme}"] = fold
    for m, wk in (("RECAST", "phi"), ("SIGnature", "psi")):
        raw = np.full((len(vids), n, K), np.nan, "float32"); zt = np.full((len(vids), n, K), np.nan, "float32")
        for f in range(5):
            tr, te = fold != f, fold == f
            for k, c in enumerate(cand):
                wt, cr = w[wk][f, k], w["cref"][f, k]
                for vi, v in enumerate(vids):
                    gidx = np.array([gi[g] for g in dict.fromkeys(V[v][c]) if g in gi], dtype=int)
                    if gidx.size == 0:
                        continue
                    S = (np.clip(X[:, gidx] - cr[gidx], 0.0, None) * wt[gidx]).mean(1)
                    mu, sd = S[tr].mean(), S[tr].std()
                    raw[vi, te, k] = S[te]
                    zt[vi, te, k] = (S[te] - mu) / (sd if sd > 0 else 1.0)
        res[f"{m}__{scheme}__raw"], res[f"{m}__{scheme}__ztrain"] = raw, zt
np.savez_compressed(out, variants=np.array(vids), cand=np.array(cand), **res)
print(f"done {task} {len(vids)} variants {time.time()-t0:.0f}s", flush=True)
