"""Panel variants with added genes (Fig. 6d).

Per task, two pools built from all of the task's cells (independent of any scorer), excluding every gene in any
candidate's curated panel:
  random    genes detected (> 0) in >= 10% of the task's cells
  lineage   genes detected in >= 50% of the cells of every candidate, ranked by between-candidate differential
            d_g = max_c |mean_c - mean_all| / (sd_all + 1e-6) ascending, the 300 least differential.  If fewer than
            100 genes pass at 50%, the detection threshold steps down 0.4, 0.3, 0.2, 0.1 until >= 100 pass (the
            threshold used is recorded).
Variants: k in {5, 10, 20, 40}, draws 0..4, rng = default_rng([task_index, type_index, k, draw]); each candidate with
a non-empty panel gets k genes drawn without replacement from the pool, added to its curated panel; empty panels
stay empty.  Variant "orig" = the curated panels (k = 0).
Writes program_scoring/added_genes/variants_B/<task>.json  {variant_id: {candidate: [genes]}} and pools_B.csv."""
import glob, json, os
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

SB = "/path/to/research/analysis"
OUT = f"{SB}/program_scoring/added_genes/variants_B"
os.makedirs(OUT, exist_ok=True)
TASKS = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{SB}/program_scoring/tasks/*.h5ad"))
assert len(TASKS) == 28
DOSES, DRAWS, TYPES = [5, 10, 20, 40], range(5), ["random", "lineage"]
rows = []
for ti, t in enumerate(TASKS):
    a = ad.read_h5ad(f"{SB}/program_scoring/tasks/{t}.h5ad")
    X = sp.csr_matrix(a.X); y = a.obs["label"].astype(str).to_numpy(); cand = list(a.uns["cand"])
    panels = json.loads(a.uns["panels"]); genes = np.asarray(a.var_names).astype(str)
    inpanel = np.isin(genes, sorted({g for c in cand for g in panels[c]}))
    det_all = np.asarray((X > 0).mean(0)).ravel()
    det_c = np.vstack([np.asarray((X[y == c] > 0).mean(0)).ravel() for c in cand])
    mean_all = np.asarray(X.mean(0)).ravel()
    sd_all = np.sqrt(np.maximum(np.asarray(X.multiply(X).mean(0)).ravel() - mean_all ** 2, 0))
    mean_c = np.vstack([np.asarray(X[y == c].mean(0)).ravel() for c in cand])
    d = np.abs(mean_c - mean_all).max(0) / (sd_all + 1e-6)
    pool = {"random": genes[(det_all >= 0.10) & ~inpanel]}
    for thr in (0.5, 0.4, 0.3, 0.2, 0.1):
        ok = (det_c >= thr).all(0) & ~inpanel
        if ok.sum() >= 100:
            break
    order = np.where(ok)[0][np.argsort(d[ok], kind="stable")][:300]
    pool["lineage"] = genes[order]
    rows.append(dict(task=t, n_random=len(pool["random"]), n_lineage=len(pool["lineage"]), lineage_thr=thr,
                     lineage_d_max=float(d[order].max()), lineage_examples=",".join(pool["lineage"][:8])))
    V = {"orig": {c: list(panels[c]) for c in cand}}
    for yi, typ in enumerate(TYPES):
        P = np.sort(pool[typ])
        for k in DOSES:
            assert len(P) >= k, (t, typ, len(P))
            for r in DRAWS:
                rng = np.random.default_rng([ti, yi, k, r])
                V[f"{typ}_k{k}_d{r}"] = {c: (list(panels[c]) + list(rng.choice(P, k, replace=False)) if panels[c] else [])
                                        for c in cand}
    json.dump(V, open(f"{OUT}/{t}.json", "w"))
    print(t, rows[-1]["n_random"], rows[-1]["n_lineage"], thr, rows[-1]["lineage_examples"], flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/pools_B.csv", index=False)
