"""Python scorers without labels on every panel variant of a task (CPU server).

  python score_variants_python.py <task> <method> <variants.json> <out.npz> [<i>/<n>]
  optional <i>/<n>: score only the variants vids[i::n] (chunked runs; merge_chunks.py reassembles them)
method in: ANS Seurat Seurat_AG Seurat_LVG Scanpy Jasmine_LH Jasmine_OR UCell   (signaturescoring, env scoring_python,
           settings of ans_replication/ans_baselines.py; np.random.seed(0) before every call)
           expr_mean SCSE                                                     (numpy, env scoring_python)
           decoupleR_ULM                                                      (decoupler 2.2.0 dc.mt.ulm, tmin=1, env
                                                                               decoupler; all variants in one network)
All cells of the task are scored together, once, without labels.  An empty panel -> NaN column.
Output npz: S (n_variants, n_cells, K) float32, variants, cand."""
import json, sys, time, warnings
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
warnings.filterwarnings("ignore")
ROOT = "/path/to/cpu_server/program_scoring"
task, method, vfile, out = sys.argv[1:5]
a = ad.read_h5ad(f"{ROOT}/inputs/{task}.h5ad")
cand = list(a.uns["cand"]); V = json.load(open(vfile)); vids = list(V)
if len(sys.argv) > 5:
    ci, cn = map(int, sys.argv[5].split("/")); vids = vids[ci::cn]
X = sp.csr_matrix(a.X); n, K = a.n_obs, len(cand)
gi = {g: i for i, g in enumerate(a.var_names)}
def panel(v, c):                                  # the variant's panel, de-duplicated, restricted to the universe
    return [g for g in dict.fromkeys(V[v][c]) if g in gi]
S = np.full((len(vids), n, K), np.nan, "float32")
t0 = time.time()
SSC = {"ANS": ("adjusted_neighborhood_scoring", {"ctrl_size": 100}, False),
       "Seurat": ("seurat_scoring", {"ctrl_size": 100, "n_bins": 25}, False),
       "Seurat_AG": ("seurat_ag_scoring", {"n_bins": 25}, False),
       "Seurat_LVG": ("seurat_lvg_scoring", {"ctrl_size": 100, "n_bins": 25, "lvg_computation_version": "v1",
                                             "lvg_computation_method": "seurat"}, False),
       "Scanpy": ("scanpy_scoring", {"ctrl_size": 100, "n_bins": 25}, False),
       "Jasmine_LH": ("jasmine_scoring", {"score_method": "likelihood"}, True),
       "Jasmine_OR": ("jasmine_scoring", {"score_method": "oddsratio"}, True),
       "UCell": ("ucell_scoring", {"maxRank": 1500}, True)}
nfail = 0
if method in SSC:
    import signaturescoring as ssc
    meth, kw, serial = SSC[method]; kw = dict(kw)
    if serial:
        kw["joblib_kwargs"] = {"n_jobs": 1}
    A = ad.AnnData(X=X.toarray().astype("float32"), var=pd.DataFrame(index=a.var_names))
    A.obs_names = [str(i) for i in range(n)]
    for vi, v in enumerate(vids):
        for k, c in enumerate(cand):
            genes = panel(v, c)
            if not genes:
                continue
            np.random.seed(0)
            try:
                ssc.score_signature(A, gene_list=genes, method=meth, score_name="_tmp", use_raw=False, **kw)
                S[vi, :, k] = A.obs["_tmp"].to_numpy(dtype="float64")
            except Exception as e:  # noqa: BLE001 -- as ans_baselines.score_all_methods: NaN column
                nfail += 1
                print(f"[{task}] {method} {v} {c} failed: {type(e).__name__}: {e}", flush=True)
            if "_tmp" in A.obs.columns:
                del A.obs["_tmp"]
elif method == "expr_mean":
    Xd = X.toarray().astype("float32")
    for vi, v in enumerate(vids):
        for k, c in enumerate(cand):
            idx = [gi[g] for g in panel(v, c)]
            if idx:
                S[vi, :, k] = Xd[:, idx].mean(1)
elif method == "SCSE":
    E = X.copy().astype("float64"); E.data = np.expm1(E.data); E = E.tocsc()
    for vi, v in enumerate(vids):
        for k, c in enumerate(cand):
            idx = [gi[g] for g in panel(v, c)]
            if idx:
                S[vi, :, k] = np.asarray(E[:, idx].sum(1)).ravel() / 1e4
elif method == "decoupleR_ULM":
    import decoupler as dc
    net = pd.DataFrame([(f"{v}::{c}", g, 1.0) for v in vids for c in cand for g in panel(v, c)],
                       columns=["source", "target", "weight"])
    A = ad.AnnData(X=X.astype("float32"), var=pd.DataFrame(index=a.var_names)); A.obs_names = [str(i) for i in range(n)]
    dc.mt.ulm(A, net, tmin=1, verbose=False)
    sc = A.obsm["score_ulm"]
    for vi, v in enumerate(vids):
        for k, c in enumerate(cand):
            key = f"{v}::{c}"
            if key in sc.columns:
                S[vi, :, k] = sc[key].to_numpy(dtype="float64")
else:
    raise SystemExit(f"unknown method {method}")
np.savez_compressed(out, S=S, variants=np.array(vids), cand=np.array(cand))
print(f"done {task} {method} {len(vids)} variants {time.time()-t0:.0f}s failures={nfail}", flush=True)
