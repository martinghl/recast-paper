"""RECAST per-lineage timing split into stages, on the 20 subtype lineages.

  embed_s    the encoder forward pass over the lineage's cells (enc.embed on raw counts: per-cell log-normalization +
             SCimilarity encoder, batched on the GPU)
  select_s   cluster_attribution end to end (embedding pass + centroids + IG over all subtypes)
  post_s     select_s - embed_s
  ig_est_s   select_s - n_states * embed_s (estimate of the centroid and IG work)

Each measurement is the median of N_REP = 3 runs after one warm-up.  Hardware and software are recorded in the output.
-> results/recast_timing_percompartment_e2e.csv (its cell counts are read by paper/benchmark_tables/selection_times.py)
Env SC (GPU, CUDA_VISIBLE_DEVICES=1).  Run with the GPU otherwise idle."""
import os, platform, sys, time
import numpy as np, pandas as pd, anndata as ad, torch
SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/ans_replication", f"{SB}/subtype_benchmark"):
    if p not in sys.path: sys.path.insert(0, p)
import lineage_registry as C                       # noqa: E402
from select_recast import _load_raw              # noqa: E402
from scoring_benchmark.datasets import _go_gene_order, SCIMILARITY_MODEL   # noqa: E402
import recast                             # noqa: E402
from scoring_benchmark.run_scoring_benchmark import _PerCellLognormEncoder  # noqa: E402
ATLASES = f"{SB}/atlases"; OUT = f"{SB}/subtype_benchmark/results"
LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]
N_REP = 3
enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda"))
go = _go_gene_order()


def load(dsid):
    e = C.LINEAGES[dsid]; stem = e["ds"]
    lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad"); obs = np.asarray(att.obs_names).astype(str)
    keepstates = sorted([s for s in e["states"] if (lab_all == s).sum() >= e["min_cells"]])
    keep = np.isin(lab_all, keepstates); Craw_go = _load_raw(stem, obs[keep]); del att
    a_raw = ad.AnnData(Craw_go); a_raw.var_names = go; a_raw.obs["state"] = lab_all[keep].astype(str)
    return a_raw


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); return time.perf_counter() - t0


warm = load(LIN20[0]); _ = recast.cluster_attribution(enc, warm, "state", reference="rest", gate="dC", centroid="mean_lognorm", qc="off")
_ = enc.embed(warm.X)
rows = []
for dsid in LIN20:
    a = load(dsid); n = a.n_obs
    emb = np.median([timed(lambda: enc.embed(a.X)) for _ in range(N_REP)])
    sel = np.median([timed(lambda: recast.cluster_attribution(enc, a, "state", reference="rest", gate="dC", centroid="mean_lognorm", qc="off")) for _ in range(N_REP)])
    rows.append(dict(lineage=dsid, n_cells=int(n), n_states=int(a.obs.state.nunique()), embed_s=round(emb, 4), select_s=round(sel, 4),
                     post_s=round(sel - emb, 4), embed_ms_per_cell=round(emb * 1e3 / n, 4), select_ms_per_cell=round(sel * 1e3 / n, 4),
                     post_ms_per_cell=round((sel - emb) * 1e3 / n, 4), ig_est_s=round(sel - int(a.obs.state.nunique()) * emb, 4)))
    print(f"[{dsid}] n={n} embed={emb:.3f}s select={sel:.3f}s post={sel-emb:.3f}s  ({sel*1e3/n:.3f} ms/cell)", flush=True)
df = pd.DataFrame(rows)
df["gpu"] = torch.cuda.get_device_name(0); df["cpu"] = platform.processor() or "Intel Xeon 6787P"; df["torch"] = torch.__version__
df["n_rep"] = N_REP; df["batch_size"] = getattr(enc, "batch_size", "encoder default")
df.to_csv(f"{OUT}/recast_timing_percompartment_e2e.csv", index=False)
print(df[["lineage", "n_cells", "embed_ms_per_cell", "select_ms_per_cell", "post_ms_per_cell"]].to_string(index=False))
print("median ms/cell: embed", df.embed_ms_per_cell.median().round(3), "select", df.select_ms_per_cell.median().round(3), "post", df.post_ms_per_cell.median().round(3))
