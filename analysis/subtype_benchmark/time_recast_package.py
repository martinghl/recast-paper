"""RECAST selection time through the package (Supplementary Figs. 3 and 10): recast.SCimilarityEncoder(...,
normalize=True) on raw counts and cluster_attribution(reference='rest'), on the 20 subtype lineages and six atlases of
timing_inputs.py.

Per object (median of N_REP runs after a warm-up, GPU otherwise idle, torch.cuda.synchronize around every call):
  lognorm_s    recast.centroid.lognorm_rows(a.X): per-cell tp10k-lognorm, sparse, CPU
  embed_s      enc.embed(a.X): lognorm + one SCimilarity forward pass over every cell (GPU)
  profiles_s   recast.centroid.LabelProfiles(a.X, labels): per-label sums, one pass (CPU)
  select_s     cluster_attribution(..., qc='off'): embedding + profiles + per-subtype u / centroids / IG
  select_qc_s  the same with the default contrast QC (qc='silent')
  ig_s         select_s - embed_s - profiles_s: the per-subtype loop (u, centroids, 50-step IG) alone
-> results/recast_timing_lineages.csv, results/recast_timing_whole_atlas.csv
Env SC (GPU, CUDA_VISIBLE_DEVICES=1)."""
import os, platform, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_inputs import OBJECTS, OUT, SCIMILARITY_MODEL, load   # noqa: E402
import recast                                                            # noqa: E402
from recast.centroid import lognorm_rows, LabelProfiles                  # noqa: E402
N_REP = 3
enc = recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda", normalize=True)

def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); return time.perf_counter() - t0

def med(fn):
    return float(np.median([timed(fn) for _ in range(N_REP)]))

warm = load(*OBJECTS[0][:2]); _ = enc.embed(warm.X)
_ = recast.cluster_attribution(enc, warm, "state", reference="rest", gate="dC", centroid="mean_lognorm", qc="off")
del warm
rows = []
for name, kind, key in OBJECTS:
    a = load(name, kind); n, k = a.n_obs, int(a.obs[key].nunique()); lab = a.obs[key].to_numpy()
    r = dict(dataset=name, kind=kind, n_cells=int(n), n_states=k)
    r["lognorm_s"] = med(lambda: lognorm_rows(a.X))
    r["embed_s"] = med(lambda: enc.embed(a.X))
    r["profiles_s"] = med(lambda: LabelProfiles(a.X, lab, "mean_lognorm"))
    r["select_s"] = med(lambda: recast.cluster_attribution(enc, a, key, reference="rest", gate="dC", centroid="mean_lognorm", qc="off"))
    r["select_qc_s"] = med(lambda: recast.cluster_attribution(enc, a, key, reference="rest", gate="dC", centroid="mean_lognorm", qc="silent"))
    r["ig_s"] = r["select_s"] - r["embed_s"] - r["profiles_s"]
    r["select_ms_per_cell"] = r["select_s"] * 1e3 / n; r["embed_ms_per_cell"] = r["embed_s"] * 1e3 / n
    r["ig_ms_per_state"] = r["ig_s"] * 1e3 / k
    rows.append({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})
    print(f"[{kind}:{name}] n={n} K={k} lognorm={r['lognorm_s']:.3f}s embed={r['embed_s']:.3f}s profiles={r['profiles_s']:.3f}s "
          f"select={r['select_s']:.3f}s (+qc {r['select_qc_s']:.3f}s) ig/state={r['ig_ms_per_state']:.1f}ms  {r['select_ms_per_cell']:.4f} ms/cell", flush=True)
    del a
df = pd.DataFrame(rows)
df["gpu"] = torch.cuda.get_device_name(0); df["cpu"] = platform.processor() or "Intel Xeon 6787P"; df["torch"] = torch.__version__
df["recast_version"] = recast.__version__; df["n_rep"] = N_REP; df["encoder_path"] = "recast.SCimilarityEncoder(normalize=True)"
df["numpy_madvise_hugepage"] = os.environ.get("NUMPY_MADVISE_HUGEPAGE", "default")   # 0 here: this host's THP compaction stalls (page-cache-full memory) would otherwise inflate every fresh allocation 30-50x
df["load_avg_1min"] = round(os.getloadavg()[0], 1)
os.makedirs(OUT, exist_ok=True)
df[df.kind == "lineage"].drop(columns="kind").rename(columns={"dataset": "lineage"}).to_csv(f"{OUT}/recast_timing_lineages.csv", index=False)
df[df.kind == "atlas"].drop(columns="kind").rename(columns={"n_states": "n_types"}).to_csv(f"{OUT}/recast_timing_whole_atlas.csv", index=False)
print(df.to_string(index=False))
