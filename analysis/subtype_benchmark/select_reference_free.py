"""Reference-free SIGnature panels for the cross-atlas transfer (Fig. 6e).

select_recast.py writes the RECAST panel of a lineage (top 50 genes per subtype by contrastive attribution) to
panels/topN_<id>_recast_M0.csv.  This script selects the SIGnature panel from the same cells (same attribution h5ad,
same bench_labels subset, same >= 50-cell rule, same gene vocabulary); only the ranking differs:

  RECAST     positive-channel contrastive attribution of subtype c against the rest (recast.cluster_attribution)
  SIGnature  subtype mean of the stored reference-free per-cell attribution, layers['attribution_ig']

-> panels/topN_<id>_signature_native.csv  (cluster, rank, gene).  CPU only; the per-cell attribution is stored.

  cd /path/to/research/analysis
  NUMPY_MADVISE_HUGEPAGE=0 /path/to/envs/SC/bin/python subtype_benchmark/select_reference_free.py"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/subtype_benchmark"):
    if p not in sys.path:
        sys.path.insert(0, p)
import lineage_registry as C                                          # noqa: E402

ATLASES = f"{SB}/atlases"
OUT = f"{SB}/subtype_benchmark/panels"
TOPN = 50
# the transfer arm (transfer_pairs.COMP) covers these lineages only
IDS = [f"tf_{c}_{d}" for c in ("cd8", "cd4", "mono") for d in ("dom", "ren", "steph", "yosh")] + \
      [f"tf_b_{d}" for d in ("ren", "steph", "yosh")]


def select(dsid):
    e = C.LINEAGES[dsid]
    stem, states, min_cells = e["ds"], e["states"], e["min_cells"]
    lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{e['label_file']}")])
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    genes = np.asarray(att.var_names).astype(str)
    assert len(lab_all) == att.n_obs, f"{dsid}: labels {len(lab_all)} != cells {att.n_obs}"
    keepstates = sorted([s for s in states if (lab_all == s).sum() >= min_cells])
    keep = np.isin(lab_all, keepstates)
    L = att.layers["attribution_ig"]
    Aat = (L[keep].toarray() if sp.issparse(L) else np.asarray(L[keep])).astype("float32")
    lab = lab_all[keep]
    del att
    # the RECAST panel of this lineage must span the same subtypes
    ref = f"{OUT}/topN_{dsid}_recast_M0.csv"
    if os.path.exists(ref):
        rs = sorted(pd.read_csv(ref)["cluster"].astype(str).unique())
        assert rs == keepstates, f"{dsid}: subtypes differ from the RECAST panel {rs} vs {keepstates}"
    rows = []
    for c in keepstates:
        mu = Aat[lab == c].mean(0)
        top = np.argsort(-mu)[:TOPN]
        rows += [dict(cluster=str(c), rank=i + 1, gene=str(genes[j])) for i, j in enumerate(top)]
    out = f"{OUT}/topN_{dsid}_signature_native.csv"
    assert not os.path.exists(out) or os.environ.get("FORCE"), f"{out} exists"
    pd.DataFrame(rows, columns=["cluster", "rank", "gene"]).to_csv(out, index=False)
    print(f"[{dsid}] {len(rows)} rows over {len(keepstates)} states -> {out}", flush=True)


if __name__ == "__main__":
    for i in (sys.argv[1:] or IDS):
        select(i)
