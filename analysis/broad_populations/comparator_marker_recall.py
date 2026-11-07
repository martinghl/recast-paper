"""Curated-marker recall of the comparator methods on the broad populations of each atlas, from their
selected-gene .rds files (each `$result` is a per-cluster gene table in ranked order), with the expert marker sets
and the protocol used for RECAST in recast_panels.py.  Env: SC / analysis."""
import rdata, glob, os, numpy as np, pandas as pd
ATLASES = "/path/to/research/analysis/atlases"
RDS = "/path/to/research/analysis/marker_methods_whole_atlas/results"
OUT = "/path/to/research/analysis/broad_populations/results"
DSS = ["pbmc3k", "lawlor", "ss3_pbmc", "citeseq", "ren", "stephenson", "dominguez_conde", "yoshida"]


def geneset_from_rds(f):
    d = rdata.read_rds(f)
    res = d["result"] if isinstance(d, dict) and "result" in d else d
    df = res if isinstance(res, pd.DataFrame) else pd.DataFrame(res)
    cols = {c.lower(): c for c in df.columns}
    ccol = next((cols[k] for k in ("cluster", "group", "cell_type", "cell", "celltype") if k in cols), None)
    gcol = next((cols[k] for k in ("gene", "feature", "names", "symbol", "gene_name") if k in cols), None)
    if ccol is None or gcol is None:
        raise ValueError(f"no cluster/gene cols in {list(df.columns)}")
    return {str(c): list(grp[gcol].astype(str)) for c, grp in df.groupby(ccol, sort=False)}


rows = []
for ds in DSS:
    Eall = pd.read_csv(f"{ATLASES}/broad_markers/expert_markers_long.csv"); Eall = Eall[Eall.dataset == ds]
    EXP = {ct: {g for g in grp.gene.astype(str)} for ct, grp in Eall.groupby("cell_type")}
    EXP = {ct: s for ct, s in EXP.items() if s}
    nfile = 0
    for f in sorted(glob.glob(f"{RDS}/{ds}_allgenes-*.rds")):
        meth = f.split(f"{ds}_allgenes-")[1][:-4]
        try:
            gs = geneset_from_rds(f)
        except Exception as e:
            print(f"  [skip {ds}/{meth}: {e}]", flush=True); continue
        nfile += 1
        for c in EXP:
            if c in gs:
                E = EXP[c]; n = max(len(E), 10)
                rows.append(dict(dataset=ds, method=meth, cluster=c, n_expert=len(E),
                                 recall=len(set(gs[c][:n]) & E) / len(E)))
    print(f"[{ds}] {nfile} baseline methods, {len(EXP)} expert cell-types", flush=True)

R = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
R.to_csv(f"{OUT}/baseline_arm2.csv", index=False)
# quick validation: pooled recall per method vs table2a
pool = R.groupby("method")["recall"].mean().sort_values(ascending=False)
print("\n=== comparator curated-marker recall, pooled (CellMarker sets), top 12 ===")
print(pool.head(12).round(3).to_string())
print(f"\n[wrote] baseline_arm2.csv  ({R.method.nunique()} methods x {R.dataset.nunique()} datasets)")
