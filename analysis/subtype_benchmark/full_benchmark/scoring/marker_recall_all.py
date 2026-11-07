"""Recall@20 of the subtype marker panels for RECAST and every comparator (Fig. 3a), with a bootstrap of RECAST's rank.

Per unit (lineage | subtype): recall@20 = |top-20 genes ∩ panel| / |panel|, panel = the unit's genes in
ground_truth/panels/marker_panels.json (UNIT_PANELS) that are in the encoder vocabulary.  Subtypes without a panel are
skipped.  RECAST's lists are the panels topN_<lineage>_recast_M0.csv; the comparators' lists are the depth-50 files
<lineage>.csv in GENESETS_DIR.  Lineages: the original lineages (results/fs_cm_recovery_uniq_<lineage><LINEAGE_TAG>.csv)
plus the added lineages (added_datasets.NEW_IDS).

The printout bootstraps units (B = 10,000, with replacement) for RECAST's rank, P(#1), P(top 3), recall interval and
how many methods it beats (paired-difference interval), pooled and by tier (fine_immune / fine_epithelial / fine_brain;
the pancreas lineage is a distinct_control tier with strong hormone markers).
->  RECALL_OUT_DIR/recall_by_subtype.csv   [lineage, state, method, recall, unit, tier]
Env analysis (numpy, pandas).  Run by run_scoring_all.sh.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
RESULTS = f"{FSB}/results"
OUTW = os.environ.get("RECALL_OUT_DIR", f"{FULL_BENCHMARK}/scoring/out")
GENESETS = os.environ.get("GENESETS_DIR", f"{FULL_BENCHMARK}/scoring/in/genesets50")
GTAG = os.environ.get("LINEAGE_TAG", "_v3")
PANELS = os.environ.get("PANELS_DIR", f"{FULL_BENCHMARK}/scoring/in/panels")
sys.path.insert(0, FSB)
sys.path.insert(0, f"{FULL_BENCHMARK}/added_lineages")
from cluster_key import pick   # noqa: E402  tolerate make.names-renamed cluster labels
import added_datasets as D   # noqa: E402
UNIT_GT = json.load(open(os.environ.get("UNIT_PANELS", f"{FULL_BENCHMARK}/ground_truth/panels/marker_panels.json")))
GO = set(l.strip() for l in open(f"{RESULTS}/go_genes.txt") if l.strip())
K = 20
B = 10000

# tier per lineage (for stratified reporting); everything not listed is immune
TIER = {"fs_panc_baron": "distinct_control",   # endocrine alpha/beta/delta/gamma, strong hormone markers
        "fs_lung_hlca": "fine_epithelial",     # airway / alveolar epithelium
        "fs_gut_smillie": "fine_epithelial"}   # colon epithelium
for _lin in D.NEW_IDS:                         # added lineages: lung / gut epithelium and brain glia / neurons
    TIER[_lin] = "fine_brain" if ("_glia_" in _lin or "_neuron_" in _lin) else "fine_epithelial"
tier_of = lambda lin: TIER.get(lin, "fine_immune")


def m0_geneset(fsid):
    df = pd.read_csv(f"{PANELS}/topN_{fsid}_recast_M0.csv").sort_values(["cluster", "rank"])
    return {str(c): list(g["gene"].astype(str)) for c, g in df.groupby("cluster")}


def de_genesets(fsid):
    df = pd.read_csv(f"{GENESETS}/{fsid}.csv").sort_values(["method", "cluster", "rank"])
    return {str(m): {str(c): list(g["gene"].astype(str)) for c, g in gm.groupby("cluster")}
            for m, gm in df.groupby("method")}


# ---- per-unit recall: rows (lineage, state, method, recall) ----
_stems = [re.search(r"recovery_uniq_(.+)\.csv", os.path.basename(f)).group(1)
          for f in glob.glob(f"{RESULTS}/fs_cm_recovery_uniq_*.csv")]
lineages = sorted([s[: -len(GTAG)] for s in _stems if s.endswith(GTAG)] + D.NEW_IDS)
rows = []
for fsid in lineages:
    m0 = m0_geneset(fsid); de = de_genesets(fsid)
    clusters = sorted(m0.keys())
    assert {k.split("|", 1)[1] for k in UNIT_GT if k.split("|", 1)[0] == fsid} <= set(clusters), fsid
    markers = {c: set(UNIT_GT.get(f"{fsid}|{c}", [])) & GO for c in clusters}

    def recall(gs, c):
        # `pick`, not gs.get: R's make.names renames some cluster labels in the comparator harness
        # ("Naive B cells" -> "Naive.B.cells"); a plain .get() would score such a method 0 for that subtype.
        mk = markers.get(c, set())
        return (len(set(list(pick(gs, c))[:K]) & mk) / len(mk)) if mk else np.nan

    for c in clusters:
        if not markers[c]:
            continue                                            # subtype without a marker panel
        rows.append(dict(lineage=fsid, state=c, method="recast_M0", recall=recall(m0, c)))
        for m, gs in de.items():
            rows.append(dict(lineage=fsid, state=c, method=m, recall=recall(gs, c)))

df = pd.DataFrame(rows).dropna(subset=["recall"])
df["unit"] = df.lineage + "|" + df.state
df["tier"] = df.lineage.map(tier_of)


def analyze(sub, label, over="unit"):
    """Bootstrap RECAST rank/recall/beats over `over`-units (state or lineage) for a method x unit table."""
    piv = sub.pivot_table(index=over, columns="method", values="recall")
    cols = list(piv.columns)
    if "recast_M0" not in cols:
        print(f"[{label}] no RECAST"); return
    fi = cols.index("recast_M0"); comp = [i for i in range(len(cols)) if i != fi]
    U = piv.values; nU = len(piv); pe = np.nanmean(U, 0)
    pe_rank = int(np.nansum(pe > pe[fi])) + 1
    rng = np.random.default_rng(0)
    ranks = np.empty(B, int); recs = np.empty(B); diffs = np.empty((B, len(comp)))
    for b in range(B):
        idx = rng.integers(0, nU, nU); mu = np.nanmean(U[idx], 0)
        recs[b] = mu[fi]; ranks[b] = int(np.nansum(mu > mu[fi])) + 1; diffs[b] = mu[fi] - mu[comp]
    lo = np.nanpercentile(diffs, 2.5, 0); hi = np.nanpercentile(diffs, 97.5, 0)
    order = np.argsort(-pe)
    print(f"\n=== [{label}] {nU} {over}-units x {len(cols)} methods ===")
    print(f"  RECAST recall@20={pe[fi]:.3f}  point rank #{pe_rank}/{len(cols)}"
          f"  | top5: {[(cols[i], round(float(pe[i]), 3)) for i in order[:5]]}")
    print(f"  boot rank median #{int(np.median(ranks))} 95%CI[#{int(np.percentile(ranks, 2.5))},"
          f"#{int(np.percentile(ranks, 97.5))}]  | P(#1)={(ranks == 1).mean():.3f}  P(top3)={(ranks <= 3).mean():.3f}")
    print(f"  recall 95%CI[{np.percentile(recs, 2.5):.3f},{np.percentile(recs, 97.5):.3f}]"
          f"  | beats {int((lo > 0).sum())}/{len(comp)} (CI>0);  loses to {int((hi < 0).sum())}/{len(comp)} (CI<0)")


print(f"lineages ({len(lineages)}): {lineages}")
print(f"total per-state units = {df.unit.nunique()} | by tier: "
      f"{df.groupby('tier').unit.nunique().to_dict()}")

analyze(df, "POOLED all tissues (per-state)")
analyze(df[df.tier.isin(["fine_immune", "fine_epithelial", "fine_brain"])], "FINE tier (immune+epithelial+brain, per-state)")
for t in ["fine_immune", "fine_epithelial", "fine_brain", "distinct_control"]:
    s = df[df.tier == t]
    if len(s): analyze(s, f"tier={t} (per-state)")
analyze(df, "POOLED all tissues (per-lineage)", over="lineage")

df.to_csv(f"{OUTW}/recall_by_subtype.csv", index=False)
print(f"\n[wrote {OUTW}/recall_by_subtype.csv] {len(df)} rows", flush=True)
