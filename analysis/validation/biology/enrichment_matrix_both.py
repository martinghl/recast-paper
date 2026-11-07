"""The program-enrichment matrix with the fourth gene set (Fig. 5).

Same computation, background, collections, gene sets and BH-FDR as enrichment_matrix.py, plus

    DE_cosg_unique = COSG top 30 \\ RECAST top 30      (the mirror of RECAST_unique = RECAST top 30 \\ COSG top 30)

so that method-specific genes can be compared both ways (|F \\ D| == |D \\ F| by construction).  Writes
enrichment_full_matrix{VTAG}sym.csv / .log; the RECAST, RECAST_unique and DE_cosg rows are checked to equal
enrichment_full_matrix{VTAG}.csv.
Env: RECAST_RESULT_TAG, GENESETS_DIR (fs_genesets_v2 for the _v3 results)."""
import numpy as np, pandas as pd, glob, os, pathlib, re, time
from pathlib import Path
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

FSB = Path("/path/to/research/analysis/subtype_benchmark")
BV  = Path("/path/to/research/analysis/validation/biology")
TOPK = 30
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
GSDIR = os.environ.get("GENESETS_DIR") or str(FSB / "fs_genesets")
t0 = time.time(); L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))
log(f"tag={VTAG!r} genesets={GSDIR}")
BG = set(x.strip() for x in open(FSB / "results/go_genes.txt") if x.strip()); M = len(BG)

def load_terms(csv, smin=8, smax=500):
    df = pd.read_csv(csv); src, tgt = df.columns[0], df.columns[1]
    d = {}
    for s, g in df.groupby(src):
        genes = set(g[tgt]) & BG
        if smin <= len(genes) <= smax: d[s] = genes
    return d

COLLECTIONS = [("CollecTRI", load_terms(BV / "collectri_human.csv")),
               ("Hallmark",  load_terms(BV / "hallmark_human.csv")),
               ("Reactome",  load_terms(BV / "reactome_human.csv")),
               ("GO-BP",     load_terms(BV / "gobp_human.csv"))]
log(f"background={M} genes | " + " | ".join(f"{n}={len(t)}" for n, t in COLLECTIONS))

def ora_full(geneset, terms, alpha=0.05):
    S = set(geneset) & BG; N = len(S)
    if N < 3: return []
    names, ps, ks = [], [], []
    for t, tg in terms.items():
        k = len(S & tg)
        names.append(t); ks.append(k)
        ps.append(hypergeom.sf(k - 1, M, len(tg), N) if k > 0 else 1.0)
    fdr = multipletests(ps, method="fdr_bh")[1]
    out = []
    for i in range(len(names)):
        if ks[i] > 0 and fdr[i] < alpha:
            ov = sorted(set(geneset) & terms[names[i]] & BG)
            out.append((names[i], float(fdr[i]), ks[i], len(terms[names[i]]), ";".join(ov)))
    return out

rows, sizes = [], []
for pf in sorted(glob.glob(str(FSB / "panels/topN_*_recast_M0.csv"))):
    tag = re.search(r"topN_(.+)_recast_M0", os.path.basename(pf)).group(1)
    gf = pathlib.Path(GSDIR) / f"{tag}.csv"
    if not gf.exists(): continue
    foc = pd.read_csv(pf)
    de = pd.read_csv(gf); de.columns = [c.strip().strip('"') for c in de.columns]
    for c in ["method", "cluster", "gene"]:
        de[c] = de[c].astype(str).str.strip().str.strip('"')
    de_ref = de[de.method == "cosg"]
    for state, g in foc.groupby("cluster"):
        F = list(g.sort_values("rank").gene[:TOPK])
        D = list(de_ref[de_ref.cluster == state].sort_values("rank").gene[:TOPK])
        if not D: continue
        Fu = [x for x in F if x not in set(D)]
        Du = [x for x in D if x not in set(F)]
        sizes.append(dict(lineage=tag, state=str(state), n_recast=len(F), n_cosg=len(D), n_recast_unique=len(Fu),
                          n_cosg_unique=len(Du), n_recast_unique_bg=len(set(Fu) & BG), n_cosg_unique_bg=len(set(Du) & BG)))
        for label, gs in [("RECAST", F), ("RECAST_unique", Fu), ("DE_cosg", D), ("DE_cosg_unique", Du)]:
            for cname, terms in COLLECTIONS:
                for term, fdr, k, ssz, ov in ora_full(gs, terms):
                    rows.append(dict(lineage=tag, state=str(state), geneset=label, collection=cname,
                                     term=term, fdr=fdr, k_overlap=k, set_size=ssz, overlap_genes=ov))
    log(f"  {tag}: {foc.cluster.nunique()} states  [{time.time()-t0:.0f}s]")

R = pd.DataFrame(rows)
S = pd.DataFrame(sizes)
# --- check the inputs: the three shared gene sets reproduce enrichment_full_matrix{VTAG}.csv
ref = pd.read_csv(BV / f"enrichment_full_matrix{VTAG}.csv")
key = ["lineage", "state", "geneset", "collection", "term"]
a = R[R.geneset != "DE_cosg_unique"].sort_values(key).reset_index(drop=True)
b = ref.sort_values(key).reset_index(drop=True)
assert len(a) == len(b), (len(a), len(b))
assert (a[key].values == b[key].values).all(), "term rows differ from the paper's matrix"
assert np.allclose(a.fdr.values, b.fdr.values, atol=1e-12), "FDR values differ from the paper's matrix"
log(f"shared gene sets reproduce enrichment_full_matrix{VTAG}.csv: {len(b)} rows")
R.to_csv(BV / f"enrichment_full_matrix{VTAG}sym.csv", index=False)
S.to_csv(BV / f"enrichment_setsizes{VTAG}sym.csv", index=False)
log(f"states={S.shape[0]} | |F\\D| == |D\\F| in all states: {bool((S.n_recast_unique == S.n_cosg_unique).all())} | "
    f"median |F\\D| = {S.n_recast_unique.median():.0f} (min {S.n_recast_unique.min()}, max {S.n_recast_unique.max()})")
for label in ["RECAST", "RECAST_unique", "DE_cosg", "DE_cosg_unique"]:
    d = R[R.geneset == label]
    log(f"  {label:15s}: {len(d)} sig-term rows over {d[['lineage','state']].drop_duplicates().shape[0]} states")
    for cname, _ in COLLECTIONS:
        dc = d[d.collection == cname]
        log(f"       {cname:10s}: {len(dc):5d} sig terms; {dc[['lineage','state']].drop_duplicates().shape[0]} states with >=1")
log(f"\n[done {time.time()-t0:.0f}s] wrote enrichment_full_matrix{VTAG}sym.csv")
open(BV / f"enrichment_full_matrix{VTAG}sym.log", "w").write("\n".join(L))
