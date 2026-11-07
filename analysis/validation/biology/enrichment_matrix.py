"""Full subtype x term over-representation matrix of the RECAST top-30 genes (Fig. 5).

Same computation, inputs, gene sets, background and BH-FDR as pathway_regulon_enrichment.py, but every term with
BH-FDR < 0.05 is written with its FDR and gene overlap, for the RECAST top 30, the RECAST-unique genes (RECAST top 30
not in the COSG top 30) and the COSG top 30, so all gene sets share one term universe.

CPU only (scipy hypergeom); the collections are the stored CSVs.  Out:
  enrichment_full_matrix.csv   lineage,state,geneset,collection,term,fdr,k_overlap,set_size,overlap_genes
  enrichment_full_matrix.log"""
import numpy as np, pandas as pd, glob, os, pathlib, re, time
from pathlib import Path
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

FSB = Path("/path/to/research/analysis/subtype_benchmark")
BV  = Path("/path/to/research/analysis/validation/biology")
TOPK = 30
# RECAST panels from subtype_benchmark/panels; the COSG ranking that defines RECAST_unique from the comparator
# gene lists in GSDIR (GENESETS_DIR).
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
GSDIR = os.environ.get("GENESETS_DIR") or str(FSB / "fs_genesets")
t0 = time.time(); L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))

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
    """All BH-FDR<0.05 terms. FDR corrected over the FULL collection (constant hypothesis count)."""
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

rows = []
npan = 0
for pf in sorted(glob.glob(str(FSB / "panels/topN_*_recast_M0.csv"))):
    tag = re.search(r"topN_(.+)_recast_M0", os.path.basename(pf)).group(1)
    gf = pathlib.Path(GSDIR) / f"{tag}.csv"
    if not gf.exists(): continue
    npan += 1
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
        for label, gs in [("RECAST", F), ("RECAST_unique", Fu), ("DE_cosg", D)]:
            for cname, terms in COLLECTIONS:
                for term, fdr, k, ssz, ov in ora_full(gs, terms):
                    rows.append(dict(lineage=tag, state=str(state), geneset=label, collection=cname,
                                     term=term, fdr=fdr, k_overlap=k, set_size=ssz, overlap_genes=ov))
    log(f"  {tag}: {foc.cluster.nunique()} states  [{time.time()-t0:.0f}s]")

R = pd.DataFrame(rows)
R.to_csv(BV / f"enrichment_full_matrix{VTAG}.csv", index=False)
log("")
log(f"panels={npan} | states={R[['lineage','state']].drop_duplicates().shape[0]} | rows(sig terms)={len(R)}")
for label in ["RECAST", "RECAST_unique", "DE_cosg"]:
    d = R[R.geneset == label]
    nst = d[["lineage", "state"]].drop_duplicates().shape[0]
    log(f"  {label:13s}: {len(d)} sig-term rows over {nst} states")
    for cname, _ in COLLECTIONS:
        dc = d[d.collection == cname]
        enr = dc[["lineage", "state"]].drop_duplicates().shape[0]
        log(f"       {cname:10s}: {len(dc):5d} sig terms; {enr} states with >=1")
log(f"\n[done {time.time()-t0:.0f}s] wrote enrichment_full_matrix.csv")
open(BV / f"enrichment_full_matrix{VTAG}.log", "w").write("\n".join(L))
