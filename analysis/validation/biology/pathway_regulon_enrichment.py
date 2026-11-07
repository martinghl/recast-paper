"""Over-representation of transcription-factor regulons (CollecTRI) and MSigDB Hallmark programs in the
RECAST top genes of every subtype, and in the RECAST genes that the comparator ranking misses (RECAST-unique), against
a background of all genes.

Hypergeometric test, BH-FDR per gene set; CPU only (scipy); collections from decoupler 2.2.0.
Inputs: subtype_benchmark/panels/topN_<lin>_recast_M0.csv (RECAST), fs_genesets/<lin>.csv (comparators, COSG),
        validation/biology/{collectri_human.csv, hallmark_human.csv}, results/go_genes.txt (background).
Out: validation/biology/{enrichment_per_state.csv, enrichment_summary.txt}"""
import numpy as np, pandas as pd, glob, os, re
from pathlib import Path
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

FSB = Path("/path/to/research/analysis/subtype_benchmark")
BV = Path("/path/to/research/analysis/validation/biology")
RNG = np.random.default_rng(0); TOPK = 30
lines = []
def log(s=""):
    print(s); lines.append(str(s))

# background gene universe (what RECAST could pick)
BG = set(x.strip() for x in open(FSB/"results/go_genes.txt") if x.strip())
M = len(BG)

def load_terms(csv, smin=8, smax=500):
    df = pd.read_csv(csv)
    src, tgt = df.columns[0], df.columns[1]
    d = {}
    for s, g in df.groupby(src):
        genes = set(g[tgt]) & BG
        if smin <= len(genes) <= smax: d[s] = genes
    return d

REG = load_terms(BV/"collectri_human.csv")
HALL = load_terms(BV/"hallmark_human.csv")
log(f"background={M} genes | CollecTRI regulons(8-500)={len(REG)} | Hallmark sets={len(HALL)}")

def ora(geneset, terms):
    """Return (best_term, best_fdr, n_sig@0.05) for a gene set vs a term dict. BH-FDR is corrected over the FULL
    collection (non-overlapping terms get p=1) so the hypothesis count is constant regardless of gene-set size —
    avoids the small-set n_sig inflation of correcting over only-overlapping terms."""
    S = set(geneset) & BG; N = len(S)
    if N < 3: return (None, 1.0, 0)
    names, ps = [], []
    for t, tg in terms.items():
        k = len(S & tg)
        names.append(t); ps.append(hypergeom.sf(k - 1, M, len(tg), N) if k > 0 else 1.0)
    if not ps: return (None, 1.0, 0)
    fdr = multipletests(ps, method="fdr_bh")[1]
    j = int(np.argmin(fdr))
    return (names[j], float(fdr[j]), int((fdr < 0.05).sum()))

def rand_sig(size, terms, B=40):
    bg = list(BG); s = []
    for _ in range(B):
        gs = RNG.choice(len(bg), size, replace=False)
        s.append(ora([bg[i] for i in gs], terms)[2])
    return float(np.mean(s))

# match RECAST panels to DE genesets by lineage tag
rows = []
for pf in sorted(glob.glob(str(FSB/"panels/topN_*_recast_M0.csv"))):
    tag = re.search(r"topN_(.+)_recast_M0", os.path.basename(pf)).group(1)
    gf = FSB/f"fs_genesets/{tag}.csv"
    if not gf.exists(): continue
    foc = pd.read_csv(pf)                                   # cluster, rank, gene
    de = pd.read_csv(gf)                                    # method, cluster, rank, gene
    de.columns = [c.strip().strip('"') for c in de.columns]
    for c in ["method", "cluster", "gene"]:
        de[c] = de[c].astype(str).str.strip().str.strip('"')
    de_ref = de[de.method == "cosg"]
    for state, g in foc.groupby("cluster"):
        F = list(g.sort_values("rank").gene[:TOPK])
        D = list(de_ref[de_ref.cluster == state].sort_values("rank").gene[:TOPK])
        if not D: continue
        Fu = [x for x in F if x not in set(D)]
        for label, gs in [("RECAST", F), ("RECAST_unique", Fu), ("DE_cosg", D)]:
            rt, rf, rn = ora(gs, REG); ht, hf, hn = ora(gs, HALL)
            rows.append(dict(lineage=tag, state=state, geneset=label, n=len(gs),
                             reg_top=rt, reg_fdr=rf, reg_nsig=rn, hall_top=ht, hall_fdr=hf, hall_nsig=hn))
    log(f"  {tag}: {foc.cluster.nunique()} states")

R = pd.DataFrame(rows)
R.to_csv(BV/"enrichment_per_state.csv", index=False)

# random floors (size-matched to typical sets)
rf_reg = rand_sig(TOPK, REG); rf_hall = rand_sig(TOPK, HALL)
med_uniq = int(R[R.geneset == "RECAST_unique"].n.median())
rfu_reg = rand_sig(med_uniq, REG); rfu_hall = rand_sig(med_uniq, HALL)

log(""); log("="*88); log("ORA coherence — fraction of states with >=1 significant term (BH-FDR<0.05)"); log("="*88)
for label in ["RECAST", "RECAST_unique", "DE_cosg"]:
    d = R[R.geneset == label]; nst = len(d)
    log(f"  {label:13s} (n_states={nst}, median size={int(d.n.median())}):")
    log(f"     CollecTRI regulon:  {int((d.reg_fdr<0.05).sum())}/{nst} states enriched ({(d.reg_fdr<0.05).mean()*100:.0f}%), median best-FDR {d.reg_fdr.median():.1e}, mean n_sig {d.reg_nsig.mean():.1f}")
    log(f"     Hallmark pathway:   {int((d.hall_fdr<0.05).sum())}/{nst} states enriched ({(d.hall_fdr<0.05).mean()*100:.0f}%), median best-FDR {d.hall_fdr.median():.1e}, mean n_sig {d.hall_nsig.mean():.1f}")
log(f"  RANDOM floor (size {TOPK}):     regulon mean n_sig {rf_reg:.2f} | hallmark mean n_sig {rf_hall:.2f}")
log(f"  RANDOM floor (size {med_uniq}, matched to RECAST_unique): regulon {rfu_reg:.2f} | hallmark {rfu_hall:.2f}")

# qualitative: top regulon per RECAST state (is the state's TF recovered?)
log(""); log("Top CollecTRI regulon per RECAST state (FDR<0.05 only):")
for _, r in R[(R.geneset == "RECAST") & (R.reg_fdr < 0.05)].sort_values("reg_fdr").head(20).iterrows():
    log(f"     {r.lineage:16s} {r.state:34s} -> {r.reg_top:10s} (FDR {r.reg_fdr:.1e})")

log(""); log("="*88); log("READING"); log("="*88)
log("""[interpret against the random floor above]
  - If RECAST and RECAST_unique enrich regulons/pathways at a rate FAR above random, RECAST's genes (incl. those DE
    misses) are biologically coherent -> orthogonal, panel-independent validation that RECAST recovers real biology.
  - The per-state top regulon shows whether the state's canonical TF is recovered (e.g. Treg -> FOXP3).""")
open(BV/"enrichment_summary.txt", "w").write("\n".join(lines))
log(""); log(f"[wrote] {BV}/enrichment_per_state.csv, enrichment_summary.txt")
