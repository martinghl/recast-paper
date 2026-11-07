import os
"""Complete 28,231-gene ranking of the DA state (gene, recast_rank, de_rank, eta2_dpt), for the rank-against-
rank panel of Supplementary Fig. 7.  Same computation and inputs as development_case.py, which writes only the top 20.
Env: SC (GPU)."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys, scanpy as sc, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig
from captum.attr import IntegratedGradients
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)

SB = "/path/to/research/analysis"; OUT = f"{SB}/validation/biology"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)
K = 20; DEK = 50; NBIN = 25; STATE = "DA"
def log(s=""): print(s, flush=True)

log("loading neural_diff …")
raw = ad.read_h5ad(f"{SB}/ans_replication/bundles/neural_diff_a_raw.h5ad")
genes = np.asarray(raw.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
ct = raw.obs["celltype"].astype(str).to_numpy()
Xc = raw.X.tocsr() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)
tot = np.asarray(Xc.sum(1)).ravel(); Xln = Xc.multiply(1e4/np.clip(tot,1,None)[:,None]).log1p().tocsr().astype("float32")
n = Xc.shape[0]; log(f"neural_diff {Xc.shape}")

log("embedding (SCimilarity) + diffusion pseudotime …")
Z = contrastive_ig.embed(Xln.toarray())
A = ad.AnnData(X=sp.csr_matrix((n,1),dtype="float32")); A.obsm["X_scim"] = Z.astype("float32")
sc.pp.neighbors(A, use_rep="X_scim", n_neighbors=15); sc.tl.diffmap(A)
prog = np.where(ct=="P_FPP")[0]; cen = Z[prog].mean(0)
A.uns["iroot"] = int(prog[np.argmin(((Z[prog]-cen)**2).sum(1))]); sc.tl.dpt(A)
dpt = A.obs["dpt_pseudotime"].to_numpy().astype("float64"); dpt[~np.isfinite(dpt)] = np.nanmax(dpt[np.isfinite(dpt)])
lb = ad.read_h5ad(f"{SB}/ans_replication/bundles/neural_diff_a_log.h5ad")
assert (lb.obs["celltype"].astype(str).to_numpy()==ct).all()
tp = lb.obs["time_point"].astype(str).to_numpy(); del lb
tnum = np.array([{"D11":0,"D30":1,"D52":2}.get(x,np.nan) for x in tp])
if spearmanr(dpt, tnum, nan_policy="omit").correlation < 0: dpt = dpt.max()-dpt
log(f"[validation] Spearman(dpt, real day) = {spearmanr(dpt,tnum,nan_policy='omit').correlation:+.3f}")

def eta2_binned(Xsp, order, B):
    edges = np.quantile(order, np.linspace(0,1,B+1)); edges[-1]+=1e-9
    b = np.clip(np.searchsorted(edges, order, side="right")-1, 0, B-1)
    S = sp.csr_matrix((np.ones(len(b)),(b,np.arange(len(b)))), shape=(B,len(b)))
    nk = np.asarray(S.sum(1)).ravel(); binmean = np.asarray((S@Xsp).todense())/np.clip(nk[:,None],1,None)
    grand = np.asarray(Xsp.sum(0)).ravel()/Xsp.shape[0]
    ss_b = (nk[:,None]*(binmean-grand[None,:])**2).sum(0)
    sq = np.asarray(Xsp.multiply(Xsp).sum(0)).ravel(); ss_t = sq - Xsp.shape[0]*grand**2
    return np.clip(ss_b/np.clip(ss_t,1e-12,None),0,1), binmean
eta_dpt, binmean = eta2_binned(Xln, dpt, NBIN)

# ---- RECAST one-vs-rest FULL ranking for DA ----
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
def recast_full(tgt, ref, n_steps=50):
    u = Z[tgt].mean(0)-Z[ref].mean(0); u=(u/(np.linalg.norm(u)+1e-12)).astype("float32")
    Cc = fc.centroid(Xc, tgt); Cr = fc.centroid(Xc, ref)   # paper centroid (mean_lognorm); validation/recast_centroid.py
    inp = torch.tensor(Cc[None],device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None],device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None],device=contrastive_ig.dev),), n_steps=n_steps).detach().cpu().numpy().ravel()
    a = np.where((Cc-Cr)>0, np.clip(a,0,None), -np.inf); return np.argsort(-a), a
tgt = np.where(ct==STATE)[0]; ref = np.where(ct!=STATE)[0]
fr, attr = recast_full(tgt, ref); recast_rank = {int(j): r+1 for r, j in enumerate(fr)}

# ---- DE full ranking for DA (one-vs-rest wilcoxon, ALL genes) ----
Ade = ad.AnnData(X=Xln, obs=pd.DataFrame({"ct":ct}, index=raw.obs_names)); Ade.var_names = genes
sc.tl.rank_genes_groups(Ade, "ct", method="wilcoxon", n_genes=len(genes))
de_order = list(Ade.uns["rank_genes_groups"]["names"][STATE]); de_rank = {g: r+1 for r, g in enumerate(de_order)}

# ---- FULL dump: every gene, RECAST rank vs one-vs-rest DE rank ----
full = pd.DataFrame({
    "gene": genes,
    "recast_rank": [recast_rank[i] for i in range(len(genes))],
    "de_rank": [de_rank.get(g, np.nan) for g in genes],
    "recast_attr": attr,
    "eta2_dpt": eta_dpt,
    "de_missed_top50": [ (de_rank.get(g, 10**9) > DEK) for g in genes ],
}).sort_values("recast_rank")
full["de_rank"] = full["de_rank"].astype("Int64")
full.to_csv(f"{OUT}/discovery_case_da_fullranks{VTAG}.csv", index=False)
log(f"[wrote] discovery_case_da_fullranks.csv  ({len(full)} genes)")

# ---- sanity: reproduce the highlighted stars exactly ----
STARS = ["SNCA","ELAVL3","L1CAM","ELAVL2","TTC9B","RPRM"]
log("\n=== highlight sanity (RECAST rank / DE rank) ===")
for g in STARS:
    r = full[full.gene==g].iloc[0]
    log(f"   {g:8s}  RECAST #{int(r.recast_rank):>3}  DE #{int(r.de_rank):>5}  eta2={r.eta2_dpt:.3f}")
log(f"\nfield size = {len(full)} genes; RECAST in [{full.recast_rank.min()},{full.recast_rank.max()}]  DE in [{int(full.de_rank.min())},{int(full.de_rank.max())}]")
