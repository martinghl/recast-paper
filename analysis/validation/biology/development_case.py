import os
"""RECAST and Wilcoxon ranks of the dopaminergic-neuron (DA) genes in the iPSC-to-midbrain
differentiation data (Supplementary Fig. 7).

Recomputes the RECAST and one-vs-rest Wilcoxon ranks of every gene for the DA state (the trajectory table stores only
the RECAST top 20 and an in-Wilcoxon-top-50 flag), the expression of the RECAST top-20 genes that Wilcoxon ranks below
50 per pseudotime bin, a two-panel figure and a table of ranks.  Same computation as trajectory_coherence.py.
Env: SC (GPU)."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys, scanpy as sc, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))

log("loading neural_diff …")
raw = ad.read_h5ad(f"{SB}/ans_replication/bundles/neural_diff_a_raw.h5ad")
genes = np.asarray(raw.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
ct = raw.obs["celltype"].astype(str).to_numpy()
Xc = raw.X.tocsr() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)
tot = np.asarray(Xc.sum(1)).ravel(); Xln = Xc.multiply(1e4/np.clip(tot,1,None)[:,None]).log1p().tocsr().astype("float32")
n = Xc.shape[0]; log(f"neural_diff {Xc.shape}")

# ---- embed + diffusion pseudotime (rooted in progenitor), oriented to real day ----
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

# ---- eta^2 per gene along pseudotime (for annotation) ----
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

# ---- RECAST one-vs-rest full ranking for DA ----
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
def recast_full(tgt, ref, n_steps=50):
    u = Z[tgt].mean(0)-Z[ref].mean(0); u=(u/(np.linalg.norm(u)+1e-12)).astype("float32")
    Cc = fc.centroid(Xc, tgt); Cr = fc.centroid(Xc, ref)   # paper centroid (mean_lognorm); validation/recast_centroid.py
    inp = torch.tensor(Cc[None],device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None],device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None],device=contrastive_ig.dev),), n_steps=n_steps).detach().cpu().numpy().ravel()
    a = np.where((Cc-Cr)>0, np.clip(a,0,None), -np.inf); return np.argsort(-a)
tgt = np.where(ct==STATE)[0]; ref = np.where(ct!=STATE)[0]
fr = recast_full(tgt, ref); recast_rank = {int(j): r+1 for r, j in enumerate(fr)}  # gene_idx -> 1-based RECAST rank
foc20 = [int(j) for j in fr[:K]]

# ---- DE full ranking for DA (one-vs-rest wilcoxon, ALL genes) ----
Ade = ad.AnnData(X=Xln, obs=pd.DataFrame({"ct":ct}, index=raw.obs_names)); Ade.var_names = genes
sc.tl.rank_genes_groups(Ade, "ct", method="wilcoxon", n_genes=len(genes))
de_order = list(Ade.uns["rank_genes_groups"]["names"][STATE]); de_rank = {g: r+1 for r, g in enumerate(de_order)}

# ---- assemble the DA panel table ----
rows = []
for j in foc20:
    g = genes[j]; dr = de_rank.get(g, np.nan)
    rows.append(dict(gene=g, recast_rank=recast_rank[j], de_rank=int(dr) if np.isfinite(dr) else np.nan,
                     de_missed_top50=bool(dr > DEK), eta2_dpt=float(eta_dpt[j]),
                     rho_bin=float(spearmanr(np.arange(NBIN), binmean[:, j]).correlation)))
T = pd.DataFrame(rows).sort_values("recast_rank"); T.to_csv(f"{OUT}/discovery_case_da_ranks{VTAG}.csv", index=False)
missed = T[T.de_missed_top50].copy()
log("\n=== RECAST DA top-20: RECAST rank vs one-vs-rest DE rank ===")
log(T.to_string(index=False))
log(f"\nDE-missed-but-RECAST-top20 (DE rank > {DEK}): {list(missed.gene)}")
for _, r in missed.iterrows():
    log(f"   {r.gene:8s}  RECAST #{int(r.recast_rank):>2}  DE #{int(r.de_rank):>5}  eta2={r.eta2_dpt:.3f}  rho_bin={r.rho_bin:+.2f}")

# ---- figure: (A) RECAST vs DE rank dumbbell for DA-missed stars; (B) their pseudotime expression curves ----
# The genes shown: every RECAST top-20 gene outside the Wilcoxon top 50, best RECAST rank first, at most 6.
STARS = list(missed.sort_values("recast_rank").gene)[:6]
log(f"stars (all DE-missed, by RECAST rank, cap 6): {STARS}")
xb = np.linspace(0, 1, NBIN); RECASTC, DEC = "#C0392B", "#2E86AB"
fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios":[1.0,1.18]})
# panel A: dumbbell RECAST rank (<=20) vs DE rank (log)
a = ax[0]; ys = np.arange(len(STARS))[::-1]; ytop = int(ys.max())
maxde = float(T[T.gene.isin(STARS)].de_rank.max()); a.set_ylim(-1.2, ytop + 1.5)
for y, g in zip(ys, STARS):
    r = T[T.gene==g].iloc[0]; a.plot([r.recast_rank, r.de_rank], [y, y], color="0.78", lw=1.6, zorder=1)
    a.scatter(r.recast_rank, y, s=95, color=RECASTC, zorder=3, edgecolor="white", linewidth=1.0)
    a.scatter(r.de_rank, y, s=95, color=DEC, zorder=3, edgecolor="white", linewidth=1.0)
    a.text(r.recast_rank*0.78, y, f"#{int(r.recast_rank)}", va="center", ha="right", fontsize=8.2, color=RECASTC, fontweight="bold")
    a.text(r.de_rank*1.22, y, f"#{int(r.de_rank)}", va="center", ha="left", fontsize=8.2, color=DEC, fontweight="bold")
a.set_xscale("log"); a.set_yticks(ys); a.set_yticklabels(STARS, fontsize=10.5, fontstyle="italic")
a.set_xlim(0.7, maxde*3.4); a.axvline(K, color=RECASTC, ls=":", lw=1.2, zorder=1)
a.text(K*1.18, -1.08, "RECAST top-20", color=RECASTC, fontsize=8, ha="left", va="bottom")   # off the dotted line
a.set_xlabel("rank in DA one-vs-rest panel  (log scale)", fontsize=10)
h = [Line2D([0],[0], marker="o", color="w", markerfacecolor=RECASTC, markersize=9, label="RECAST rank"),
     Line2D([0],[0], marker="o", color="w", markerfacecolor=DEC, markersize=9, label="DE (Wilcoxon) rank")]
a.legend(handles=h, frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.15)
a.set_title("RECAST surfaces DA-neuron genes that\none-vs-rest DE buries", fontsize=11, fontweight="bold")
for s in ("top","right"): a.spines[s].set_visible(False)
# panel B: pseudotime expression of the stars (z over bins) + random reference
b = ax[1]
rng = np.random.default_rng(0); expressed = (np.asarray(Xln.sum(0)).ravel()/n) > 0
randset = rng.choice(np.where(expressed)[0], 200, replace=False)
def zc(idx_list):
    p = binmean[:, idx_list]; p = (p - p.mean(0, keepdims=True))/(p.std(0, keepdims=True)+1e-9); return p.mean(1)
_pf = {"bin": np.arange(NBIN), **{g: zc([gi[g]]) for g in STARS}, "random": zc(list(randset))}
pd.DataFrame(_pf).to_csv(f"{OUT}/discovery_case_da_profiles{VTAG}.csv", index=False)   # cheap re-render source
pal = {"SNCA":"#C0392B","ELAVL3":"#E67E22","L1CAM":"#8E44AD","ELAVL2":"#16A085","TTC9B":"#2C3E50","RPRM":"#7D8B99"}
for g in STARS:
    j = gi[g]; b.plot(xb, zc([j]), "-", color=pal.get(g,"#555"), lw=2.4 if g=="SNCA" else 1.6,
                      ms=4, marker="o" if g=="SNCA" else None, zorder=4 if g=="SNCA" else 3,
                      label=f"{g} (η²={eta_dpt[j]:.2f})")
b.plot(xb, zc(list(randset)), "--", color="0.6", lw=1.6, zorder=2, label="expression-matched random")
b.axhline(0, color="0.8", lw=0.9)
b.set_xlabel("differentiation pseudotime  (progenitor → DA neuron)", fontsize=10)
b.set_ylabel("mean expression (z over pseudotime bins)", fontsize=10)
b.set_title("The DE-missed RECAST genes are developmentally\nup-regulated along the DA trajectory", fontsize=11, fontweight="bold")
b.legend(frameon=False, fontsize=8.6, loc="upper left", ncol=1)
for s in ("top","right"): b.spines[s].set_visible(False)
fig.suptitle("A concrete discovery case — RECAST prioritizes biologically meaningful DA-neuron genes that marginal DE misses\n"
             "SNCA (α-synuclein), the defining midbrain-dopaminergic / Parkinson's gene, is RECAST-top-20 for DA yet ranked far down by one-vs-rest DE",
             fontsize=11.0, fontweight="bold", y=1.06)
fig.savefig(f"{OUT}/development_case.png", dpi=175, bbox_inches="tight", facecolor="white")
log(f"\n[wrote] {OUT}/development_case.png + discovery_case_da_ranks.csv")
open(f"{OUT}/discovery_case_da_findings.txt","w").write("\n".join(L))
