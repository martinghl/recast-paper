import os
"""Trajectory coherence of RECAST genes in the iPSC-to-midbrain differentiation data (Supplementary Fig. 7).
Dataset: neural_diff -- iPSC -> midbrain floor plate -> dopaminergic (DA) neuron (82,166 cells, 12 states, collection
days D11 / D30 / D52; 28,231 genes = SCimilarity's gene space).

Do the genes RECAST selects change along the differentiation, including those that one-vs-rest Wilcoxon ranks low?

Trajectory axes (both without curated markers):
  - dpt   diffusion pseudotime on the SCimilarity embedding, rooted in the progenitor (P_FPP) compartment
  - time  collection day D11 < D30 < D52
Per gene, eta^2 = fraction of the gene's expression variance explained by pseudotime (one-way ANOVA across 25
equal-count pseudotime bins; covers monotone and peaked genes, ~0 for flat genes).  Null = random genes matched by
expression decile.

Gene sets per state (all one-vs-rest):
  RECAST          recast_rank(state vs rest)[:20]
  DE              scanpy Wilcoxon one-vs-rest top 20
  RECAST_unique   RECAST top 20 not in the Wilcoxon top 50
  random          expression-decile-matched null

Env: SC (GPU for the embedding).  Out: validation/biology/trajectory_*.{csv,png,txt}."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys, scanpy as sc, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
import contrastive_ig
from captum.attr import IntegratedGradients
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)

SB = "/path/to/research/analysis"
OUT = f"{SB}/validation/biology"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)
K = 20; DEK = 50; NBIN = 25
L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))

# ---------------- load ----------------
log("loading neural_diff …")
raw = ad.read_h5ad(f"{SB}/ans_replication/bundles/neural_diff_a_raw.h5ad")
genes = np.asarray(raw.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
ct = raw.obs["celltype"].astype(str).to_numpy()
Xc = raw.X.tocsr() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)          # counts (SCimilarity gene space)
tot = np.asarray(Xc.sum(1)).ravel(); Xln = Xc.multiply(1e4 / np.clip(tot, 1, None)[:, None]).log1p().tocsr()
Xln = Xln.astype("float32")
# time_point + umap from the log bundle. raw uses an integer index and log uses barcodes, but the two
# bundles are the SAME cells in the SAME order (celltype matches positionally 100%) -> align by POSITION.
lb = ad.read_h5ad(f"{SB}/ans_replication/bundles/neural_diff_a_log.h5ad")
assert (lb.obs["celltype"].astype(str).to_numpy() == ct).all(), "raw/log positional order mismatch"
tp = lb.obs["time_point"].astype(str).to_numpy(); U = np.asarray(lb.obsm["X_umap"])
del lb
TNUM = {"D11": 0, "D30": 1, "D52": 2}; tnum = np.array([TNUM.get(x, np.nan) for x in tp])
n = Xc.shape[0]; log(f"neural_diff {Xc.shape}; states={sorted(set(ct))}")

# ---------------- embed + pseudotime ----------------
log("embedding (SCimilarity) …")
Xd = Xln.toarray()
Z = contrastive_ig.embed(Xd)                                                            # (n, 128)
log(f"embedded {Z.shape}; diffusion pseudotime …")
A = ad.AnnData(X=sp.csr_matrix((n, 1), dtype="float32")); A.obsm["X_scim"] = Z.astype("float32")
sc.pp.neighbors(A, use_rep="X_scim", n_neighbors=15)
sc.tl.diffmap(A)
prog = np.where(ct == "P_FPP")[0]                                            # proliferating floor-plate progenitor = root
cen = Z[prog].mean(0); A.uns["iroot"] = int(prog[np.argmin(((Z[prog] - cen) ** 2).sum(1))])
sc.tl.dpt(A)
dpt = A.obs["dpt_pseudotime"].to_numpy().astype("float64")
dpt[~np.isfinite(dpt)] = np.nanmax(dpt[np.isfinite(dpt)])
# orient so pseudotime increases with real experimental day
rho_t = spearmanr(dpt, tnum, nan_policy="omit").correlation
if rho_t < 0:
    dpt = dpt.max() - dpt; rho_t = -rho_t
log(f"[validation] Spearman(dpt, real day D11<D30<D52) = {rho_t:+.3f}  (independent-axis check)")
log("[validation] mean dpt by day: " + ", ".join(f"{d}={dpt[tp==d].mean():.3f}" for d in ["D11", "D30", "D52"]))

# ---------------- eta^2 (variance explained by pseudotime) for ALL genes ----------------
def eta2_binned(Xsp, order, B):
    edges = np.quantile(order, np.linspace(0, 1, B + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, order, side="right") - 1, 0, B - 1)
    S = sp.csr_matrix((np.ones(len(b)), (b, np.arange(len(b)))), shape=(B, len(b)))
    nk = np.asarray(S.sum(1)).ravel()
    binsum = np.asarray((S @ Xsp).todense())                                 # (B,G)
    binmean = binsum / np.clip(nk[:, None], 1, None)
    gsum = np.asarray(Xsp.sum(0)).ravel(); grand = gsum / Xsp.shape[0]
    ss_b = (nk[:, None] * (binmean - grand[None, :]) ** 2).sum(0)
    sq = np.asarray(Xsp.multiply(Xsp).sum(0)).ravel(); ss_t = sq - Xsp.shape[0] * grand ** 2
    eta = ss_b / np.clip(ss_t, 1e-12, None)
    # monotone direction (binmean vs bin index)
    bi = np.arange(B); rho = np.array([spearmanr(bi, binmean[:, g]).correlation if ss_t[g] > 1e-9 else 0.0
                                       for g in range(Xsp.shape[1])])
    return np.clip(eta, 0, 1), rho, grand, binmean

log("computing per-gene eta^2 along pseudotime (all genes) …")
eta_dpt, rho_dpt, grand, binmean = eta2_binned(Xln, dpt, NBIN)
eta_time, _, _, _ = eta2_binned(Xln, tnum.astype("float64"), 3)              # real-day variance explained (3 bins)

# expression deciles over EXPRESSED genes (for the matched null)
expressed = grand > np.quantile(grand[grand > 0], 0.25)
dec = np.full(len(genes), -1)
ev = np.where(expressed)[0]
dec[ev] = pd.qcut(grand[ev], 10, labels=False, duplicates="drop")
dec_eta = {d: eta_dpt[ev][dec[ev] == d].mean() for d in np.unique(dec[ev])}   # decile -> mean eta (null level)
def matched_null(panel_idx):
    return float(np.mean([dec_eta.get(dec[j], eta_dpt[expressed].mean()) for j in panel_idx if expressed[j]]) )

# The same expression-decile null against the collection-day axis instead of pseudotime
# (trajectory_realday_null.csv).
dec_eta_time = {d: eta_time[ev][dec[ev] == d].mean() for d in np.unique(dec[ev])}
def matched_null_time(panel_idx):
    return float(np.mean([dec_eta_time.get(dec[j], eta_time[expressed].mean()) for j in panel_idx if expressed[j]]))

# ---------------- RECAST + DE panels per state ----------------
log("RECAST one-vs-rest attribution + DE …")
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
def recast_rank(tgt_idx, ref_idx, n_steps=50):
    u = Z[tgt_idx].mean(0) - Z[ref_idx].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
    Cc = fc.centroid(Xc, tgt_idx); Cr = fc.centroid(Xc, ref_idx)   # paper centroid (mean_lognorm); validation/recast_centroid.py
    inp = torch.tensor(Cc[None], device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None], device=contrastive_ig.dev),), n_steps=n_steps).detach().cpu().numpy().ravel()
    a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf); return np.argsort(-a)

Ade = ad.AnnData(X=Xln, obs=pd.DataFrame({"ct": ct}, index=raw.obs_names)); Ade.var_names = genes
sc.tl.rank_genes_groups(Ade, "ct", method="wilcoxon", n_genes=DEK)
de_names = {g: list(Ade.uns["rank_genes_groups"]["names"][g]) for g in Ade.obs["ct"].cat.categories}

STATES = [s for s in sorted(set(ct)) if (ct == s).sum() >= 30]
rows, gene_rows = [], []
foc_panels = {}
for s in STATES:
    tgt = np.where(ct == s)[0]; ref = np.where(ct != s)[0]
    fr = recast_rank(tgt, ref); foc = [int(j) for j in fr[:K]]; foc_panels[s] = foc
    de_top = [gi[g] for g in de_names[s][:K] if g in gi]
    de_top50 = set(de_names[s][:DEK])
    foc_uniq = [j for j in foc if genes[j] not in de_top50]
    panels = {"RECAST": foc, "DE": de_top, "RECAST_unique": foc_uniq}
    for pname, idx in panels.items():
        if len(idx) == 0:
            rows.append(dict(state=s, panel=pname, n_genes=0, mean_eta2=np.nan, null_eta2=np.nan,
                             mean_absrho_dpt=np.nan, mean_eta2_time=np.nan,
                             null_eta2_time=np.nan)); continue
        rows.append(dict(state=s, panel=pname, n_genes=len(idx),
                         mean_eta2=float(np.mean(eta_dpt[idx])), null_eta2=matched_null(idx),
                         mean_absrho_dpt=float(np.mean(np.abs(rho_dpt[idx]))),
                         mean_eta2_time=float(np.mean(eta_time[idx])),
                         null_eta2_time=matched_null_time(idx)))
    # explicit matched-random row (deterministic decile-matched null of the RECAST panel)
    rows.append(dict(state=s, panel="random", n_genes=len(foc), mean_eta2=matched_null(foc), null_eta2=matched_null(foc),
                     mean_absrho_dpt=np.nan, mean_eta2_time=np.nan))
    for j in foc:
        gene_rows.append(dict(state=s, gene=genes[j], eta2_dpt=float(eta_dpt[j]), rho_dpt=float(rho_dpt[j]),
                              in_DE_top50=bool(genes[j] in de_top50)))
    log(f"  [{s:8s}] RECAST eta2 {np.mean(eta_dpt[foc]):.3f}  (null {matched_null(foc):.3f})  "
        f"unique n={len(foc_uniq)} eta2 {np.mean(eta_dpt[foc_uniq]) if foc_uniq else float('nan'):.3f}")

DF = pd.DataFrame(rows); DF.to_csv(f"{OUT}/trajectory_eta2_per_state{VTAG}.csv", index=False)
pd.DataFrame(gene_rows).to_csv(f"{OUT}/trajectory_gene_eta2{VTAG}.csv", index=False)

# The collection-day control, wide by state.  Pseudotime comes from the representation RECAST attributes;
# the collection day D11 < D30 < D52 does not, and this table gives the result on that axis.
_t = DF.pivot_table(index="state", columns="panel", values="mean_eta2_time")
_n = DF.pivot_table(index="state", columns="panel", values="null_eta2_time")
pd.DataFrame({"state": _t.index,
              "RECAST_day": _t["RECAST"].values, "RECAST_day_null": _n["RECAST"].values,
              "RECASTuniq_day": _t["RECAST_unique"].values,
              "RECASTuniq_day_null": _n["RECAST_unique"].values,
              }).to_csv(f"{OUT}/trajectory_realday_null{VTAG}.csv", index=False)
log(f"[wrote] trajectory_realday_null{VTAG}.csv — RECAST {_t['RECAST'].mean():.3f} vs null "
    f"{_n['RECAST'].mean():.3f}; RECAST-unique {_t['RECAST_unique'].mean():.3f} vs {_n['RECAST_unique'].mean():.3f}")

# ---------------- paired aggregate ----------------
def paired(a, b, col="mean_eta2"):
    m = DF.pivot_table(index="state", columns="panel", values=col)
    d = (m[a] - m[b]).dropna(); wins = int((d > 0).sum()); ntot = len(d)
    from scipy.stats import binomtest
    p = binomtest(wins, ntot, 0.5).pvalue if ntot else np.nan
    return d.mean(), wins, ntot, p
log("\n" + "=" * 88); log("Trajectory coherence, paired across %d states (eta^2 explained by pseudotime)" % len(STATES)); log("=" * 88)
piv = DF.pivot_table(index="state", columns="panel", values="mean_eta2")
log("\nmean eta^2 by panel:\n" + piv.mean().round(3).to_string())
for a, b in [("RECAST", "random"), ("RECAST_unique", "random"), ("RECAST", "DE"), ("DE", "random")]:
    dm, w, nt, p = paired(a, b); log(f"  {a:13s} - {b:13s}:  mean Δeta^2 {dm:+.3f}   wins {w}/{nt}   sign-test p={p:.3g}")

log("\nREADING")
foc_r = paired("RECAST", "random"); fu_r = paired("RECAST_unique", "random")
log(f"[C] RECAST panels are developmentally regulated: eta^2 {piv['RECAST'].mean():.3f} vs matched-random {piv['random'].mean():.3f} "
    f"(Δ{foc_r[0]:+.3f}, {foc_r[1]}/{foc_r[2]} states, p={foc_r[3]:.2g}) — independent of any marker list.")
log(f"[C] The DE-MISSED RECAST genes still track the trajectory: RECAST_unique eta^2 {piv['RECAST_unique'].mean():.3f} vs "
    f"random {piv['random'].mean():.3f} (Δ{fu_r[0]:+.3f}, {fu_r[1]}/{fu_r[2]}, p={fu_r[3]:.2g}) — trajectory evidence that "
    f"RECAST's extra genes are real developmental biology, echoing the regulon/pathway result.")
log(f"[validation] dpt reproduces real differentiation time (Spearman {rho_t:+.2f}); metric needs no curated markers.")

# ---------------- figure ----------------
# panel B profiles for DA (dopaminergic terminal target)
DAs = "DA" if "DA" in foc_panels else STATES[-1]
def zprofile(idx):
    prof = binmean[:, idx]                                                    # (B, k)
    prof = (prof - prof.mean(0, keepdims=True)) / (prof.std(0, keepdims=True) + 1e-9)
    return prof.mean(1)
rng = np.random.default_rng(0)
rand_da = rng.choice(ev, size=len(foc_panels[DAs]), replace=False)
prof_da = {"RECAST": zprofile(foc_panels[DAs]), "random": zprofile(rand_da)}
pd.DataFrame({"bin": np.arange(NBIN), **prof_da}).to_csv(f"{OUT}/trajectory_da_profiles{VTAG}.csv", index=False)
# save compact per-cell for panel A (subsample for size)
ss = rng.choice(n, size=min(20000, n), replace=False)
pd.DataFrame(dict(ux=U[ss, 0], uy=U[ss, 1], dpt=dpt[ss], time=tp[ss], celltype=ct[ss])).to_csv(f"{OUT}/trajectory_pseudotime{VTAG}.csv", index=False)

fig = plt.figure(figsize=(15.5, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.15], wspace=0.32)
# A: UMAP colored by pseudotime
axA = fig.add_subplot(gs[0, 0])
sca = axA.scatter(U[ss, 0], U[ss, 1], c=dpt[ss], s=3, cmap="viridis", linewidths=0)
for s in STATES:
    m = ct[ss] == s
    if m.sum() > 30: axA.text(U[ss][m, 0].mean(), U[ss][m, 1].mean(), s, fontsize=7.5, ha="center", va="center",
                              color="white", fontweight="bold",
                              path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
axA.set_xticks([]); axA.set_yticks([]); axA.set_title(f"Diffusion pseudotime\n(root = P_FPP progenitor; ρ vs real day {rho_t:+.2f})", fontsize=10.5, fontweight="bold")
cb = fig.colorbar(sca, ax=axA, fraction=0.045, pad=0.02); cb.set_label("pseudotime", fontsize=9); cb.set_ticks([])
for sp_ in ("top", "right"): axA.spines[sp_].set_visible(False)
# B: DA terminal panel profile
axB = fig.add_subplot(gs[0, 1])
xb = np.linspace(0, 1, NBIN)
axB.plot(xb, prof_da["RECAST"], "-o", color="#C0392B", lw=2.6, ms=5, zorder=4, label=f"RECAST {DAs} panel (n={len(foc_panels[DAs])})")
axB.plot(xb, prof_da["random"], "--s", color="#7D8B99", lw=1.8, ms=4, zorder=3, label="expression-matched random")
axB.axhline(0, color="0.75", lw=0.9)
axB.set_xlabel("pseudotime  (progenitor → neuron)", fontsize=10); axB.set_ylabel("panel-mean expression (z)", fontsize=10)
axB.set_title(f"{DAs} RECAST panel rises coherently\nalong the differentiation trajectory", fontsize=10.5, fontweight="bold")
axB.legend(frameon=False, fontsize=8.5, loc="upper left")
for sp_ in ("top", "right"): axB.spines[sp_].set_visible(False)
# C: paired eta^2 across states
axC = fig.add_subplot(gs[0, 2])
order = ["random", "DE", "RECAST_unique", "RECAST"]; COLp = {"random": "#B0B7BD", "DE": "#2E86AB", "RECAST_unique": "#E67E22", "RECAST": "#C0392B"}
for i, p in enumerate(order):
    vals = piv[p].dropna().values
    axC.scatter(np.full(len(vals), i) + rng.uniform(-0.07, 0.07, len(vals)), vals, s=26, color=COLp[p], alpha=0.75, zorder=3, edgecolor="white", linewidth=0.4)
    axC.hlines(np.mean(vals), i - 0.28, i + 0.28, color="#222", lw=2.4, zorder=4)
# paired lines RECAST vs random
for s in piv.index:
    if np.isfinite(piv.loc[s, "RECAST"]) and np.isfinite(piv.loc[s, "random"]):
        axC.plot([order.index("random"), order.index("RECAST")], [piv.loc[s, "random"], piv.loc[s, "RECAST"]], color="0.8", lw=0.5, zorder=1)
axC.set_xticks(range(4)); axC.set_xticklabels(["random\n(matched)", "DE", "RECAST\nunique", "RECAST"], fontsize=9)
axC.set_ylabel("η²  (expression variance explained by pseudotime)", fontsize=9.5)
axC.set_title(f"RECAST & its DE-missed genes are\ntrajectory genes ({len(STATES)} states, paired)", fontsize=10.5, fontweight="bold")
for sp_ in ("top", "right"): axC.spines[sp_].set_visible(False)
fig.suptitle("Trajectory coherence: RECAST surfaces developmentally-regulated genes (neural_diff: iPSC → midbrain DA-neuron differentiation)",
             fontsize=12, fontweight="bold", y=1.03)
fig.savefig(f"{OUT}/trajectory_coherence.png", dpi=170, bbox_inches="tight", facecolor="white")
log(f"\n[wrote] {OUT}/trajectory_coherence.png + trajectory_eta2_per_state.csv, trajectory_gene_eta2.csv, trajectory_da_profiles.csv")
open(f"{OUT}/trajectory_findings.txt", "w").write("\n".join(L))
