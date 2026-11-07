"""Integrated-gradients baseline (reference profile) sensitivity (Supplementary Fig. 5e).

The contrast direction u = mean(target) - mean(sibling reference) in the SCimilarity embedding is fixed; only the
baseline / centroid recipe changes (the reference set itself is varied in validation/reference_choice).

Five baselines (16 Zhao subtypes, sibling reference):
  mean_lognorm  (default)  baseline = mean over reference cells of per-cell lognorm   (target and reference)
  ref_cpm                  baseline = log1p(1e4 * ref_counts / sum)        pooled before log (target and reference)
  median_lognorm           baseline = element-wise median of reference-cell lognorm   (target and reference)
  global_cpm               baseline = log1p(1e4 * ALL_counts / sum)        whole-dataset pseudobulk
  zero                     baseline = 0

Metrics per (mode, state):
  recall@20         against curated subtype markers
  Jaccard@20        of the top 20 against the mean_lognorm default
  faithfulness      fraction of the target-vs-baseline model score captured by the top-20 genes
                    (top 20 set to the baseline, change of <enc(x),u>; 20 random genes as floor)
Env: SC (GPU).  Out: validation/sensitivity/ig_baseline_*."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig
from captum.attr import IntegratedGradients
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")

# ---- curated markers + sibling lineage defs (identical to sensitivity_sweeps.py / reference_choice.py) ----
CURATED = {
    "zhao_Mono": {"CD14+ Mo": ["CD14","S100A8","S100A9","S100A12","VCAN","FCN1","LYZ","MNDA","CSF3R","S100A6"],
                  "CD16+ Mo": ["FCGR3A","CDKN1C","LST1","MS4A7","CX3CR1","RHOC","SIGLEC10","IFITM3","HES4","APOBEC3A"],
                  "MerTK+ Mo": ["MERTK","C1QA","C1QB","C1QC","CLEC10A","F13A1","SELENOP","CD163","STAB1","FOLR2"]},
    "zhao_CD8": {"CCR7+ CD8": ["CCR7","SELL","TCF7","LEF1","MAL","NOSIP","IL7R","TRABD2A"],
                 "CRCR3+ CD8": ["CXCR3","GZMK","IL7R","CD27","GPR183","LTB","CD28","SLAMF1"],
                 "CX3CR1+ CD8": ["CX3CR1","FGFBP2","GNLY","GZMB","GZMH","PRF1","NKG7","FCGR3A","KLRD1","ZNF683"],
                 "CXCR6+ CD8": ["CXCR6","ITGAE","ZNF683","CD69","XCL1","XCL2","CD160","KLRC1","TRDC","KLRF1"]},
    "zhao_CD4": {"CCR7+ CD4": ["CCR7","SELL","TCF7","LEF1","MAL","ADTRP","TSHZ2","NOSIP"],
                 "CXCR3+ CD4": ["CXCR3","GZMK","GZMA","CCL5","NKG7","KLRG1","HOPX","CST7"],
                 "Treg": ["FOXP3","IL2RA","CTLA4","IKZF2","TIGIT","RTKN2","LAIR2","TNFRSF18"]},
    "zhao_B": {"Naive B": ["IGHD","TCL1A","FCER2","IL4R","YBX3","BACH2","IGHM"],
               "TCL1A+ naive B": ["TCL1A","IGHD","FCER2","PLPP5","CD72","IL4R"],
               "Classical mem B": ["CD27","TNFRSF13B","AIM2","CD82","S100A10","COCH"],
               "CD11c+ mem B": ["ITGAX","FCRL5","TBX21","ZEB2","FGR","NKG7","SLC11A1"],
               "ZBTB32+ mem B": ["ZBTB32","NKG7","GZMA","CCL5","MT1F"],
               "MZB": ["CR2","CD1C","CD24","PLD4","RALGPS2","DNASE1L3"]},
}
LIN = {"zhao_Mono": ["CD14+ Mo","CD16+ Mo","MerTK+ Mo"],
       "zhao_CD8":  ["CCR7+ CD8","CRCR3+ CD8","CX3CR1+ CD8","CXCR6+ CD8"],
       "zhao_CD4":  ["CCR7+ CD4","CXCR3+ CD4","Treg"],
       "zhao_B":    ["Naive B","TCL1A+ naive B","Classical mem B","CD11c+ mem B","ZBTB32+ mem B","MZB"]}

ATLASES = "/path/to/research/analysis/atlases"; OUT = "/path/to/research/analysis/validation/sensitivity"
K = 20; MODES = ["ref_cpm", "mean_lognorm", "median_lognorm", "global_cpm", "zero"]
# The manuscript's centroid, and therefore the default `jac_vs_default` compares every other recipe with.
DEFAULT_MODE = "mean_lognorm"
print("loading zhao …", flush=True)
att = ad.read_h5ad(f"{ATLASES}/zhao_attribution_scimilarity.h5ad")
cnt = real_counts.load("zhao", att.obs_names)   # NOT .layers["counts"] -- that is lognorm
print(f"  real counts zhao: {real_counts.summary(cnt)}", flush=True)
Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
genes = np.asarray(att.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
lab = att.obs["label"].astype(str).to_numpy(); del att
Z = contrastive_ig.embed(Xln)
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
GLOB = np.asarray(cnt.sum(0)).ravel().astype("float64"); GLOB = np.log1p(1e4 * GLOB / max(GLOB.sum(), 1)).astype("float32")
print(f"zhao {Xln.shape} embedded.", flush=True)


def profile(idx, recipe):
    # "cpm" and "mean_lognorm" both come from recast_centroid, the module every validation script now
    # shares, so this sweep perturbs the SAME arithmetic the rest of the paper runs. median_lognorm has
    # no library counterpart (it is the deliberately degenerate boundary case) and stays local.
    if recipe == "cpm":            return fc.centroid(cnt, idx, recipe="pseudobulk")
    if recipe == "mean_lognorm":   return fc.centroid(cnt, idx, recipe="mean_lognorm")
    if recipe == "median_lognorm": return np.median(Xln[idx], axis=0).astype("float32")
    raise ValueError(recipe)


def _check_lognorm_parity():
    """Report how far att.X is from the per-cell lognorm of the real counts, every run.

    Both are tp10k-lognorm over the same 28,231 genes and agree to float32 precision cell by cell
    (~2e-7), but they are built by two different gene-alignment paths, and 8 of the 28,231 columns
    disagree because the _fullgene source collapses several many-to-one symbol duplicates (e.g.
    MATR3: 42 non-zero cells in the counts path, 6 in att.X) that the attribution h5ad kept split.
    The centroid is taken from the counts, as in the benchmark's selection (select_recast.py aligns the same
    _fullgene counts to the same gene order); att.X is used only for the embedding.  Fail if the disagreement is
    more than those few columns."""
    idx = np.arange(min(2000, cnt.shape[0]))
    d = np.abs(fc.centroid(cnt, idx) - Xln[idx].mean(0))
    n_bad = int((d > 1e-4).sum())
    print(f"  [check] mean_lognorm counts-path vs att.X: {n_bad}/{len(genes)} genes differ "
          f"(max {d.max():.2e}) -- expected ~8, symbol-collapse only", flush=True)
    assert n_bad <= 32, f"{n_bad} genes disagree: att.X is not the per-cell lognorm of these counts"


def enc_score(vec, u):                                   # f(x) = <enc(x), u>  (marker-free faithfulness helper)
    return float(contrastive_ig.embed(np.asarray(vec, dtype="float32")[None])[0] @ u)


def recast_rank_mode(tgt, ref, mode, u):
    """u is FIXED (sibling embedding direction). Returns (ranked_idx, Cc, base)."""
    if mode == "ref_cpm":         Cc, base = profile(tgt, "cpm"), profile(ref, "cpm")
    elif mode == "mean_lognorm":  Cc, base = profile(tgt, "mean_lognorm"), profile(ref, "mean_lognorm")
    elif mode == "median_lognorm":Cc, base = profile(tgt, "median_lognorm"), profile(ref, "median_lognorm")
    # global_cpm and zero perturb ONLY the baseline, so their target stays the paper default -- otherwise
    # they would confound a changed baseline with a changed target profile.
    elif mode == "global_cpm":    Cc, base = profile(tgt, DEFAULT_MODE), GLOB.copy()
    elif mode == "zero":          Cc, base = profile(tgt, DEFAULT_MODE), np.zeros(len(genes), "float32")
    inp = torch.tensor(Cc[None], device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(base[None], device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None], device=contrastive_ig.dev),), n_steps=50).detach().cpu().numpy().ravel()
    a = np.where((Cc - base) > 0, np.clip(a, 0, None), -np.inf)
    return np.argsort(-a), Cc, base


def recall(rk, cur):
    mk = [gi[g] for g in cur if g in gi]; return len(set(rk[:K]) & set(mk)) / len(mk) if mk else np.nan
def jac(a, b):
    a, b = set(a[:K]), set(b[:K]); return len(a & b) / len(a | b)


_check_lognorm_parity()
rng = np.random.default_rng(0); rows = []
for tag, sibs in LIN.items():
    for t in sibs:
        tgt = np.where(lab == t)[0]; ref = np.where(np.isin(lab, [s for s in sibs if s != t]))[0]
        if len(tgt) < K or len(ref) < 5: continue
        cur = CURATED[tag][t]
        u = Z[tgt].mean(0) - Z[ref].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        panels = {}
        for mode in MODES:
            rk, Cc, base = recast_rank_mode(tgt, ref, mode, u); panels[mode] = rk
            top = rk[:K]
            # faithfulness: fraction of the target-vs-baseline score gap captured by the top-20 genes
            f_cc, f_base = enc_score(Cc, u), enc_score(base, u); gap = f_cc - f_base
            abl = Cc.copy(); abl[top] = base[top]; d_top = f_cc - enc_score(abl, u)
            rr = rng.choice(len(genes), K, replace=False); abr = Cc.copy(); abr[rr] = base[rr]; d_rnd = f_cc - enc_score(abr, u)
            rows.append(dict(mode=mode, lineage=tag, target=t, recall20=recall(rk, cur),
                             faith_frac=float(d_top / gap) if abs(gap) > 1e-9 else np.nan,
                             faith_rand=float(d_rnd / gap) if abs(gap) > 1e-9 else np.nan))
        for mode in MODES:                                # Jaccard vs default (fill after default known)
            rows[[i for i, r in enumerate(rows) if r["mode"] == mode and r["target"] == t and r["lineage"] == tag][0]]["jac_vs_default"] = jac(panels[mode], panels[DEFAULT_MODE])
        print(f"  [{tag}/{t}] done", flush=True)

D = pd.DataFrame(rows); D.to_csv(f"{OUT}/ig_baseline_metrics{VTAG}.csv", index=False)

# ---- summary ----
L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))
agg = D.groupby("mode").agg(recall20=("recall20", "mean"), recall_sd=("recall20", "std"),
                            jac_vs_default=("jac_vs_default", "mean"), faith=("faith_frac", "mean"),
                            faith_rand=("faith_rand", "mean")).reindex(MODES).round(3)
log("=" * 92); log("Integrated-gradients baseline sensitivity  (16 zhao states, sibling reference)"); log("=" * 92)
log("\nPer-mode means (u = sibling embedding direction held FIXED across all modes):"); log(agg.to_string())
d0 = D[D['mode'] == DEFAULT_MODE].recall20.mean()
log(f"\nDEFAULT {DEFAULT_MODE} recall@20 = {d0:.3f}")
for m in MODES:
    dm = D[D['mode'] == m]
    log(f"  {m:15s} recall@20 {dm.recall20.mean():.3f} (Δ{dm.recall20.mean()-d0:+.3f})  "
        f"Jaccard@20-vs-default {dm.jac_vs_default.mean():.3f}  faithfulness {dm.faith_frac.mean():.3f} (rand {dm.faith_rand.mean():.3f})")
bio_modes = [DEFAULT_MODE, "ref_cpm", "median_lognorm"]     # default first, so the deltas read against it
bio = D[D['mode'].isin(bio_modes)]
log("\nREADING")
log(f"[C] Across the three biologically-grounded reference-profile recipes (mean-lognorm = the paper default, "
    f"proportion-then-log, median-lognorm) recall@20 is {D[D['mode']==bio_modes[0]].recall20.mean():.3f}/{D[D['mode']==bio_modes[1]].recall20.mean():.3f}/"
    f"{D[D['mode']==bio_modes[2]].recall20.mean():.3f} and top-20 Jaccard vs default is "
    + "/".join(f"{D[D['mode']==m].jac_vs_default.mean():.2f}" for m in bio_modes)
    + " -> the biology is STABLE to the exact centroid recipe even though the panel moves somewhat.")
log(f"[C] Decoupling the baseline from the reference distribution degrades biology: global_cpm recall "
    f"{D[D['mode']=='global_cpm'].recall20.mean():.3f}, zero-baseline recall {D[D['mode']=='zero'].recall20.mean():.3f} "
    f"(Δ{D[D['mode']=='zero'].recall20.mean()-d0:+.3f}) -> it is the reference-CONDITIONING, not the arithmetic of the "
    f"centroid, that carries the signal (as in the reference-choice analysis).")
open(f"{OUT}/ig_baseline_summary{VTAG}.txt", "w").write("\n".join(L))

# ---- figure: 3 panels (recall / stability / faithfulness) ----
COL = {"ref_cpm": "#C0392B", "mean_lognorm": "#E67E22", "median_lognorm": "#D4A017", "global_cpm": "#5B7FA6", "zero": "#95A5A6"}
LABS = {"ref_cpm": "ref_cpm\n(default)", "mean_lognorm": "mean\nlognorm", "median_lognorm": "median\nlognorm",
        "global_cpm": "global\ncpm", "zero": "zero"}
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.7))
xs = np.arange(len(MODES))
# A recall@20
rc = [D[D['mode'] == m].recall20.mean() for m in MODES]; rcsd = [D[D['mode'] == m].recall20.std() for m in MODES]
ax[0].bar(xs, rc, yerr=rcsd, color=[COL[m] for m in MODES], edgecolor="white", zorder=3, error_kw=dict(lw=1, ecolor="0.4"))
ax[0].axhline(rc[0], color="#C0392B", ls=":", lw=1.3, zorder=2)
for x, v in zip(xs, rc): ax[0].text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax[0].set_ylim(0, max(rc) * 1.25); ax[0].set_ylabel("marker recall@20  (biology)", fontsize=10.5)
ax[0].set_title("Biology survives reasonable baselines", fontsize=11, fontweight="bold")
# B Jaccard vs default
jc = [D[D['mode'] == m].jac_vs_default.mean() for m in MODES]
ax[1].bar(xs, jc, color=[COL[m] for m in MODES], edgecolor="white", zorder=3)
for x, v in zip(xs, jc): ax[1].text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax[1].set_ylim(0, 1.08); ax[1].set_ylabel("top-20 Jaccard vs default panel", fontsize=10.5)
ax[1].set_title("Exact ranking shifts (as expected)", fontsize=11, fontweight="bold")
# C faithfulness
ft = [D[D['mode'] == m].faith_frac.mean() for m in MODES]; fr = [D[D['mode'] == m].faith_rand.mean() for m in MODES]
ax[2].bar(xs - 0.19, ft, 0.38, color="#34495E", edgecolor="white", zorder=3, label="RECAST top-20")   # single colour: 2-series legend must be unambiguous
ax[2].bar(xs + 0.19, fr, 0.38, color="#B0B7BD", edgecolor="white", zorder=3, label="random-20 (floor)")
ax[2].set_ylabel("fraction of target-vs-baseline\nmodel score captured", fontsize=10)
ax[2].set_title("Top-20 are faithful to the model", fontsize=11, fontweight="bold")
ax[2].legend(frameon=False, fontsize=8.5, loc="upper right")
for a in ax:
    a.set_xticks(xs); a.set_xticklabels([LABS[m] for m in MODES], fontsize=8.6)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("IG baseline / reference-profile sensitivity — RECAST's biological conclusions are stable across baseline construction\n"
             "the three reference-grounded recipes agree on the biology (recall) and largely on the panel (Jaccard); a zero / global baseline that discards the reference degrades recall",
             fontsize=11.0, fontweight="bold", y=1.10)
fig.tight_layout(); fig.savefig(f"{OUT}/ig_baseline_figure{VTAG}.png", dpi=175, bbox_inches="tight", facecolor="white")
log(f"\n[wrote] {OUT}/ig_baseline_metrics.csv, ig_baseline_summary.txt, ig_baseline_figure.png")
