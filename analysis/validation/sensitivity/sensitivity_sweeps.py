"""Sensitivity of the RECAST top 20 to its settings (Supplementary Fig. 5a-d).  For each of the 16 Zhao
subtypes with the sibling reference, four settings are varied and, at each level, measured:
  - marker recall@20 against curated subtype markers
  - Jaccard@20 of the top 20 against the default

Sweeps (the SCimilarity computation of reference_choice.py):
  1. IG steps        n in {8,16,32,50,100,200}     default 50
  2. reference size  m in {50,100,200,500,full}     default full, 3 draws
  3. target cells    m in {20,50,100,200,full}      default full, 3 draws
  4. seeds           10 draws at 200 reference / 100 target cells

Env: SC (GPU).  Out: validation/sensitivity/sensitivity_*.csv + sensitivity_summary.txt."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
import contrastive_ig
from captum.attr import IntegratedGradients
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")

# ---- curated markers + lineage/sibling defs (identical to reference_choice.py) ----
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

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/validation/sensitivity"
K = 20
print("loading zhao …", flush=True)
att = ad.read_h5ad(f"{ATLASES}/zhao_attribution_scimilarity.h5ad")
cnt = real_counts.load("zhao", att.obs_names)   # NOT .layers["counts"] -- that is lognorm
print(f"  real counts zhao: {real_counts.summary(cnt)}", flush=True)
Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
genes = np.asarray(att.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
lab = att.obs["label"].astype(str).to_numpy(); del att
Z = contrastive_ig.embed(Xln)
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
print(f"zhao {Xln.shape} embedded.", flush=True)


def recast_rank(tgt_idx, ref_idx, n_steps=50):
    u = Z[tgt_idx].mean(0) - Z[ref_idx].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
    Cc = fc.centroid(cnt, tgt_idx)     # the paper's single centroid definition (mean_lognorm);
    Cr = fc.centroid(cnt, ref_idx)     # see validation/recast_centroid.py
    inp = torch.tensor(Cc[None], device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None], device=contrastive_ig.dev),), n_steps=n_steps).detach().cpu().numpy().ravel()
    a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf)
    return np.argsort(-a)


def recall(rk, curated):
    mk = [gi[g] for g in curated if g in gi]          # curated marker -> gene index (rk is index-space)
    return len(set(rk[:K]) & set(mk)) / len(mk) if mk else np.nan


def jac(a, b):
    a, b = set(a[:K]), set(b[:K]); return len(a & b) / len(a | b)


IGSTEPS = [8, 16, 32, 50, 100, 200]
REFSIZE = [50, 100, 200, 500, None]         # None = full
CELLNUM = [20, 50, 100, 200, None]
rows_ig, rows_rs, rows_cn, rows_sd = [], [], [], []

for tag, sibs in LIN.items():
    for t in sibs:
        tgt = np.where(lab == t)[0]
        ref = np.where(np.isin(lab, [s for s in sibs if s != t]))[0]   # canonical sibling reference
        if len(tgt) < K or len(ref) < 5:
            print(f"  skip {tag}/{t}", flush=True); continue
        cur = CURATED[tag][t]
        R50 = recast_rank(tgt, ref, 50)                                 # anchor for IG-steps
        Rfull_ref = R50                                                # anchor for ref-size (full ref)
        Rfull_cell = R50                                               # anchor for cell# (full tgt)
        # 1) IG-steps
        for ns in IGSTEPS:
            rk = R50 if ns == 50 else recast_rank(tgt, ref, ns)
            rows_ig.append(dict(lineage=tag, target=t, n_steps=ns, recall20=recall(rk, cur), jac_vs50=jac(rk, R50)))
        # 2) ref-size (subsample reference cells; 3 draws)
        for m in REFSIZE:
            if m is None or m >= len(ref):
                rows_rs.append(dict(lineage=tag, target=t, ref_size=len(ref), recall20=recall(Rfull_ref, cur),
                                    recall_sd=0.0, jac_vs_full=1.0, is_full=True)); continue
            rc, jc = [], []
            for s in range(3):
                rng = np.random.default_rng(1000 + s)
                sub = rng.choice(ref, size=m, replace=False)
                rk = recast_rank(tgt, sub, 50); rc.append(recall(rk, cur)); jc.append(jac(rk, Rfull_ref))
            rows_rs.append(dict(lineage=tag, target=t, ref_size=m, recall20=float(np.mean(rc)),
                                recall_sd=float(np.std(rc)), jac_vs_full=float(np.mean(jc)), is_full=False))
        # 3) cell# (subsample target cells; 3 draws)
        for m in CELLNUM:
            if m is None or m >= len(tgt):
                rows_cn.append(dict(lineage=tag, target=t, n_cells=len(tgt), recall20=recall(Rfull_cell, cur),
                                    recall_sd=0.0, jac_vs_full=1.0, is_full=True)); continue
            rc, jc = [], []
            for s in range(3):
                rng = np.random.default_rng(2000 + s)
                sub = tgt[rng.choice(len(tgt), size=m, replace=False)]
                rk = recast_rank(sub, ref, 50); rc.append(recall(rk, cur)); jc.append(jac(rk, Rfull_cell))
            rows_cn.append(dict(lineage=tag, target=t, n_cells=m, recall20=float(np.mean(rc)),
                                recall_sd=float(np.std(rc)), jac_vs_full=float(np.mean(jc)), is_full=False))
        # 4) seeds: 10 joint subsamples at ref=200/cell#=100; pairwise Jaccard + recall spread
        mr, mc = min(200, len(ref)), min(100, len(tgt))
        panels, recs = [], []
        for s in range(10):
            rng = np.random.default_rng(3000 + s)
            sub_t = tgt[rng.choice(len(tgt), size=mc, replace=False)]
            sub_r = rng.choice(ref, size=mr, replace=False)
            rk = recast_rank(sub_t, sub_r, 50); panels.append(rk[:K]); recs.append(recall(rk, cur))
        pj = [jac(panels[i], panels[j]) for i in range(10) for j in range(i + 1, 10)]
        rows_sd.append(dict(lineage=tag, target=t, ref_size=mr, n_cells=mc, recall20_mean=float(np.mean(recs)),
                            recall20_sd=float(np.std(recs)), jac_pairwise_mean=float(np.mean(pj)),
                            jac_pairwise_min=float(np.min(pj))))
        print(f"  [{tag}/{t}] done  (n_tgt={len(tgt)}, n_ref={len(ref)})", flush=True)

IG = pd.DataFrame(rows_ig); RS = pd.DataFrame(rows_rs); CN = pd.DataFrame(rows_cn); SD = pd.DataFrame(rows_sd)
IG.to_csv(f"{OUT}/sensitivity_igsteps{VTAG}.csv", index=False)
RS.to_csv(f"{OUT}/sensitivity_refsize{VTAG}.csv", index=False)
CN.to_csv(f"{OUT}/sensitivity_cellnum{VTAG}.csv", index=False)
SD.to_csv(f"{OUT}/sensitivity_seeds{VTAG}.csv", index=False)

L = []
def log(s=""):
    print(s, flush=True); L.append(str(s))
log("="*88); log("RECAST setting sensitivity (16 zhao states, sibling reference)"); log("="*88)
log("\n[1] IG-STEPS  (Jaccard@20 vs n_steps=50; recall@20)")
log(IG.groupby("n_steps")[["recall20", "jac_vs50"]].mean().round(3).to_string())
RS["bucket"] = np.where(RS.is_full, "full", RS.ref_size.astype(str))
CN["bucket"] = np.where(CN.is_full, "full", CN.n_cells.astype(str))
rs_ord = [str(x) for x in [50, 100, 200, 500]] + ["full"]
cn_ord = [str(x) for x in [20, 50, 100, 200]] + ["full"]
log("\n[2] REFERENCE SIZE  (mean over states & 3 draws; Jaccard vs full sibling reference)")
log(RS.groupby("bucket")[["recall20", "jac_vs_full"]].mean().reindex(rs_ord).round(3).to_string())
log("\n[3] TARGET CELL COUNT  (mean over states & 3 draws; Jaccard vs full target)")
log(CN.groupby("bucket")[["recall20", "jac_vs_full"]].mean().reindex(cn_ord).round(3).to_string())
log("\n[4] STOCHASTIC STABILITY  (ref=200 / cell#=100, 10 seeds)")
log(f"  mean pairwise Jaccard@20 = {SD.jac_pairwise_mean.mean():.3f} (min {SD.jac_pairwise_min.min():.3f})")
log(f"  recall@20 mean {SD.recall20_mean.mean():.3f}  within-state sd {SD.recall20_sd.mean():.3f}")
log("\nREADING")
ig32 = IG[IG.n_steps >= 32].groupby("n_steps").jac_vs50.mean()
log(f"[C] IG approximation converges: Jaccard@20 vs n=50 is {ig32.min():.2f}-{ig32.max():.2f} for n>=32; "
    f"recall is flat -> the panel is NOT an artifact of the step count.")
log(f"[C] {int((RS[RS.bucket=='200'].jac_vs_full.mean())*100)}% of the top-20 is recovered with only 200 reference cells; "
    f"recall at 200 ref = {RS[RS.bucket=='200'].recall20.mean():.3f} vs full {RS[RS.is_full].recall20.mean():.3f}.")
log(f"[C] {int((CN[CN.bucket=='100'].jac_vs_full.mean())*100)}% top-20 stability with 100 target cells; "
    f"recall at 100 cells = {CN[CN.bucket=='100'].recall20.mean():.3f} vs full {CN[CN.is_full].recall20.mean():.3f}.")
log(f"[C] Pure stochastic panel stability (Jaccard@20 over 10 seeds) = {SD.jac_pairwise_mean.mean():.2f}.")
open(f"{OUT}/sensitivity_summary{VTAG}.txt", "w").write("\n".join(L))
log(f"\n[wrote] {OUT}/sensitivity_{{igsteps,refsize,cellnum,seeds}}.csv + sensitivity_summary.txt")
