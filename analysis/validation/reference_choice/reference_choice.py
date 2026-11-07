"""Reference choice (Fig. 4): the reference set decides the biological resolution of the explanation.

For each Zhao target subtype, vary the reference set and measure:
  - marker recovery   (recall@20 against curated subtype markers)
  - specificity       (within-lineage on-target ratio + panel AUROC: target against its siblings)
  - ranking stability (bootstrap Jaccard@20 of the RECAST top 20 under cell resampling)
  - overlap of the top 20 with the sibling-reference and the global-reference lists

References per target (masks over the full Zhao dataset):
  nearest   the single nearest sibling (pseudobulk correlation)
  sibling   the other subtypes of the lineage                  [RECAST's default]
  broad     other cells of the same broad lineage
  global    all other cells
  geometry  N embedding-kNN cells of the target centroid       [not biological, N = |siblings|]
  random    N random non-target cells                          [control, N = |siblings|]
  wrong     cells of an unrelated broad lineage                [incorrect on purpose]

Uses the SCimilarity attribution files (counts + labels).  Env: SC (GPU).  Out: validation/reference_choice/."""
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, sys, json
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/analysis/subtype_benchmark")
sys.path.insert(0, "/path/to/research/analysis")
import real_counts          # REAL integer counts; layers["counts"] here is lognorm (aliasing bug)
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)
import os
import contrastive_ig
from captum.attr import IntegratedGradients
from sklearn.metrics import roc_auc_score

# curated canonical subtype markers
CURATED = {
    "zhao_Mono": {"CD14+ Mo": ["CD14", "S100A8", "S100A9", "S100A12", "VCAN", "FCN1", "LYZ", "MNDA", "CSF3R", "S100A6"],
                  "CD16+ Mo": ["FCGR3A", "CDKN1C", "LST1", "MS4A7", "CX3CR1", "RHOC", "SIGLEC10", "IFITM3", "HES4", "APOBEC3A"],
                  "MerTK+ Mo": ["MERTK", "C1QA", "C1QB", "C1QC", "CLEC10A", "F13A1", "SELENOP", "CD163", "STAB1", "FOLR2"]},
    "zhao_CD8": {"CCR7+ CD8": ["CCR7", "SELL", "TCF7", "LEF1", "MAL", "NOSIP", "IL7R", "TRABD2A"],
                 "CRCR3+ CD8": ["CXCR3", "GZMK", "IL7R", "CD27", "GPR183", "LTB", "CD28", "SLAMF1"],
                 "CX3CR1+ CD8": ["CX3CR1", "FGFBP2", "GNLY", "GZMB", "GZMH", "PRF1", "NKG7", "FCGR3A", "KLRD1", "ZNF683"],
                 "CXCR6+ CD8": ["CXCR6", "ITGAE", "ZNF683", "CD69", "XCL1", "XCL2", "CD160", "KLRC1", "TRDC", "KLRF1"]},
    "zhao_CD4": {"CCR7+ CD4": ["CCR7", "SELL", "TCF7", "LEF1", "MAL", "ADTRP", "TSHZ2", "NOSIP"],
                 "CXCR3+ CD4": ["CXCR3", "GZMK", "GZMA", "CCL5", "NKG7", "KLRG1", "HOPX", "CST7"],
                 "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "RTKN2", "LAIR2", "TNFRSF18"]},
    "zhao_B": {"Naive B": ["IGHD", "TCL1A", "FCER2", "IL4R", "YBX3", "BACH2", "IGHM"],
               "TCL1A+ naive B": ["TCL1A", "IGHD", "FCER2", "PLPP5", "CD72", "IL4R"],
               "Classical mem B": ["CD27", "TNFRSF13B", "AIM2", "CD82", "S100A10", "COCH"],
               "CD11c+ mem B": ["ITGAX", "FCRL5", "TBX21", "ZEB2", "FGR", "NKG7", "SLC11A1"],
               "ZBTB32+ mem B": ["ZBTB32", "NKG7", "GZMA", "CCL5", "MT1F"],
               "MZB": ["CR2", "CD1C", "CD24", "PLD4", "RALGPS2", "DNASE1L3"]},
}

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/validation/reference_choice"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)
K = 20; BSTAB = 40; RNG = np.random.default_rng(0)

BROAD = {
    "T":    ["CCR7+ CD4", "CXCR3+ CD4", "Treg", "CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8", "MAIT"],
    "B":    ["CD11c+ mem B", "Classical mem B", "MZB", "Naive B", "TCL1A+ naive B", "ZBTB32+ mem B"],
    "Mono": ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo", "Macrophage", "CD1c+ DCs"],
}
# lineage tag -> (sibling states, own broad, WRONG broad)
LIN = {
    "zhao_Mono": (["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"], "Mono", "T"),
    "zhao_CD8":  (["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"], "T", "B"),
    "zhao_CD4":  (["CCR7+ CD4", "CXCR3+ CD4", "Treg"], "T", "B"),
    "zhao_B":    (["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"], "B", "Mono"),
}

print("loading zhao …", flush=True)
att = ad.read_h5ad(f"{ATLASES}/zhao_attribution_scimilarity.h5ad")
# att.X IS the correct lognorm (the embedding path was never affected). att.layers["counts"]
# is NOT counts -- lognorm_counts aliased it -- so pull real counts from the fullgene file.
Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
genes = np.asarray(att.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
lab = att.obs["label"].astype(str).to_numpy()
cnt = real_counts.load("zhao", att.obs_names)
print(f"real counts: {real_counts.summary(cnt)}", flush=True)
del att
print(f"zhao {Xln.shape}; embedding …", flush=True)
Z = contrastive_ig.embed(Xln)
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval(); ig = IntegratedGradients(mod, multiply_by_inputs=True)
tot_all = np.asarray(cnt.sum(0)).ravel().astype("float64")
# mean_lognorm normalizes every cell before pooling, so it cannot be read off a column sum. This
# script draws 40 bootstrap references per (target, reference) pair, i.e. ~4,500 centroids over a
# 60k-cell dataset -- transform once here and each centroid becomes a row-slice mean.
CENT = fc.Centroider(cnt)
print("lognorm cache built.", flush=True)
print("embedded.", flush=True)


def recast_rank(tgt_idx, ref_idx):
    """Full RECAST gene ranking (indices, best-first) for target-cell-set vs reference-cell-set."""
    u = Z[tgt_idx].mean(0) - Z[ref_idx].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
    Cc, Cr = CENT(tgt_idx), CENT(ref_idx)          # the paper's single centroid definition
    #                                                (mean_lognorm; see validation/recast_centroid.py)
    inp = torch.tensor(Cc[None], device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None], device=contrastive_ig.dev),), n_steps=50).detach().cpu().numpy().ravel()
    a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf)
    return np.argsort(-a)


def recall_at(top_genes, curated):
    mk = [g for g in curated if g in gi]
    return len(set(top_genes[:K]) & set(mk)) / len(mk) if mk else np.nan


def specificity(top_idx, in_lin, tgt_in_lin):
    """within-lineage on-target ratio + panel AUROC (target vs siblings)."""
    idx = list(top_idx[:K])
    Xl = Xln[in_lin]; ymask = tgt_in_lin
    ot = np.mean([Xl[ymask, j].mean() / (Xl[ymask, j].mean() + Xl[~ymask, j].mean() + 1e-12) for j in idx])
    try:
        au = roc_auc_score(ymask.astype(int), Xl[:, idx].mean(1))
    except Exception:
        au = np.nan
    return float(ot), float(au)


rows = []
genes_dump = {}
for tag, (sibs, broad, wrong) in LIN.items():
    lin_states = sibs
    in_lin = np.isin(lab, lin_states)                 # lineage cells (for within-lineage specificity)
    lin_idx_all = np.where(in_lin)[0]
    for t in sibs:
        tm = lab == t
        tgt_idx = np.where(tm)[0]
        if len(tgt_idx) < 20:
            print(f"  skip {tag}/{t} (<20 cells)", flush=True); continue
        others = sibs[:]; others.remove(t)
        tvec = Xln[tm].mean(0)
        nearest = max(others, key=lambda s: np.corrcoef(tvec, Xln[lab == s].mean(0))[0, 1]) if others else t
        nsib = int(in_lin.sum() - tm.sum())            # reference size to match for geometry/random
        # embedding-kNN geometry reference: nsib nearest non-target cells to target centroid
        cen = Z[tm].mean(0)
        d2 = ((Z - cen) ** 2).sum(1); d2[tm] = np.inf
        geo_idx = np.argsort(d2)[:max(nsib, 20)]
        nontgt = np.where(~tm)[0]
        rand_idx = RNG.choice(nontgt, size=max(nsib, 20), replace=False)
        REFS = {
            "nearest":  np.where(lab == nearest)[0],
            "sibling":  np.where(np.isin(lab, others))[0],
            "broad":    np.where(np.isin(lab, [s for s in BROAD[broad] if s != t]))[0],
            "global":   nontgt,
            "geometry": geo_idx,
            "random":   rand_idx,
            "wrong":    np.where(np.isin(lab, BROAD[wrong]))[0],
        }
        ranks = {}
        for rn, ridx in REFS.items():
            if len(ridx) < 5: continue
            rk = recast_rank(tgt_idx, ridx); ranks[rn] = rk
        # stability via cell bootstrap (resample target + reference cells)
        for rn, ridx in REFS.items():
            if rn not in ranks: continue
            pt = ranks[rn][:K]
            jac = []
            for _ in range(BSTAB):
                bt = tgt_idx[RNG.integers(0, len(tgt_idx), len(tgt_idx))]
                br = ridx[RNG.integers(0, len(ridx), len(ridx))]
                bk = recast_rank(bt, br)[:K]
                jac.append(len(set(bk) & set(pt)) / len(set(bk) | set(pt)))
            top_g = [genes[j] for j in ranks[rn][:K]]
            in_lin_tgt = (lab[in_lin] == t)
            ot, au = specificity(ranks[rn], in_lin, in_lin_tgt)
            rows.append(dict(lineage=tag, target=t, reference=rn, n_ref=int(len(REFS[rn])),
                             recall20=recall_at(top_g, CURATED[tag][t]), ontarget=ot, auroc=au,
                             stability_jac=float(np.mean(jac)), nearest=nearest,
                             top8="|".join(top_g[:8])))
        # identity(global) vs state(sibling) overlap of every reference's top-20
        if "sibling" in ranks and "global" in ranks:
            state_set = set(ranks["sibling"][:K]); id_set = set(ranks["global"][:K])
            for rn in ranks:
                st = set(ranks[rn][:K])
                for r in rows:
                    if r["lineage"] == tag and r["target"] == t and r["reference"] == rn:
                        r["overlap_state"] = len(st & state_set) / K
                        r["overlap_identity"] = len(st & id_set) / K
        genes_dump[f"{tag}|{t}"] = {rn: [genes[j] for j in ranks[rn][:K]] for rn in ranks}
        print(f"  [{tag}/{t}] done ({len(ranks)} refs)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/ref_robust_metrics{VTAG}.csv", index=False)
json.dump(genes_dump, open(f"{OUT}/ref_robust_genes{VTAG}.json", "w"), indent=1)

# ---- summary: mean over targets per reference ----
REF_ORDER = ["nearest", "sibling", "broad", "global", "geometry", "random", "wrong"]
agg = df.groupby("reference")[["recall20", "ontarget", "auroc", "stability_jac", "overlap_state", "overlap_identity"]].mean()
agg = agg.reindex([r for r in REF_ORDER if r in agg.index])
print("\n=== reference-robustness summary (mean over 16 target states, zhao) ===", flush=True)
print(agg.round(3).to_string(), flush=True)
agg.round(4).to_csv(f"{OUT}/ref_robust_summary{VTAG}.csv")
print(f"\n[wrote] {OUT}/ref_robust_metrics.csv, ref_robust_summary.csv, ref_robust_genes.json", flush=True)
