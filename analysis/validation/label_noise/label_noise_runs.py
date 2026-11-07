"""RECAST under perturbed subtype labels (Supplementary Fig. 5f-h): boundary-weighted and random
relabelling.

Cells and embeddings are fixed; only the labels RECAST sees change, so each condition is a new mask and one 50-step
integrated-gradients run per subtype.  Evaluation always uses the clean labels and the eps = 0 run.

Usage:  python label_noise_runs.py [lineage]      (default tf_cd8_dom; any working directory)
Env: SC (GPU).  Outputs in this directory:
  ps_recast_<lin>.csv          read-outs per (state, mode, eps, seed)
  ps_genes_<lin>.json          top 50 per condition
  ps_labels_<lin>/             one label file per condition (lineage cells, in h5ad order), for the Wilcoxon arm
  ps_export_<lin>.npz          sparse lineage counts + genes + clean labels (input of select_wilcoxon.py)"""
import os, sys, json
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch
from collections import Counter
from scipy.stats import spearmanr

sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
import contrastive_ig
from captum.attr import IntegratedGradients
import sys as _sys; _sys.path.insert(0, "/path/to/research/analysis")
import os as _os, real_counts   # REAL counts: the h5ad layers["counts"] is lognorm (aliasing bug)
sys.path.insert(0, "/path/to/research/analysis/validation")
import recast_centroid as fc  # the paper's ONE centroid definition (mean_lognorm)
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")

SB = "/path/to/research/analysis"
ATLASES = f"{SB}/atlases"
FSB = f"{SB}/subtype_benchmark"
OUT = f"{SB}/validation/label_noise"
K = 20
EPS = [0.025, 0.05, 0.10, 0.20, 0.40]
SEEDS = [0, 1, 2]

# lineage -> (attribution h5ad stem, bench label file, sibling states)
LINEAGES = {
    "tf_cd8_dom": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                   ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells",
                    "Tem/Trm cytotoxic T cells", "gdT"]),
    "tf_b_yosh": ("yoshida", "bench_labels_yoshida.txt",
                  ["Naive B cells", "Non-switched memory B cells",
                   "Switched memory B cells", "Memory B cells"]),
    "tf_mono_steph": ("stephenson", "bench_labels_stephenson.txt",
                      ["Classical monocytes", "Non-classical monocytes"]),
    "fs_lung_hlca": ("hlca_lung", "bench_labels_hlca_lung.txt",
                     ["AT1", "AT2", "basal", "ciliated", "club", "ionocyte"]),
}

LIN = sys.argv[1] if len(sys.argv) > 1 else "tf_cd8_dom"
stem, labfile, STATES = LINEAGES[LIN]

# ---------------- load lineage subset ----------------
print(f"[{LIN}] loading {stem} …", flush=True)
att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
if "label" in att.obs:
    lab_all = att.obs["label"].astype(str).to_numpy()
else:
    lab_all = np.array([l.rstrip("\n") for l in open(f"{ATLASES}/{labfile}")])
    assert len(lab_all) == att.n_obs, f"label file length {len(lab_all)} != n_obs {att.n_obs}"
keep = np.isin(lab_all, STATES)
sub = att[keep]
Xln = (sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)).astype("float32")
cnt = real_counts.load(stem, sub.obs_names)   # NOT .layers["counts"] -- that is lognorm
print(f"  real counts {stem}: {real_counts.summary(cnt)}", flush=True)
genes = np.asarray(sub.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
lab0 = lab_all[keep]                                    # CLEAN labels (never shown to methods at eps>0)
del att, sub
print(f"[{LIN}] {Xln.shape[0]} cells x {Xln.shape[1]} genes;",
      {s: int((lab0 == s).sum()) for s in STATES}, flush=True)

# ---------------- markers: CellMarker subtype-unique genes (as in the benchmark) ----------------
GT = json.load(open(f"{FSB}/results/cellmarker_gt_fine.json"))
GO = set(l.strip() for l in open(f"{FSB}/results/go_genes.txt") if l.strip())
markers = {c: (set(GT.get(c, [])) & GO) for c in STATES}
shared = {g for g, n in Counter(g for c in STATES for g in markers[c]).items() if n >= 2}
UNIQ = {c: (markers[c] - shared) for c in STATES}
print(f"[{LIN}] uniq GT sizes:", {c: len(UNIQ[c]) for c in STATES}, flush=True)

# ---------------- embed once (fixed encoder), RECAST functions ----------------
Z = contrastive_ig.embed(Xln)
mod = contrastive_ig.CentroidTarget(contrastive_ig.enc).to(contrastive_ig.dev).eval()
ig = IntegratedGradients(mod, multiply_by_inputs=True)


def recast_rank(tgt_idx, ref_idx):
    """Benchmark RECAST recipe (== reference_choice.py): the paper's mean_lognorm centroid C (per-cell
    lognorm THEN mean over the cell set -- validation/recast_centroid.py), u from embedding means,
    IG C_R->C_T, gate dC>0, signed ranking. Returns (order, u)."""
    u = Z[tgt_idx].mean(0) - Z[ref_idx].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
    Cc = fc.centroid(cnt, tgt_idx)     # the paper's single centroid definition (mean_lognorm);
    Cr = fc.centroid(cnt, ref_idx)     # see validation/recast_centroid.py
    inp = torch.tensor(Cc[None], device=contrastive_ig.dev).requires_grad_(True)
    a = ig.attribute(inp, baselines=torch.tensor(Cr[None], device=contrastive_ig.dev), target=0,
                     additional_forward_args=(torch.tensor(u[None], device=contrastive_ig.dev),),
                     n_steps=50).detach().cpu().numpy().ravel()
    a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf)
    return np.argsort(-a), u


# ---------------- perturbation generator ----------------
cen = {s: Z[lab0 == s].mean(0) for s in STATES}
D = np.stack([np.linalg.norm(Z - cen[s], axis=1) for s in STATES], axis=1)   # [cells, states]
sidx = {s: j for j, s in enumerate(STATES)}
own = np.array([sidx[s] for s in lab0])
d_own = D[np.arange(len(lab0)), own]
D_other = D.copy(); D_other[np.arange(len(lab0)), own] = np.inf
near_other = np.argmin(D_other, axis=1)                 # nearest OTHER state per cell
d_other = D_other[np.arange(len(lab0)), near_other]
margin = d_own - d_other                                # high = ambiguous / misplaced
amb_order = np.argsort(-margin)                         # most ambiguous first


def perturb(mode, eps, seed):
    rng = np.random.default_rng(seed)
    n = len(lab0); k = int(round(eps * n))
    lab = lab0.copy()
    if mode == "boundary":
        pool = amb_order[:min(2 * k, n)]
        pick = rng.choice(pool, size=k, replace=False)
        for i in pick:
            lab[i] = STATES[near_other[i]]
    else:                                               # random
        pick = rng.choice(n, size=k, replace=False)
        for i in pick:
            lab[i] = STATES[rng.choice([j for j in range(len(STATES)) if j != own[i]])]
    return lab


# ---------------- run: baseline then grid ----------------
os.makedirs(f"{OUT}/ps_labels_{LIN}", exist_ok=True)
# tagged like every other output (RECAST_RESULT_TAG)
sp.save_npz(f"{OUT}/ps_export_{LIN}_counts{VTAG}.npz", cnt)
np.savez_compressed(f"{OUT}/ps_export_{LIN}_meta{VTAG}.npz",
                    genes=genes, lab0=lab0, states=np.array(STATES, dtype=object))

base_rank, base_u, base_top = {}, {}, {}
for s in STATES:
    order, u = recast_rank(np.where(lab0 == s)[0], np.where((lab0 != s))[0])
    base_rank[s], base_u[s] = order, u
    base_top[s] = [genes[j] for j in order[:50]]
print(f"[{LIN}] baseline done", flush=True)


def readouts(s, order, u):
    top = [genes[j] for j in order[:K]]
    b = base_rank[s]
    jac = len(set(order[:K]) & set(b[:K])) / len(set(order[:K]) | set(b[:K]))
    union = list(set(list(order[:50]) + list(b[:50])))
    pos_a = {j: r for r, j in enumerate(order)}; pos_b = {j: r for r, j in enumerate(b)}
    rho = spearmanr([pos_a[j] for j in union], [pos_b[j] for j in union]).statistic
    mk = [g for g in UNIQ[s] if g in gi]
    rec = len(set(top) & set(mk)) / len(mk) if mk else np.nan
    return jac, float(rho), float(np.dot(u, base_u[s])), rec, top


rows, gdump = [], {}
for s in STATES:
    jac, rho, cu, rec, top = readouts(s, base_rank[s], base_u[s])
    rows.append(dict(lineage=LIN, state=s, mode="none", eps=0.0, seed=-1,
                     jaccard20=1.0, spearman=1.0, cos_u=1.0, recall20=rec,
                     n_tgt_seen=int((lab0 == s).sum())))
for mode in ["boundary", "random"]:
    for eps in EPS:
        for seed in SEEDS:
            lab = perturb(mode, eps, seed)
            tagc = f"{mode}_e{int(eps*1000):03d}_s{seed}"
            with open(f"{OUT}/ps_labels_{LIN}/{tagc}{VTAG}.txt", "w") as fh:
                fh.write("\n".join(lab) + "\n")
            for s in STATES:
                tgt = np.where(lab == s)[0]; ref = np.where(lab != s)[0]
                if len(tgt) < 20:
                    print(f"  ! {tagc}/{s}: only {len(tgt)} target cells, skipped", flush=True)
                    continue
                order, u = recast_rank(tgt, ref)
                jac, rho, cu, rec, top = readouts(s, order, u)
                rows.append(dict(lineage=LIN, state=s, mode=mode, eps=eps, seed=seed,
                                 jaccard20=jac, spearman=rho, cos_u=cu, recall20=rec,
                                 n_tgt_seen=int(len(tgt))))
                gdump[f"{tagc}|{s}"] = top
        print(f"[{LIN}] {mode} eps={eps} done", flush=True)

gdump["baseline"] = base_top
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/ps_recast_{LIN}{VTAG}.csv", index=False)
json.dump(gdump, open(f"{OUT}/ps_genes_{LIN}{VTAG}.json", "w"), indent=1)

agg = (df[df["mode"] != "none"]
       .groupby(["mode", "eps"])[["jaccard20", "spearman", "cos_u", "recall20"]]
       .mean().round(3))
base_rec = df[df["mode"] == "none"].recall20.mean()
print(f"\n=== {LIN} RECAST partition-stability (mean over {len(STATES)} states x {len(SEEDS)} seeds) ===")
print(f"baseline recall20 = {base_rec:.3f}")
print(agg.to_string(), flush=True)
print(f"[wrote] ps_recast_{LIN}.csv, ps_genes_{LIN}.json, ps_labels_{LIN}/, ps_export_{LIN}_*.npz")
