"""Label-free reference choice for RECAST (Fig. 4f): can the reference be chosen from embedding geometry alone?

Without annotation one clusters the whole dataset (all lineages), gets numbered clusters and picks a target; there is
no sibling annotation.  Per target subtype, label-free rules with fixed parameters are compared with the
annotation-based reference:

  anno_sib   reference = the annotated sibling subtypes (uses the hierarchy; the benchmark's reference)
  vs_all     reference = every other cell of the dataset
  near_K     reference = the K nearest other clusters by centroid cosine (K fixed)
  phase0     rho < 0.7 -> the nearest single cluster, else vs_all

Everything downstream (u, centroid C, IG positive channel, Recall@20) is that of recast_subtypes.py; only the cells
that form the reference change.  The centroid is population_ig.cluster_C (mean_lognorm, as recast_subtypes.py;
RECAST_CENTROID=pseudobulk switches to pooled counts).  Writes results/labelfree_recovery<VTAG>.csv (and the clusters
near_K picked).
Env: SC (GPU).  Run: CUDA_VISIBLE_DEVICES=1 RECAST_RESULT_TAG=_v3 python embedding_reference.py"""
import sys, os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")
import population_ig as L                                              # noqa: E402
from sibling_targets import embed, attr_dir, ig_of                    # noqa: E402
from recast_subtypes import CFG, MARKERS, state_key, ATLASES                       # noqa: E402

OUT = "/path/to/research/analysis/subtype_benchmark/results"
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)
KREC = 20
NEARK = [1, 2, 3, 5]

# target state -> its annotated siblings (from CFG lineage grouping); and source stem -> its target states
TARGET2SIB, SRC2TARGETS = {}, {}
for key, (stem, finef, STATES) in CFG.items():
    for c in STATES:
        TARGET2SIB[(stem, c)] = [s for s in STATES if s != c]
        SRC2TARGETS.setdefault(stem, {"finef": finef, "targets": []})
        if c not in SRC2TARGETS[stem]["targets"]:
            SRC2TARGETS[stem]["targets"].append(c)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def recast_genes(tgt, ref_mask, Z, Craw, rawvar, genes, topn=30):
    """u + centroid C_c/C_o over target vs the chosen reference cells -> positive-channel IG -> top genes."""
    u = unit(Z[tgt].mean(0) - Z[ref_mask].mean(0)).astype("float32")
    sel = tgt | ref_mask
    binlab = np.where(tgt, "TGT", "OTHER")[sel]
    C = L.cluster_C(Craw[sel], binlab, ["TGT", "OTHER"], rawvar)      # [2, 28231] in go order
    C_c, C_o = C[0], C[1]
    m0 = np.where((C_c - C_o) > 0, attr_dir(C_c, C_o, u), -np.inf)
    return [ig_of[int(j)] for j in np.argsort(-m0)[:topn]]


def recov(top, c, genes):
    mk = MARKERS[state_key(c)] & set(genes)
    return (len(set(top[:KREC]) & mk) / len(mk)) if mk else np.nan


def run_source(stem):
    finef = SRC2TARGETS[stem]["finef"]; targets = SRC2TARGETS[stem]["targets"]
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    genes = np.asarray(att.var_names).astype(str); bc = np.asarray(att.obs_names).astype(str)
    fine = np.array([l.strip() for l in open(f"{ATLASES}/{finef}") if l.strip()])
    assert len(fine) == att.n_obs; del att
    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pmap = {b: i for i, b in enumerate(np.asarray(fg.obs_names).astype(str))}
    Craw = fg.X.tocsr()[np.array([pmap[b] for b in bc])]; rawvar = np.asarray(fg.var_names).astype(str); del fg

    Z = embed(Xln)
    states = [s for s in np.unique(fine) if (fine == s).sum() >= 20]
    cen = {s: Z[fine == s].mean(0) for s in states}
    ucen = {s: unit(cen[s]) for s in states}
    rows = []
    for c in targets:
        tgt = fine == c
        others = [s for s in states if s != c]
        order = sorted(others, key=lambda s: -float(ucen[c] @ ucen[s]))     # nearest first (cosine desc)
        # rho for phase0
        mu_rest = Z[~tgt].mean(0)
        rho = float(np.linalg.norm(cen[c] - cen[order[0]]) / (np.linalg.norm(cen[c] - mu_rest) + 1e-12))

        def refmask(state_list):
            return np.isin(fine, state_list)

        # parameter-FREE auto: include neighbours more similar to c than c is to the whole-dataset centre
        thr = float(ucen[c] @ unit(Z.mean(0)))
        auto = [s for s in order if float(ucen[c] @ ucen[s]) > thr]
        rules = {"anno_sib": TARGET2SIB[(stem, c)], "vs_all": others, "near_auto": (auto or order[:1])}
        for K in NEARK:
            rules[f"near{K}"] = order[:K]
        rules["phase0"] = (order[:1] if rho < 0.7 else others)
        for rule, rlist in rules.items():
            rm = refmask(rlist)
            if rm.sum() < 10:
                continue
            top = recast_genes(tgt, rm, Z, Craw, rawvar, genes)
            rows.append(dict(source=stem, target=c, panel=state_key(c), rule=rule,
                             recall20=round(recov(top, c, genes), 4), rho=round(rho, 3),
                             ref_states=";".join(rlist[:6]) + ("..." if len(rlist) > 6 else "")))
        print(f"  [{stem}] {c:32s} rho={rho:.2f}  nearest3={order[:3]}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allr = pd.concat([run_source(s) for s in ["pbmcbmn", "yoshida", "dominguez_conde"]], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    allr.to_csv(f"{OUT}/labelfree_recovery{VTAG}.csv", index=False)
    piv = allr.pivot_table(index="rule", values="recall20", aggfunc="mean").reindex(
        ["anno_sib", "near_auto", "near1", "near2", "near3", "near5", "phase0", "vs_all"])
    nsel = allr[allr.rule == "near_auto"].ref_states.apply(lambda s: s.count(";") + 1)
    print(f"\nnear_auto picked {nsel.mean():.1f} neighbours on average (range {nsel.min()}-{nsel.max()})")
    print("\n=== mean Recall@20 over 14 targets, by reference rule ===")
    print(piv.round(3).to_string())
    print(f"\nwrote labelfree_recovery{VTAG}.csv")
