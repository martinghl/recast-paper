import os
"""Deletion / insertion faithfulness of the gene rankings (Fig. 2e, Supplementary Fig. 1a).

For a subtype c against its siblings the encoder scalar is
  f(x) = <enc(x), u>,   u = normalize(mu_c - mu_sib)   (SCimilarity embedding; enc output unit-norm).
Integrated gradients from the baseline C_o (sibling centroid) to the input C_c (target centroid) satisfies
completeness, sum_g attr_g = f(C_c) - f(C_o), so the test runs along the C_o <-> C_c axis, gene by gene, in the
order of a ranking:

  DELETION  : start at C_c; set the top-k ranked genes to their C_o values; re-embed; measure f.  f falls toward
              f(C_o); a ranking that puts the most responsible genes first makes f fall fastest -> lowest
              deletion AUC.
  INSERTION : start at C_o; set the top-k ranked genes to their C_c values; re-embed; measure f.  f rises toward
              f(C_c); a faithful ranking makes f rise fastest -> highest insertion AUC.

Rankings per subtype:
  RECAST   IG positive channel  (where(C_c - C_o > 0, attr_dir(C_c, C_o, u), -inf); argsort descending)
  DE       expression shift (argsort of C_c - C_o, descending)
  random   a fixed per-subtype permutation
f is min-max normalized per subtype to [0, 1]: (f - f(C_o)) / (f(C_c) - f(C_o)).  Reported: deletion AUC, insertion
AUC and gap = insertion - deletion AUC over the fraction of genes in [0, 1], plus a top-512 zoom.  Four immune lineages
(recast_subtypes.CFG).  Env: SC (GPU).  Writes results/faithfulness_delins_{auc,curves}.csv."""
import sys, os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")
import population_ig as L                                      # noqa: E402  props_and_C / cluster_countsums
from sibling_targets import embed, attr_dir, ig_of             # noqa: E402  contrastive_ig encoder on GPU + signed IG
from recast_subtypes import CFG                                     # noqa: E402  lineage -> (stem, labels_file, states)
VTAG = os.environ.get("RECAST_RESULT_TAG", "")   # suffix of the output files (README: result tags)

ATLASES = "/path/to/research/analysis/atlases"
OUT = "/path/to/research/analysis/subtype_benchmark/results"
NSTEPS = 50
ZOOM = 512                                                 # top-N genes for the marker-regime zoom AUC


def f_of(C_batch, u):
    """f(x)=<enc(x),u> for a stack of centroids [M,G]; enc output already unit-norm -> f in [-1,1]."""
    return embed(np.ascontiguousarray(C_batch, dtype="float32")) @ u.astype("float32")


def curves(C_c, C_o, u, order, ks):
    """Deletion (C_c -> C_o over ranked genes) and insertion (C_o -> C_c) f-values at each k in ks."""
    G = C_c.shape[0]
    Del = np.tile(C_c, (len(ks), 1)); Ins = np.tile(C_o, (len(ks), 1))
    for i, k in enumerate(ks):
        if k:
            idx = order[:k]
            Del[i, idx] = C_o[idx]        # delete = revert top-k ranked genes to reference
            Ins[i, idx] = C_c[idx]        # insert = set top-k ranked genes to target
    return f_of(Del, u), f_of(Ins, u)


def auc_norm(fvals, f_o, f_c, xs):
    """min-max normalize f to [0,1] vs (f_o,f_c) and trapezoid-integrate over xs (fraction axis)."""
    n = (fvals - f_o) / (f_c - f_o + 1e-12)
    auc = float(np.sum((n[1:] + n[:-1]) / 2.0 * np.diff(xs)))    # np.trapz removed in numpy 2.x
    return auc, n


def run_lineage(key, aucrows, curverows):
    stem, finef, STATES = CFG[key]
    att = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    genes = np.asarray(att.var_names).astype(str)          # go order (28231)
    bc = np.asarray(att.obs_names).astype(str)
    fine = np.array([l.strip() for l in open(f"{ATLASES}/{finef}") if l.strip()])
    assert len(fine) == att.n_obs, f"[{key}] label/obs mismatch {len(fine)} vs {att.n_obs}"
    del att
    # raw counts (fullgene), aligned to att cell order — identical to recast_subtypes
    fg = ad.read_h5ad(f"{ATLASES}/{stem}_fullgene.h5ad")
    pmap = {b: i for i, b in enumerate(np.asarray(fg.obs_names).astype(str))}
    Craw_all = fg.X.tocsr()[np.array([pmap[b] for b in bc])]
    rawvar = np.asarray(fg.var_names).astype(str)
    del fg

    STATES = [s for s in STATES if (fine == s).sum() >= 60]
    keep = np.isin(fine, STATES)
    Xln, labels, Craw = Xln[keep], fine[keep], Craw_all[keep]
    del Craw_all
    G = Xln.shape[1]
    # k grid: log-spaced over [1,G] + 0 + G, unique ints. Dense early where the action is.
    ks = sorted(set([0] + [int(round(x)) for x in np.geomspace(1, G, 44)] + [G]))
    fr = np.array(ks) / G                                   # fraction axis for full-range AUC
    zmask = np.array(ks) <= ZOOM                            # top-512 zoom
    frz = np.array(ks)[zmask] / ZOOM

    Z = embed(Xln)
    print(f"[{key}] {len(labels)} cells, states={[(s, int((labels==s).sum())) for s in STATES]}", flush=True)
    for si, c in enumerate(STATES):
        tgt = labels == c
        if tgt.sum() < 5 or (~tgt).sum() < 5:
            continue
        u = Z[tgt].mean(0) - Z[~tgt].mean(0); u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        binlab = np.where(tgt, c, "OTHER")
        # the paper's single centroid definition (mean_lognorm by default; $RECAST_CENTROID
        # =pseudobulk switches to pooled counts). See population_ig.cluster_C.
        C = L.cluster_C(Craw, binlab, [c, "OTHER"], rawvar)
        C_c, C_o = C[0].astype("float32"), C[1].astype("float32")
        f_c = float(f_of(C_c[None], u)[0]); f_o = float(f_of(C_o[None], u)[0])
        pos = (C_c - C_o) > 0

        m0 = np.where(pos, attr_dir(C_c, C_o, u, n_steps=NSTEPS), -np.inf)
        orders = {
            "RECAST":  np.argsort(-m0),                      # IG positive channel
            "DE":     np.argsort(-(C_c - C_o)),             # raw expression shift, descending
            "random": np.random.default_rng(1234 + si).permutation(G),
        }
        summ = {}
        for rk, order in orders.items():
            fdel, fins = curves(C_c, C_o, u, order, ks)
            del_auc, ndel = auc_norm(fdel, f_o, f_c, fr)
            ins_auc, nins = auc_norm(fins, f_o, f_c, fr)
            del_z, _ = auc_norm(fdel[zmask], f_o, f_c, frz)
            ins_z, _ = auc_norm(fins[zmask], f_o, f_c, frz)
            aucrows.append(dict(lineage=key, state=c, ranking=rk, f_c=round(f_c, 4), f_o=round(f_o, 4),
                                del_auc=round(del_auc, 4), ins_auc=round(ins_auc, 4), gap=round(ins_auc - del_auc, 4),
                                del_auc_top512=round(del_z, 4), ins_auc_top512=round(ins_z, 4)))
            summ[rk] = (del_auc, ins_auc)
            for k, nd, ni in zip(ks, ndel, nins):
                curverows.append(dict(lineage=key, state=c, ranking=rk, k=k, frac=k / G,
                                      del_f=round(float(nd), 5), ins_f=round(float(ni), 5)))
        print(f"  [{key}/{c}] f_o={f_o:.3f} f_c={f_c:.3f} | "
              + " | ".join(f"{rk} del={d:.3f} ins={i:.3f}" for rk, (d, i) in summ.items()), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    aucrows, curverows = [], []
    for key in CFG:
        run_lineage(key, aucrows, curverows)
    A = pd.DataFrame(aucrows); A.to_csv(f"{OUT}/faithfulness_delins_auc{VTAG}.csv", index=False)
    pd.DataFrame(curverows).to_csv(f"{OUT}/faithfulness_delins_curves{VTAG}.csv", index=False)
    print("\n=== per-ranking summary (mean over all states, 4 lineages) ===")
    print(A.groupby("ranking")[["del_auc", "ins_auc", "gap", "del_auc_top512", "ins_auc_top512"]].mean().round(3).to_string())
    print("\n=== per-lineage gap (ins_auc - del_auc; higher=more faithful) ===")
    print(A.pivot_table(index="lineage", columns="ranking", values="gap", aggfunc="mean").round(3).to_string())
    print("\nRECAST faithful iff del_auc(RECAST) < DE < random AND ins_auc(RECAST) > DE > random.")
    print("[done]", flush=True)
