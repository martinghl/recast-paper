"""Supplementary Fig. 7: RECAST panels of the iPSC-to-midbrain differentiation data, recomputed with the package.

validation/biology/trajectory_coherence.py embedded the 82,166 cells with SCimilarity, computed diffusion
pseudotime on the embedding (15 neighbours, rooted in the floor-plate progenitor closest to its type centroid,
oriented to collection day) and ranked genes for every type with at least 30 cells against all other cells. The
top 20 genes per type are stored with their eta^2 in trajectory_gene_eta2_v3.csv (panel b), 20,000 cells with their
pseudotime in trajectory_pseudotime_v3.csv (panel a), the profiles of the dopaminergic-neuron genes in
discovery_case_da_profiles_v3.csv (panel c) and the complete dopaminergic-neuron ranking in
discovery_case_da_fullranks_v3.csv (panel d). The original ranking set gated-in genes with negative attribution to
zero and gated-out genes to minus infinity, so only the genes with positive attribution have a defined order.

Here the embedding is SCimilarityEncoder.embed, the rankings are one recast.attribute call, and the pseudotime,
eta^2 and profiles are recomputed with the original code on the package embedding. Diffusion pseudotime turns
differences of 1e-7 in the embedding into differences of up to 0.07, so the embedding is made as the original run
made it: from the original log-normalized matrix (x * (1e4 / total), then log1p) in batches of 512 cells.

    python check_development.py
Writes development_panels.csv and development_summary.csv."""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from scipy.stats import spearmanr

import common as C
import recast

BUN = os.path.join(C.SB, "ans_replication", "bundles")
BV = os.path.join(C.SB, "validation", "biology")
K, NBIN = 20, 25


def pseudotime(Z, ct, tnum):
    """trajectory_coherence.py: diffusion pseudotime on the embedding, oriented to collection day."""
    n = len(ct)
    A = ad.AnnData(X=sp.csr_matrix((n, 1), dtype="float32"))
    A.obsm["X_scim"] = Z.astype("float32")
    sc.pp.neighbors(A, use_rep="X_scim", n_neighbors=15)
    sc.tl.diffmap(A)
    prog = np.where(ct == "P_FPP")[0]
    cen = Z[prog].mean(0)
    A.uns["iroot"] = int(prog[np.argmin(((Z[prog] - cen) ** 2).sum(1))])
    sc.tl.dpt(A)
    dpt = A.obs["dpt_pseudotime"].to_numpy().astype("float64")
    dpt[~np.isfinite(dpt)] = np.nanmax(dpt[np.isfinite(dpt)])
    if spearmanr(dpt, tnum, nan_policy="omit").correlation < 0:
        dpt = dpt.max() - dpt
    return dpt


def eta2_binned(Xsp, order, B):
    """trajectory_coherence.py: variance explained by B equal-count bins of the ordering, and the bin means."""
    edges = np.quantile(order, np.linspace(0, 1, B + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, order, side="right") - 1, 0, B - 1)
    S = sp.csr_matrix((np.ones(len(b)), (b, np.arange(len(b)))), shape=(B, len(b)))
    nk = np.asarray(S.sum(1)).ravel()
    binmean = np.asarray((S @ Xsp).todense()) / np.clip(nk[:, None], 1, None)
    grand = np.asarray(Xsp.sum(0)).ravel() / Xsp.shape[0]
    ss_b = (nk[:, None] * (binmean - grand[None, :]) ** 2).sum(0)
    sq = np.asarray(Xsp.multiply(Xsp).sum(0)).ravel(); ss_t = sq - Xsp.shape[0] * grand ** 2
    return np.clip(ss_b / np.clip(ss_t, 1e-12, None), 0, 1), binmean, grand


def zprofile(binmean, idx):
    p = binmean[:, idx]
    p = (p - p.mean(0, keepdims=True)) / (p.std(0, keepdims=True) + 1e-9)
    return p.mean(1)


def main():
    raw = ad.read_h5ad(os.path.join(BUN, "neural_diff_a_raw.h5ad"))
    genes = np.asarray(raw.var_names).astype(str)
    assert list(genes) == C.scimilarity_genes()
    ct = raw.obs["celltype"].astype(str).to_numpy()
    a = ad.AnnData(sp.csr_matrix(raw.X))
    a.var_names = genes
    a.obs["ct"] = ct
    del raw
    lb = ad.read_h5ad(os.path.join(BUN, "neural_diff_a_log.h5ad"), backed="r")
    assert (lb.obs["celltype"].astype(str).to_numpy() == ct).all()
    tp = lb.obs["time_point"].astype(str).to_numpy(); U = np.asarray(lb.obsm["X_umap"])
    lb.file.close()
    tnum = np.array([{"D11": 0, "D30": 1, "D52": 2}.get(x, np.nan) for x in tp])
    n = len(ct)

    enc = C.scimilarity()
    tot = np.asarray(a.X.sum(1)).ravel()
    Xln = a.X.multiply(1e4 / np.clip(tot, 1, None)[:, None]).log1p().tocsr().astype("float32")
    Z = enc.embed(Xln, batch_size=512)
    dpt = pseudotime(Z, ct, tnum)
    eta, binmean, grand = eta2_binned(Xln, dpt, NBIN)
    summ = []

    # panel a: the 20,000 drawn cells (the draw follows a 20-gene draw from the same generator)
    expressed = grand > np.quantile(grand[grand > 0], 0.25)
    rng = np.random.default_rng(0)
    rng.choice(np.where(expressed)[0], size=K, replace=False)
    ss = rng.choice(n, size=min(20000, n), replace=False)
    P = pd.read_csv(os.path.join(BV, "trajectory_pseudotime_v3.csv"))
    if not np.allclose(P[["ux", "uy"]].to_numpy(), U[ss]):      # the gene draw depends on a float threshold
        from scipy.spatial import cKDTree
        ss = cKDTree(U).query(P[["ux", "uy"]].to_numpy())[1]
    assert np.allclose(P[["ux", "uy"]].to_numpy(), U[ss]) and (P.celltype.to_numpy() == ct[ss]).all()
    d = np.abs(dpt[ss] - P.dpt.to_numpy())
    summ.append(dict(item="a pseudotime (20,000 cells)", n=len(ss), equal=int((d <= 1e-6).sum()), max_abs_diff=d.max(),
                     spearman=spearmanr(dpt[ss], P.dpt).correlation))

    # panel b: RECAST top 20 per type, and the eta^2 of those genes
    states = [s for s in sorted(set(ct)) if (ct == s).sum() >= 30]
    res = recast.attribute(enc, a, "ct", target=states, reference="rest", gate="dC", centroid="mean_lognorm",
                           qc="off", embeddings=Z)
    G = pd.read_csv(os.path.join(BV, "trajectory_gene_eta2_v3.csv"))
    gi = {g: i for i, g in enumerate(genes)}
    rows = []
    for s in states:
        stored = list(G[G.state == s].gene.astype(str))
        top = res.genes[s][:K]
        e_pkg = np.array([eta[gi[g]] for g in stored])
        rows.append(dict(state=s, n_cells=int((ct == s).sum()), **C.compare_lists(top, stored),
                         max_abs_eta2_diff=float(np.abs(e_pkg - G[G.state == s].eta2_dpt.to_numpy()).max())))
    R = pd.DataFrame(rows)
    summ.append(dict(item="b top-20 lists", n=len(R), equal=int(R.identical.sum()),
                     max_abs_diff=R.max_abs_eta2_diff.max()))

    # panel c: profiles of the dopaminergic-neuron genes along pseudotime (genes from the original table)
    Pc = pd.read_csv(os.path.join(BV, "discovery_case_da_profiles_v3.csv"))
    r2 = np.random.default_rng(0)
    randset = r2.choice(np.where((np.asarray(Xln.sum(0)).ravel() / n) > 0)[0], 200, replace=False)
    dc = [np.abs(zprofile(binmean, [gi[g]]) - Pc[g].to_numpy()).max() for g in Pc.columns if g not in ("bin", "random")]
    dc.append(np.abs(zprofile(binmean, list(randset)) - Pc["random"].to_numpy()).max())
    summ.append(dict(item="c profiles (5 genes + random)", n=len(dc), equal=int((np.array(dc) <= 1e-6).sum()),
                     max_abs_diff=float(max(dc))))

    # panel d: the complete dopaminergic-neuron ranking
    F = pd.read_csv(os.path.join(BV, "discovery_case_da_fullranks_v3.csv")).sort_values("recast_rank")
    fa = F.recast_attr.to_numpy(float)
    att = res.attribution["DA"].reindex(F.gene).to_numpy()
    dC = res.dC["DA"].reindex(F.gene).to_numpy()
    npos = int((fa > 0).sum())
    pos_stored = list(F.gene[fa > 0])
    r = C.compare_lists(res.genes["DA"][:npos], pos_stored)
    same = int(sum(x == y for x, y in zip(res.genes["DA"][:npos], pos_stored)))
    summ.append(dict(item=f"d ranks 1-{npos} (positive attribution)", n=npos, equal=same,
                     max_abs_diff=float(np.abs(att[fa > 0] - fa[fa > 0]).max()), first_diff_rank=r["first_diff_rank"]))
    if r["first_diff_rank"]:
        k = int(r["first_diff_rank"])
        for j in range(max(k - 2, 1), min(k + 2, npos) + 1):
            g, h = pos_stored[j - 1], res.genes["DA"][j - 1]
            print(f"  DA rank {j}: original {g} {fa[j - 1]:.9g} | package {h} {res.attribution['DA'][h]:.9g}")
    zero_pkg = set(F.gene[(dC > 0) & (att <= 0)]); zero_stored = set(F.gene[fa == 0])
    gated_pkg = set(F.gene[dC <= 0]); gated_stored = set(F.gene[np.isneginf(fa)])
    summ.append(dict(item="d gated in, attribution <= 0 (original ranks tied at 0)", n=len(zero_stored),
                     equal=len(zero_pkg & zero_stored), max_abs_diff=np.nan))
    summ.append(dict(item="d gated out (original ranks tied at -inf)", n=len(gated_stored),
                     equal=len(gated_pkg & gated_stored), max_abs_diff=np.nan))
    C.assert_package_only()
    S = pd.DataFrame(summ)
    print(R.to_string(index=False))
    print(S.to_string(index=False))
    C.write(R, "development_panels.csv")
    C.write(S, "development_summary.csv")


if __name__ == "__main__":
    main()
