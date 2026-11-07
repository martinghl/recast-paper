"""Contrastive centroid integrated gradients: the contrast enters the attribution target and the baseline.

  target   f_c(x) = <psi(x), u_c>,  u_c = normalize(zbar_c - zbar_negc)   (psi = SCimilarity encoder, L2-normalized
                                                                           output)
  baseline x' = x_bar_negc = mean log-normalized profile of the reference set
  sign     kept (no abs); no row rescaling, so sum_g a_ig = f_c(x_i) - f_c(x') (completeness)

Also the shared encoder loading (`go`, `embed`, `CentroidTarget`) that the other modules of this folder import.
Run on its own, it computes a centroid diagnostic and the attributions for a labelled dataset and checks
completeness and marker ranks.

Usage:
  python contrastive_ig.py --dataset pbmc3k
  python contrastive_ig.py --dataset pbmcbmn --labels <per-cell labels, aligned to attr_pbmcbmn_cells.txt>"""
import argparse, sys, json, warnings, numpy as np, pandas as pd, anndata as ad, torch, scipy.sparse as sp
import torch.nn as nn
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
sys.path.insert(0, "/path/to/research/analysis/attribution_methods")
sys.path.insert(0, "/path/to/research/external/Signature/src")
import data as sig_data
from captum.attr import IntegratedGradients
from engine import AttributionEngine

MODEL = "/path/to/scimilarity/model_v1.1"
ATLASES = "/path/to/research/analysis/atlases"
eng = AttributionEngine.from_scimilarity(MODEL, use_gpu=True)
go = sig_data.read_gene_order(MODEL); enc, dev = eng.encoder, eng.device
gi = {g: i for i, g in enumerate(go)}
rng = np.random.default_rng(0)


def embed(X, bs=512):
    """psi(x): SCimilarity encoder output (already unit-norm)."""
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            out.append(enc(torch.tensor(X[i:i+bs], dtype=torch.float32, device=dev)).cpu().numpy())
    return np.vstack(out)


class CentroidTarget(nn.Module):
    """f_c(x) = <psi(x), u>, u a fixed per-cluster unit direction (additional_forward_arg, no grad)."""
    def __init__(self, encoder): super().__init__(); self.encoder = encoder
    def forward(self, x, u):
        s = (self.encoder(x) * u).sum(1)
        return torch.stack([s, s], dim=1)          # [B,2] for captum target=0


def load(dataset, labels_path):
    if dataset == "pbmc3k":
        raw = ad.read_h5ad(f"{ATLASES}/pbmc3k_fullgene.h5ad")
        adata = sig_data.harmonize(raw, go)
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        labels = raw.obs["label"].astype(str).to_numpy()
        markers = {str(k): set(v) for k, v in json.load(open("marker_sets_celltype_pbmc3k.json")).items() if v}
        att = ad.read_h5ad(f"{ATLASES}/pbmc3k_attribution_scimilarity.h5ad")   # for reduction check
        Xattr = att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)
        ig_stored = np.asarray(att.layers["attribution_ig"].todense()) if sp.issparse(att.layers["attribution_ig"]) else np.asarray(att.layers["attribution_ig"])
        return X.astype("float32"), labels, markers, Xattr.astype("float32"), ig_stored
    if dataset == "pbmcbmn":
        att = ad.read_h5ad(f"{ATLASES}/pbmcbmn_attribution_scimilarity.h5ad")   # X = harmonized lognorm (28231, go order)
        X = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
        cells = [l.strip() for l in open(f"{ATLASES}/attr_pbmcbmn_cells.txt")]
        assert X.shape[0] == len(cells), f"{X.shape[0]} cells vs {len(cells)} in attr_pbmcbmn_cells.txt"
        if labels_path is None:
            raise SystemExit("pbmcbmn needs --labels <per-cell l2 labels aligned to attr_pbmcbmn_cells.txt>")
        labels = np.array([l.strip() for l in open(labels_path) if l.strip()])
        assert len(labels) == X.shape[0], f"labels {len(labels)} != cells {X.shape[0]}"
        ig_stored = att.layers["attribution_ig"]        # keep sparse; reduction_check densifies only sub-rows
        markers = {  # canonical Azimuth-l2 markers (sign-sanity only; recovery is expression's turf)
            "CD14 Mono": {"CD14", "S100A8", "S100A9", "LYZ", "FCN1", "VCAN"},
            "CD16 Mono": {"FCGR3A", "MS4A7", "LST1", "CDKN1C", "LYPD2", "CSF1R"},
            "NK": {"NKG7", "GNLY", "KLRD1", "NCR1", "PRF1", "GZMB"},
            "NK_CD56bright": {"NCAM1", "SELL", "GZMK", "XCL1", "IL7R"},
            "NK Proliferating": {"MKI67", "TOP2A", "NKG7", "GZMB"},
            "B naive": {"MS4A1", "CD79A", "CD79B", "TCL1A", "IGHD", "IL4R"},
            "B memory": {"MS4A1", "CD79A", "CD27", "TNFRSF13B", "AIM2"},
            "B intermediate": {"MS4A1", "CD79A", "CD79B", "TCL1A"},
            "Plasmablast": {"MZB1", "JCHAIN", "XBP1", "PRDM1", "CD38", "TNFRSF17"},
        }
        return X, labels, markers, X, ig_stored
    raise SystemExit(f"unknown dataset {dataset}")


def phase0(X, labels, types):
    """Centroid diagnostic gate (no gradients): rho_c, negative-choice margin, directional bootstrap
    uncertainty, encoder-nonlinearity ratio. Decides vs-rest / vs-nearest / temperature."""
    Z = embed(X)                                     # psi per cell, unit-norm
    zc = np.vstack([Z[labels == t].mean(0) for t in types])              # [K,d] cluster centroids
    zrest = np.vstack([Z[labels != t].mean(0) for t in types])          # vs-rest negative pole
    D = np.linalg.norm(zc[:, None] - zc[None], axis=2)                   # centroid-centroid dists
    np.fill_diagonal(D, np.inf)
    Xsum = X.sum(0); N = X.shape[0]
    rows = []
    for i, t in enumerate(types):
        order = np.argsort(D[i]); c2, c3 = order[0], order[1]
        d1, d2 = D[i, c2], D[i, c3]
        rho = np.linalg.norm(zc[i] - zc[c2]) / (np.linalg.norm(zc[i] - zrest[i]) + 1e-12)
        margin = (d2 - d1) / (d1 + 1e-12)
        u = (zc[i] - zrest[i]); u /= (np.linalg.norm(u) + 1e-12)
        # directional bootstrap: resample cells within cluster, recompute zbar_c, angle vs u
        idx = np.where(labels == t)[0]; angs = []
        for _ in range(50):
            bs = rng.choice(idx, size=len(idx), replace=True)
            zb = Z[bs].mean(0); ub = (zb - zrest[i]); ub /= (np.linalg.norm(ub) + 1e-12)
            angs.append(np.degrees(np.arccos(np.clip(ub @ u, -1, 1))))
        # encoder non-linearity: ||psi(xbar_negc) - zbar_negc|| / ||zbar_c - zbar_negc||
        nc = int((labels == t).sum()); xbar = (Xsum - X[labels == t].sum(0)) / (N - nc)
        nl = np.linalg.norm(embed(xbar[None].astype("float32"))[0] - zrest[i]) / (np.linalg.norm(zc[i] - zrest[i]) + 1e-12)
        rows.append(dict(cluster=t, n=int((labels == t).sum()), nearest=types[c2],
                         rho_c=round(float(rho), 3), neg_margin=round(float(margin), 3),
                         dir_ang95=round(float(np.percentile(angs, 95)), 2), enc_nl=round(float(nl), 3)))
    return pd.DataFrame(rows), zc, zrest


def cig_attribute(X, labels, types, zc, zrest, n_steps=50):
    """Phase 1: per-cluster IG with target=<psi,u_c>, baseline=xbar_negc, signed, no rescale.
    Accumulates the signed cluster mean abar [K,G] on the fly (never stores an [N,G] array), plus
    per-cell margin and completeness rel-error."""
    mod = CentroidTarget(enc).to(dev).eval()
    ig = IntegratedGradients(mod, multiply_by_inputs=True)
    G = X.shape[1]; Xsum = X.sum(0); Ntot = X.shape[0]
    abar = np.zeros((len(types), G), dtype=np.float64); margin = np.zeros(X.shape[0]); comp = []
    for i, t in enumerate(types):
        idx = np.where(labels == t)[0]
        u = (zc[i] - zrest[i]); u /= (np.linalg.norm(u) + 1e-12)
        u_t = torch.tensor(u, dtype=torch.float32, device=dev)
        xbar = ((Xsum - X[idx].sum(0)) / (Ntot - len(idx))).astype("float32")
        with torch.no_grad():
            f_base = float((enc(torch.tensor(xbar[None], device=dev)) * u_t).sum())
        acc = np.zeros(G, dtype=np.float64)
        for s in range(0, len(idx), 256):
            b = idx[s:s+256]; n = len(b)
            inp = torch.tensor(X[b], dtype=torch.float32, device=dev).requires_grad_(True)
            bt = torch.tensor(np.tile(xbar, (n, 1)), dtype=torch.float32, device=dev)
            ut = u_t.repeat(n, 1)
            a = ig.attribute(inp, baselines=bt, target=0, additional_forward_args=(ut,),
                             n_steps=n_steps, internal_batch_size=64*256).detach().cpu().numpy()
            acc += a.sum(0)
            with torch.no_grad():
                fx = (enc(inp.detach()) * u_t).sum(1).cpu().numpy()
            margin[b] = fx - f_base
            comp.append(np.abs(a.sum(1) - (fx - f_base)) / (np.abs(fx - f_base) + 1e-9))
        abar[i] = acc / len(idx)
    return abar, margin, float(np.median(np.concatenate(comp)))


def reduction_check(Xattr, ig_stored, n_steps=50):
    """cIG must be a strict generalization: self-target Σ_d|w_d e_d(x)|, baseline 0, abs, rescale-1000
    reproduces the stored attribution_ig (SIGnature's recipe), whose cotangent is the self embedding
    w = e(x).detach(), not an all-ones vector."""
    from formulations.targets import SelfRepr, TargetModule
    mod = TargetModule(enc, SelfRepr()); ig = IntegratedGradients(mod, multiply_by_inputs=True)
    N = min(400, Xattr.shape[0]); sub = rng.choice(Xattr.shape[0], N, replace=False)
    rep = np.zeros((N, Xattr.shape[1]))
    for s in range(0, N, 200):
        b = sub[s:s+200]
        inp = torch.tensor(Xattr[b], dtype=torch.float32, device=dev).requires_grad_(True)
        with torch.no_grad(): ref = enc(inp).detach()
        a = ig.attribute(inp, baselines=torch.zeros_like(inp), target=0,
                         additional_forward_args=(ref,), n_steps=n_steps, internal_batch_size=64*200).detach().cpu().numpy()
        rep[s:s+len(b)] = np.abs(a)
    rep = rep / (rep.sum(1, keepdims=True) + 1e-12) * 1000.0
    stored = ig_stored[sub]
    stored = np.asarray(stored.todense()) if sp.issparse(stored) else np.asarray(stored)
    r = [spearmanr(rep[k], stored[k]).correlation for k in range(N) if stored[k].sum() > 0]
    relerr = np.abs(rep - stored).sum(1) / (stored.sum(1) + 1e-9)
    return float(np.nanmedian(r)), float(np.median(relerr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pbmc3k")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--n_steps", type=int, default=50)
    args = ap.parse_args()

    X, labels, markers, Xattr, ig_stored = load(args.dataset, args.labels)
    types = sorted(set(labels))
    print(f"\n{'='*78}\n cIG on {args.dataset}: {X.shape[0]} cells x {X.shape[1]} genes, {len(types)} clusters\n{'='*78}")

    # ---- reduction check: with the self target, zero baseline, abs and rescaling, cIG reproduces attribution_ig ----
    rho, relerr = reduction_check(Xattr, ig_stored, args.n_steps)
    print(f"\n[reduction check]  self-target+zero-baseline+abs+rescale vs stored attribution_ig:")
    print(f"   median per-cell Spearman = {rho:.4f}   median rel-L1 error = {relerr:.4f}   "
          f"({'PASS' if rho > 0.99 and relerr < 0.02 else 'CHECK'})")

    # ---- Phase 0 gate ----
    p0, zc, zrest = phase0(X, labels, types)
    print(f"\n[Phase 0 gate]  rho_c=||c-c2||/||c-rest|| (~1 => vs-rest~=vs-nearest);  "
          f"neg_margin=(d2-d1)/d1 (small => argmin unstable);\n                dir_ang95=95th pct bootstrap "
          f"angle of u_c (deg);  enc_nl=||psi(xbar_negc)-zbar_negc||/||zbar_c-zbar_negc||")
    print(p0.to_string(index=False))
    hi_nl = p0.enc_nl.median()
    rec = "vs-rest" if (p0.rho_c.median() > 0.7 or p0.neg_margin.min() < 0.1) else "vs-nearest OK"
    print(f"   -> median rho_c={p0.rho_c.median():.2f}, min neg_margin={p0.neg_margin.min():.2f}, "
          f"median enc_nl={hi_nl:.2f}  => recommend main track: {rec}"
          f"{'; enc_nl large -> honest reference is psi(xbar_negc), consider sampled baseline (spec 4.4)' if hi_nl>0.25 else '; completeness wording holds'}")

    # ---- Phase 1 cIG + acceptance ----
    abar, margin, comp = cig_attribute(X, labels, types, zc, zrest, args.n_steps)
    print(f"\n[Phase 1 cIG]  signed attribution done. completeness median rel-error = {comp:.4f} "
          f"({'PASS <1%' if comp < 0.01 else 'raise n_steps' if comp < 0.05 else 'CHECK'})")

    # sign sanity + marker rank (only where we have a marker set)
    if markers:
        print(f"\n[sign sanity]  known markers -> positive abar in home cluster + rank among all genes:")
        for i, t in enumerate(types):
            mk = [g for g in markers.get(t, []) if g in gi]
            if not mk: continue
            order = np.argsort(-abar[i]); rank = {g: int(np.where(order == gi[g])[0][0]) for g in mk}
            pos = np.mean([abar[i, gi[g]] > 0 for g in mk])
            print(f"   {t:14s} pos_frac={pos:.2f}  ranks=" +
                  ", ".join(f"{g}:{rank[g]}" for g in sorted(mk, key=lambda g: rank[g])[:6]))

    # falsification: cIG must not be a monotone transform of logFC
    print(f"\n[falsification]  Spearman(abar_c, logFC_c) over genes -- must be materially < 1:")
    sp_rows = []
    for i, t in enumerate(types):
        logfc = X[labels == t].mean(0) - X[labels != t].mean(0)
        rr = spearmanr(abar[i], logfc).correlation
        sp_rows.append(rr)
        print(f"   {t:14s} rho={rr:.3f}")
    print(f"   median Spearman(cIG, logFC) = {np.nanmedian(sp_rows):.3f}  "
          f"({'FM reweights, not just rescales' if np.nanmedian(sp_rows) < 0.9 else 'WARNING: ~logFC, FM adds nothing'})")

    np.save(f"results/cig_abar_{args.dataset}.npy", abar)
    pd.DataFrame({"cluster": types}).to_csv(f"results/cig_types_{args.dataset}.csv", index=False)
    p0.to_csv(f"results/cig_phase0_{args.dataset}.csv", index=False)
    print(f"\n[done] wrote results/cig_abar_{args.dataset}.npy, results/cig_phase0_{args.dataset}.csv")


if __name__ == "__main__":
    main()
