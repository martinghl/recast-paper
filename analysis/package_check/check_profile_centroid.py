"""Supplementary Fig. 2a-d: profile-centroid agreement and attribution completeness, recomputed with the package.

For every target-reference comparison in validation/profile_centroid/results/pc_align_v1.csv, and
for the step series in pc_align_steps_v1.csv, the package gives
  recast.attribute(...).completeness  f(C_T) - f(C_R) (delta_f), the sum of the attributions and their
                                      relative error at the requested number of integration steps
  enc.embed(cells)                    the centroid separation zbar_T - zbar_R
  enc.torch_encode([C_T, C_R])        the profile displacement E(C_T) - E(C_R), with C_A from
                                      recast.centroid.mean_lognorm_centroid (mean_counts_centroid for scVI,
                                      whose endpoints are mean raw counts)
and the original cos_align, ratio, delta_profile, sum_phi and completeness_rel are compared with them.

Stages (one per original cohort):
  fine65   SCimilarity, every state of the 21 benchmark lineages against the rest of its lineage
  steps    the eight SCimilarity comparisons with the largest 50-step error, at 50 / 200 / 1,000 steps
  zhaoref  Fig. 4's Zhao targets x seven references (the original generator: one draw per target)
  ssl      scTab_SSL and PBMC_SSL (recast.SSLEncoder) on the fine65 comparisons
  scvi     one scVI model per lineage, trained with the original recipe (seed 0), recast.SCVIEncoder; the
           original run kept no model and retraining on the GPU does not give the same weights, so this stage
           also runs the original computation (alignment.py) on the same retrained model and compares the two
  scf      scFoundation (recast.SCFoundationEncoder) on the six cross-encoder lineages

    python check_profile_centroid.py --stage fine65
Writes profile_centroid_<stage>.csv."""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import anndata as ad
import common as C
import check_reference_choice as RC
import recast
from recast.centroid import mean_counts_centroid, mean_lognorm_centroid

PCA = os.path.join(C.SB, "validation", "profile_centroid", "results")
SSL = [("scTab_SSL", os.path.join(C.SB, "models", "prepared", "ssl_sctab")),
       ("PBMC_SSL", os.path.join(C.SB, "models", "prepared", "ssl_pbmc"))]
SCF = os.path.join(C.SB, "models", "prepared", "scfoundation")
COLS = ["n_target", "n_reference", "norm_dz", "delta_profile", "cos_align", "ratio", "sum_phi", "completeness_rel"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fine_lineages():
    return sorted(pd.read_csv(os.path.join(C.FSB, "results", "fs_uniq_perstate_v3.csv")).lineage.unique())


def lineage(dsid):
    """(AnnData of raw counts in SCimilarity's order with obs['state'], kept states) for one benchmark lineage."""
    e = C.config("lineage_registry").LINEAGES[dsid]
    labels = C.read_labels(os.path.join(C.ATLASES, e["label_file"]))
    return C.lineage_adata(C.ATLASES, e["ds"], labels, e["states"], e["min_cells"])


def to_genes(a, genes):
    """The same cells on another encoder's gene list (zero-filled), keeping obs."""
    b = ad.AnnData(C.align(a.X, list(a.var_names), genes))
    b.var_names = list(genes)
    b.obs_names = a.obs_names
    b.obs = a.obs.copy()
    return b


def measure(enc, a, key, targets, reference, Z, n_steps=50, counts_space=False):
    """Package values for each target against `reference` ('rest' or a list of labels)."""
    res = recast.attribute(enc, a, key, target=list(targets), reference=reference, gate="dC",
                           centroid="mean_lognorm", qc="off", embeddings=Z, n_steps=n_steps)
    lab = a.obs[key].astype(str).to_numpy()
    centroid = mean_counts_centroid if counts_space else mean_lognorm_centroid
    out = {}
    for t in targets:
        tm = lab == t
        rm = ~tm if reference == "rest" else np.isin(lab, reference)
        dz = np.asarray(Z[tm], dtype="float64").mean(0) - np.asarray(Z[rm], dtype="float64").mean(0)
        P = np.vstack([centroid(a.X, tm), centroid(a.X, rm)]).astype("float32")
        with torch.no_grad():
            Ep = enc.torch_encode(torch.as_tensor(P)).detach().cpu().numpy().astype("float64")
        dE = Ep[0] - Ep[1]
        comp = res.completeness.loc[t]
        u = res.directions.loc[t].to_numpy(dtype="float64")
        out[t] = dict(n_target=int(tm.sum()), n_reference=int(rm.sum()), norm_dz=float(np.linalg.norm(dz)),
                      delta_profile=float(comp.delta_f), delta_check=float(u @ dE), cos_align=cos(dE, dz),
                      ratio=float(comp.delta_f) / float(np.linalg.norm(dz)), sum_phi=float(comp.sum_attribution),
                      completeness_rel=float(comp.rel_error))
    return out


def compare(rows, stored, keys):
    """Package rows joined to the original rows on `keys`, with absolute differences."""
    P = pd.DataFrame(rows)
    M = P.merge(stored[keys + COLS], on=keys, how="left", suffixes=("", "_stored"), validate="one_to_one")
    assert M["norm_dz_stored"].notna().all(), M[M["norm_dz_stored"].isna()][keys]
    for c in COLS:
        M[f"d_{c}"] = (M[c] - M[f"{c}_stored"]).abs()
    M["d_delta_check"] = (M.delta_check - M.delta_profile).abs()
    return M


def report(M, stage):
    same_n = int(((M.d_n_target == 0) & (M.d_n_reference == 0)).sum())
    print(f"[{stage}] {len(M)} comparisons; cell numbers equal {same_n}/{len(M)}")
    print("  max |package - original|: " + ", ".join(
        f"{c} {M[f'd_{c}'].max():.2e}" for c in COLS[2:]) + f"; u.dE vs delta_f {M.d_delta_check.max():.2e}")
    print(f"  median cos_align {M.cos_align.median():.4f} (original {M.cos_align_stored.median():.4f}); "
          f"median ratio {M.ratio.median():.4f} ({M.ratio_stored.median():.4f}); "
          f"max completeness_rel {M.completeness_rel.max():.2e} ({M.completeness_rel_stored.max():.2e})", flush=True)
    C.write(M, f"profile_centroid_{stage}.csv")


def original(cohort, encoder):
    D = pd.read_csv(os.path.join(PCA, "pc_align_v1.csv"))
    return D[(D.cohort == cohort) & (D.encoder == encoder)]


def stage_fine65(enc):
    rows = []
    for dsid in fine_lineages():
        a, keep = lineage(dsid)
        Z = enc.embed(a.X)
        for t, d in measure(enc, a, "state", keep, "rest", Z).items():
            rows.append(dict(lineage=dsid, target=t, **d))
        print(f"  {dsid}: {len(keep)} states", flush=True)
        C.assert_package_only()
    report(compare(rows, original("fine65", "SCimilarity"), ["lineage", "target"]), "fine65")


def stage_steps(enc):
    stored = pd.read_csv(os.path.join(PCA, "pc_align_steps_v1.csv"))
    rows = []
    for dsid, g in stored.groupby("lineage", sort=False):
        a, keep = lineage(dsid)
        Z = enc.embed(a.X)
        for t in g.target.unique():
            for n in (50, 200, 1000):
                d = measure(enc, a, "state", [t], "rest", Z, n_steps=n)[t]
                rows.append(dict(lineage=dsid, target=t, n_steps=n, **d))
                print(f"  {dsid} / {t}: {n} steps, relative error {d['completeness_rel']:.2e}", flush=True)
    report(compare(rows, stored, ["lineage", "target", "n_steps"]), "steps")


def stage_zhaoref(enc):
    a, Xln, obs = RC.dataset("zhao")
    lab = obs["label"].astype(str).to_numpy()
    Z = enc.embed(a.X)
    rng = np.random.default_rng(0)
    rows = []
    for tag, (sibs, broad, wrong) in RC.LIN.items():
        in_lin = np.isin(lab, sibs)
        for t in sibs:
            tm = lab == t
            tgt_idx = np.where(tm)[0]
            if len(tgt_idx) < 20:
                continue
            others = [s for s in sibs if s != t]
            tvec = np.asarray(Xln[tm].mean(0)).ravel()
            nearest = max(others, key=lambda s: np.corrcoef(tvec, np.asarray(Xln[lab == s].mean(0)).ravel())[0, 1])
            nsib = int(in_lin.sum() - tm.sum())
            d2 = ((Z - Z[tm].mean(0)) ** 2).sum(1); d2[tm] = np.inf
            nontgt = np.where(~tm)[0]
            refs = {"nearest": np.where(lab == nearest)[0],
                    "sibling": np.where(np.isin(lab, others))[0],
                    "broad": np.where(np.isin(lab, [s for s in RC.BROAD[broad] if s != t]))[0],
                    "global": nontgt,
                    "geometry": np.argsort(d2)[:max(nsib, 20)],
                    "random": rng.choice(nontgt, size=max(nsib, 20), replace=False),
                    "wrong": np.where(np.isin(lab, RC.BROAD[wrong]))[0]}
            for rn, ridx in refs.items():
                if len(ridx) < 5:
                    continue
                grp = np.full(a.n_obs, "other", dtype=object)
                grp[ridx] = "R"
                grp[tgt_idx] = "T"
                a.obs["grp"] = pd.Categorical(grp.astype(str))
                d = measure(enc, a, "grp", ["T"], ["R"], Z)["T"]
                rows.append(dict(lineage=tag, target=t, reference=rn, **d))
            print(f"  {tag} / {t}", flush=True)
    report(compare(rows, original("zhaoref", "SCimilarity"), ["lineage", "target", "reference"]), "zhaoref")


def stage_ssl():
    rows = []
    for name, path in SSL:
        enc = recast.SSLEncoder(path, device="cuda")
        genes = pd.read_parquet(os.path.join(path, "var.parquet"))["feature_name"].astype(str).tolist()
        for dsid in fine_lineages():
            a, keep = lineage(dsid)
            b = to_genes(a, genes)
            Z = enc.embed(b.X)
            for t, d in measure(enc, b, "state", keep, "rest", Z).items():
                rows.append(dict(encoder=name, lineage=dsid, target=t, **d))
            print(f"  [{name}] {dsid}", flush=True)
        del enc; torch.cuda.empty_cache()
    D = pd.read_csv(os.path.join(PCA, "pc_align_v1.csv"))
    stored = D[(D.cohort == "fine65") & D.encoder.isin([n for n, _ in SSL])]
    report(compare(rows, stored, ["encoder", "lineage", "target"]), "ssl")


def hvg_union(X, sym, marker_names, n_top=4000):
    """The original scVI gene set: the 4,000 genes with the highest count variance-to-mean ratio among genes
    with mean count > 0.01, plus the curated marker genes (subtype_benchmark/encoder_swap.py)."""
    mean = np.asarray(X.mean(0)).ravel()
    sq = np.asarray(X.multiply(X).mean(0)).ravel()
    disp = np.clip(sq - mean ** 2, 0, None) / (mean + 1e-8)
    order = np.argsort(-np.where(mean > 0.01, disp, -np.inf))
    return np.array(sorted(set(order[:n_top].tolist()) | {i for i, g in enumerate(sym) if g in marker_names}))


SAME = ["delta_profile", "cos_align", "ratio", "sum_phi", "completeness_rel"]


def original_scvi(m, a, key, targets):
    """alignment.py's scVI stage on a given model: latents from get_latent_representation, mean raw-count
    profiles, u from the latent means, E = z_encoder(log1p(x)).loc, 50-step IG of <E(x), u>."""
    from captum.attr import IntegratedGradients
    Z = m.get_latent_representation().astype("float32")
    zenc = m.module.z_encoder.eval()
    dev = next(zenc.parameters()).device
    lv = bool(getattr(m.module, "log_variational", True))

    def fwd(x):
        out = zenc(torch.log1p(x) if lv else x)
        first = out[0] if isinstance(out, (tuple, list)) else out
        return first.loc if hasattr(first, "loc") else first

    class Proj(torch.nn.Module):
        def forward(self, x, u):
            s = (fwd(x) * u).sum(1)
            return torch.stack([s, s], 1)

    lab = a.obs[key].astype(str).to_numpy()
    Cd = np.asarray(sp.csr_matrix(a.X).todense(), dtype="float64")
    tot = Cd.sum(0)
    out = {}
    for t in targets:
        tm = lab == t
        dz = Z[tm].astype("float64").mean(0) - Z[~tm].astype("float64").mean(0)
        ndz = float(np.linalg.norm(dz))
        u = (dz / ndz).astype("float32")
        cs = Cd[tm].sum(0)
        xt = torch.as_tensor((cs / tm.sum()).astype("float32")[None], device=dev)
        xr = torch.as_tensor(((tot - cs) / (~tm).sum()).astype("float32")[None], device=dev)
        with torch.no_grad():
            dE = fwd(xt).cpu().numpy().ravel().astype("float64") - fwd(xr).cpu().numpy().ravel().astype("float64")
        delta = float(u.astype("float64") @ dE)
        att = IntegratedGradients(Proj(), multiply_by_inputs=True).attribute(
            xt.clone().requires_grad_(True), baselines=xr, target=0,
            additional_forward_args=(torch.as_tensor(u[None], device=dev),), n_steps=50)
        sum_phi = float(att.detach().cpu().numpy().ravel().astype("float64").sum())
        out[t] = dict(delta_profile=delta, cos_align=cos(dE, dz), ratio=delta / ndz, sum_phi=sum_phi,
                      completeness_rel=abs(sum_phi - delta) / max(abs(delta), 1e-12))
    return Z, out


def stage_scvi(epochs=200):
    import scvi
    markers = set()
    for v in C.config("encoder_lineages").CFG.values():
        for genes in v.get("curated", {}).values():
            markers |= set(genes)
    rows = []
    for dsid in fine_lineages():
        a0, keep = lineage(dsid)
        scvi.settings.seed = 0
        sym = np.asarray(a0.var_names)
        cols = hvg_union(sp.csr_matrix(a0.X), sym, markers)
        a = ad.AnnData(X=sp.csr_matrix(a0.X)[:, cols].astype("float32").copy())
        a.var_names = [str(g) for g in sym[cols]]
        a.obs_names = [f"c{i}" for i in range(a.n_obs)]
        a.layers["counts"] = a.X.copy()
        scvi.model.SCVI.setup_anndata(a, layer="counts")
        m = scvi.model.SCVI(a, n_latent=10)
        bs = 128
        while a.n_obs % bs == 1 or int(a.n_obs * 0.9) % bs == 1:
            bs += 1
        m.train(max_epochs=epochs, early_stopping=True, accelerator="gpu", devices=1, enable_progress_bar=False,
                batch_size=bs)
        a.obs["state"] = a0.obs["state"].to_numpy()
        enc = recast.SCVIEncoder(m)
        Z = enc.embed(a.X)
        Zp, same = original_scvi(m, a, "state", keep)
        zd = float(np.abs(Zp - Z).max())
        for t, d in measure(enc, a, "state", keep, "rest", Z, counts_space=True).items():
            rows.append(dict(lineage=dsid, target=t, **d, max_abs_latent_diff=zd,
                             **{f"{c}_original_code": same[t][c] for c in SAME}))
        print(f"  [scVI] {dsid}: {a.n_obs} cells x {a.n_vars} genes", flush=True)
        del m, enc; torch.cuda.empty_cache()
    M = compare(rows, original("fine65", "scVI"), ["lineage", "target"])
    print("[scvi same model] max |package - original code|: " + ", ".join(
        f"{c} {(M[c] - M[f'{c}_original_code']).abs().max():.2e}" for c in SAME)
        + f"; max |latent difference| {M.max_abs_latent_diff.max():.2e}", flush=True)
    report(M, "scvi")


def stage_scf():
    CC = C.config("encoder_lineages")
    enc = recast.SCFoundationEncoder(SCF, device="cuda")
    rows = []
    for tag in CC.ORDER:
        c = CC.CFG[tag]
        states = list(c["curated"])
        names, obs = C.attribution_obs(C.ATLASES, c["stem"])
        if c["lab"]:
            lab_all = C.read_labels(os.path.join(C.ATLASES, c["lab"]))
        else:
            lab_all = obs["label"].astype(str).to_numpy()
        a, _ = C.lineage_adata(C.ATLASES, c["stem"], lab_all, states, min_cells=1)
        cl = [s for s in states if (a.obs["state"] == s).sum() >= 60]
        b = to_genes(a, enc.gene_order)
        Z = enc.embed(b.X)
        for t, d in measure(enc, b, "state", cl, "rest", Z).items():
            rows.append(dict(lineage=tag, target=t, **d))
        print(f"  [scFoundation] {tag}: {b.n_obs} cells", flush=True)
    report(compare(rows, original("crossfm6", "scFoundation"), ["lineage", "target"]), "scf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fine65", "steps", "zhaoref", "ssl", "scvi", "scf"])
    st = ap.parse_args().stage
    if st in ("fine65", "steps", "zhaoref"):
        {"fine65": stage_fine65, "steps": stage_steps, "zhaoref": stage_zhaoref}[st](C.scimilarity())
    else:
        {"ssl": stage_ssl, "scvi": stage_scvi, "scf": stage_scf}[st]()
    C.assert_package_only()


if __name__ == "__main__":
    main()
