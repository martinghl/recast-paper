#!/usr/bin/env python
"""Profile endpoints against embedding centroids (Supplementary Fig. 2a-d).  Env SC, GPU.

RECAST takes the contrast direction from the mean single-cell embeddings,
    u = normalize(zbar_T - zbar_R),   zbar_A = mean_i E(x_i),
but attributes, with integrated gradients along C_R -> C_T, the encoder change between the mean-expression profiles
projected onto u:
    Delta_profile = u^T [E(C_T) - E(C_R)],   C_A = mean_i x_i   (per-cell CP10k + log1p, then mean).
E is nonlinear, so E(C_A) != zbar_A and Delta_profile is not the centroid separation ||zbar_T - zbar_R||.  For every
comparison the manuscript attributes:

    cos_align        = cos( E(C_T) - E(C_R),  zbar_T - zbar_R )        direction agreement
    ratio            = u^T [E(C_T) - E(C_R)] / ||zbar_T - zbar_R||     projection ratio (profile / centroid)
    sign_pos         = Delta_profile > 0                               sign agreement
    completeness_abs = | sum_g phi_g - Delta_profile |                 numerical error of the 50-step path integral
    completeness_rel = completeness_abs / |Delta_profile|

Each object is built with the code the manuscript used for that set of comparisons:
    fine65    21 lineages / 65 sibling comparisons (the 61 benchmark states + the 4-state Baron control), SCimilarity,
              through select_recast.load_lineage + recast.SCimilarityEncoder + recast.contrast.contrast_direction +
              recast.centroid.LabelProfiles("mean_lognorm") + recast's centroid_module (reference = the rest of the
              lineage).
    zhaoref   Fig. 4's 16 Zhao targets x 7 references (nearest / sibling / broad / global / geometry / random /
              wrong), the objects of validation/reference_choice/reference_choice.py (contrastive_ig.embed,
              recast_centroid).
    ssl       the 65 sibling comparisons through the scTab_SSL and PBMC_SSL encoders (subtype_benchmark/encoder_swap.py, run_ssl).
    scvi      the 65 sibling comparisons through a scVI model trained per lineage as in subtype_benchmark/encoder_swap.py (run_scvi)
              (seed 0, 200 epochs, 10-d) -- a retrained model.
    scf       the 6 cross-encoder lineages through scFoundation (encoders_scfoundation.py).

Usage:  CUDA_VISIBLE_DEVICES=1 python alignment.py --stage fine65|zhaoref|ssl|scvi|scf|steps|normfloor|merge
Outputs: results/pc_align_<stage>_v1.csv; `merge` writes results/pc_align_v1.csv + pc_align_summary_v1.csv."""
import argparse, gc, os, sys, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, torch, torch.nn as nn

SB = "/path/to/research/analysis"
for p in (SB, f"{SB}/subtype_benchmark", f"{SB}/attribution_methods", f"{SB}/validation",
          f"{SB}/scoring_benchmark", "/path/to/SIGnature/src", "/path/to/research/external/Signature/src",
          "/path/to/research/recast"):
    if p not in sys.path:
        sys.path.insert(0, p)
HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; os.makedirs(RES, exist_ok=True)
ATLASES = f"{SB}/atlases"
N_STEPS = 50
CONTROL_LINEAGES = {"fs_panc_baron"}          # distinct-type control, not one of the 20 benchmark lineages


class _Proj(nn.Module):
    """f(x, u) = <fwd(x), u>, duplicated into two channels for Captum (target=0) -- the shape every RECAST
    centroid module here uses (recast.encoders.Encoder.centroid_module, contrastive_ig.CentroidTarget,
    SSLCentroid / SCVICentroid of encoder_swap.py, SCFCentroid of encoders_scfoundation.py)."""
    def __init__(self, fwd):
        super().__init__(); self.fwd = fwd
    def forward(self, x, u):
        s = (self.fwd(x) * u).sum(1)
        return torch.stack([s, s], 1)


def diag(fwd, dev, Z_t, Z_r, C_t, C_r, n_steps=N_STEPS):
    """One contrast.  fwd: torch callable, (1, p) float32 on `dev` -> (1, d) embedding, the encoder forward the
    cohort's attribution used.  Z_t / Z_r: the cohort's per-cell embeddings of target / reference cells.
    C_t / C_r: the cohort's population profiles (the IG input and baseline)."""
    from captum.attr import IntegratedGradients
    zt, zr = np.asarray(Z_t, dtype="float64").mean(0), np.asarray(Z_r, dtype="float64").mean(0)
    dz = zt - zr; ndz = float(np.linalg.norm(dz))
    u = (dz / ndz).astype("float32")                                   # recast.contrast.contrast_direction
    xt = torch.as_tensor(np.asarray(C_t, dtype="float32")[None], device=dev)
    xr = torch.as_tensor(np.asarray(C_r, dtype="float32")[None], device=dev)
    with torch.no_grad():
        Et = fwd(xt).detach().cpu().numpy().ravel().astype("float64")
        Er = fwd(xr).detach().cpu().numpy().ravel().astype("float64")
    dE = Et - Er; ndE = float(np.linalg.norm(dE))
    delta_profile = float(u.astype("float64") @ dE)
    ig = IntegratedGradients(_Proj(fwd), multiply_by_inputs=True)
    x = xt.clone().requires_grad_(True)
    att, cdelta = ig.attribute(x, baselines=xr, target=0, additional_forward_args=(torch.as_tensor(u[None], device=dev),),
                               n_steps=n_steps, return_convergence_delta=True)
    phi = att.detach().cpu().numpy().ravel().astype("float64"); sum_phi = float(phi.sum())
    # d' along u from the cohort's own embeddings (the contrast-QC separation statistic)
    pt, pr = np.asarray(Z_t, dtype="float64") @ (dz / ndz), np.asarray(Z_r, dtype="float64") @ (dz / ndz)
    sd = np.sqrt(0.5 * (pt.var() + pr.var())); dprime = float((pt.mean() - pr.mean()) / sd) if sd > 0 else np.nan
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return dict(n_target=int(len(Z_t)), n_reference=int(len(Z_r)), dim=int(len(zt)),
                norm_dz=ndz, norm_dE=ndE, delta_profile=delta_profile,
                cos_align=cos(dE, dz), ratio=delta_profile / ndz, sign_pos=bool(delta_profile > 0),
                sum_phi=sum_phi, captum_delta=float(np.asarray(cdelta.detach().cpu()).ravel()[0]),
                completeness_abs=abs(sum_phi - delta_profile),
                completeness_rel=abs(sum_phi - delta_profile) / max(abs(delta_profile), 1e-12),
                cos_ECt_zt=cos(Et, zt), cos_ECr_zr=cos(Er, zr),
                dist_ECt_zt=float(np.linalg.norm(Et - zt)), dist_ECr_zr=float(np.linalg.norm(Er - zr)),
                norm_zt=float(np.linalg.norm(zt)), norm_Et=float(np.linalg.norm(Et)),
                dprime=dprime, n_steps=int(n_steps))


def _save(rows, stage):
    df = pd.DataFrame(rows); out = f"{RES}/pc_align_{stage}_v1.csv"; df.to_csv(out, index=False)
    print(f"[{stage}] {len(df)} contrasts -> {out}", flush=True)
    if len(df):
        print(df[["cos_align", "ratio", "completeness_rel"]].describe().T[["min", "50%", "max"]].to_string(), flush=True)
        print(f"  sign_pos: {df.sign_pos.mean():.3f}   cos>=0.9: {(df.cos_align >= 0.9).mean():.3f}   cos>=0.95: {(df.cos_align >= 0.95).mean():.3f}", flush=True)
    return df


# ------------------------------------------------------------------------------- fine65 (SCimilarity, library objects)
def stage_fine65():
    import recast
    import select_recast as FS
    from recast.contrast import resolve_reference
    from recast.centroid import LabelProfiles
    from datasets import SCIMILARITY_MODEL
    from run_scoring_benchmark import _PerCellLognormEncoder
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda"))
    dev = torch.device("cuda")
    lineages = sorted(pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv").lineage.unique())
    rows = []
    for dsid in lineages:
        t0 = time.time()
        Craw, lab, keepstates, go = FS.load_lineage(dsid)
        Z = np.asarray(enc.embed(Craw))                                  # ONE encoder pass (attribute() l.94)
        profiles = LabelProfiles(Craw, lab, "mean_lognorm")             # ONE pass over the matrix (l.97)
        for s in keepstates:
            tmask, rmask = resolve_reference(lab, s, "rest")            # siblings = the rest of the lineage
            d = diag(enc.torch_encode, dev, Z[tmask], Z[rmask], profiles.centroid(tmask), profiles.centroid(rmask))
            rows.append(dict(cohort="fine65", encoder="SCimilarity", lineage=dsid, target=s, reference="sibling",
                             tier="control" if dsid in CONTROL_LINEAGES else "headline", **d))
        print(f"  {dsid}: {len(keepstates)} states, {Craw.shape[0]} cells, {time.time() - t0:.0f}s", flush=True)
        del Craw, Z, profiles; gc.collect(); torch.cuda.empty_cache()
    return _save(rows, "fine65")


# ------------------------------------------------------------------------------- zhaoref (Fig. 4 objects)
def stage_zhaoref():
    import real_counts, contrastive_ig
    import recast_centroid as fc
    from reference_choice.definitions import BROAD, LIN   # inlined copies (reference_choice.py runs at import)
    att = ad.read_h5ad(f"{ATLASES}/zhao_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    lab = att.obs["label"].astype(str).to_numpy(); obs = att.obs_names; del att
    cnt = real_counts.load("zhao", obs)
    Z = contrastive_ig.embed(Xln); CENT = fc.Centroider(cnt); dev = contrastive_ig.dev; fwd = contrastive_ig.enc
    RNG = np.random.default_rng(0)
    rows = []
    for tag, (sibs, broad, wrong) in LIN.items():
        in_lin = np.isin(lab, sibs)
        for t in sibs:
            tm = lab == t; tgt_idx = np.where(tm)[0]
            if len(tgt_idx) < 20:
                continue
            others = sibs[:]; others.remove(t)
            tvec = Xln[tm].mean(0)
            nearest = max(others, key=lambda s: np.corrcoef(tvec, Xln[lab == s].mean(0))[0, 1]) if others else t
            nsib = int(in_lin.sum() - tm.sum())
            cen = Z[tm].mean(0); d2 = ((Z - cen) ** 2).sum(1); d2[tm] = np.inf
            geo_idx = np.argsort(d2)[:max(nsib, 20)]
            nontgt = np.where(~tm)[0]
            rand_idx = RNG.choice(nontgt, size=max(nsib, 20), replace=False)
            REFS = {"nearest": np.where(lab == nearest)[0], "sibling": np.where(np.isin(lab, others))[0],
                    "broad": np.where(np.isin(lab, [s for s in BROAD[broad] if s != t]))[0], "global": nontgt,
                    "geometry": geo_idx, "random": rand_idx, "wrong": np.where(np.isin(lab, BROAD[wrong]))[0]}
            for rn, ridx in REFS.items():
                if len(ridx) < 5:
                    continue
                d = diag(fwd, dev, Z[tgt_idx], Z[ridx], CENT(tgt_idx), CENT(ridx))
                rows.append(dict(cohort="zhaoref", encoder="SCimilarity", lineage=tag, target=t, reference=rn, tier="reference_choice", **d))
            print(f"  {tag}/{t}: {len(REFS)} references", flush=True)
    return _save(rows, "zhaoref")


# ------------------------------------------------------------------------------- ssl (encoder_swap.py objects)
def stage_ssl():
    import real_counts
    import recast_centroid as fc
    import select_recast as FS
    from SIGnature.models.ssl import SSLWrapper
    from encoder_swap import SSL_PATHS, embed_batched, densify, states_iter
    lineages = sorted(pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv").lineage.unique())
    rows = []
    for name, path in SSL_PATHS:
        w = SSLWrapper(path); enc = w.model.eval(); dev = next(w.model.parameters()).device
        for dsid in lineages:
            t0 = time.time()
            Craw, lab, keepstates, go = FS.load_lineage(dsid)
            a_ = ad.AnnData(X=Craw.copy()); a_.var_names = go; a_.obs_names = [f"c{i}" for i in range(Craw.shape[0])]
            a_.layers["counts"] = a_.X.copy()
            adh = w.preprocess_adata(a_); genes_fm = list(map(str, adh.var_names)); Xh = adh.X
            C_fm = real_counts.align_to(Craw, go, genes_fm); CENTf = fc.Centroider(C_fm)
            Z = embed_batched(enc, dev, Xh, densify)
            for st, tgt in states_iter(lab, keepstates):
                d = diag(enc, dev, Z[tgt], Z[~tgt], CENTf(tgt), CENTf(~tgt))
                rows.append(dict(cohort="fine65", encoder=name, lineage=dsid, target=st, reference="sibling",
                                 tier="control" if dsid in CONTROL_LINEAGES else "headline", **d))
            print(f"  [{name}] {dsid}: {len(keepstates)} states, {time.time() - t0:.0f}s", flush=True)
            del a_, adh, Xh, C_fm, Z, Craw; gc.collect(); torch.cuda.empty_cache()
        del w, enc; gc.collect(); torch.cuda.empty_cache()
    return _save(rows, "ssl")


# ------------------------------------------------------------------------------- scvi (encoder_swap.py recipe, retrained)
def stage_scvi(epochs=200):
    import scvi
    import select_recast as FS
    from encoder_swap import hvg_union, states_iter
    import encoder_lineages as CC
    lineages = sorted(pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv").lineage.unique())
    # encoder_swap.py passes the lineage's curated marker names into hvg_union (kept genes = top-4000 dispersion U markers)
    gt_all = set()
    for k, v in getattr(CC, "CFG", {}).items():
        for st, genes in v.get("curated", {}).items():
            gt_all |= set(genes)
    rows = []
    for dsid in lineages:
        t0 = time.time()
        Craw, lab, keepstates, go = FS.load_lineage(dsid)
        scvi.settings.seed = 0
        sym = np.asarray(go); keep = hvg_union(Craw, sym, gt_all); genes_v = [str(g) for g in sym[keep]]
        a = ad.AnnData(X=Craw[:, keep].astype("float32").copy()); a.var_names = genes_v
        a.obs_names = [f"c{i}" for i in range(a.n_obs)]; a.layers["counts"] = a.X.copy()
        scvi.model.SCVI.setup_anndata(a, layer="counts")
        m = scvi.model.SCVI(a, n_latent=10)
        bs = 128
        while a.n_obs % bs == 1 or int(a.n_obs * 0.9) % bs == 1:
            bs += 1
        m.train(max_epochs=epochs, early_stopping=True, accelerator="gpu", devices=1, enable_progress_bar=False, batch_size=bs)
        Z = m.get_latent_representation().astype("float32")
        zenc = m.module.z_encoder.eval(); dev = next(zenc.parameters()).device
        lv = bool(getattr(m.module, "log_variational", True))
        def fwd(x, zenc=zenc, lv=lv):
            out = zenc(torch.log1p(x) if lv else x)
            first = out[0] if isinstance(out, (tuple, list)) else out
            return first.loc if hasattr(first, "loc") else first
        Cd = np.asarray(Craw[:, keep].todense(), dtype="float64"); tot = Cd.sum(0)
        for st, tgt in states_iter(lab, keepstates):
            cs = Cd[tgt].sum(0)
            Cc = (cs / max(int(tgt.sum()), 1)).astype("float32"); Cr = ((tot - cs) / max(int((~tgt).sum()), 1)).astype("float32")
            d = diag(fwd, dev, Z[tgt], Z[~tgt], Cc, Cr)
            rows.append(dict(cohort="fine65", encoder="scVI", lineage=dsid, target=st, reference="sibling",
                             tier="control" if dsid in CONTROL_LINEAGES else "headline", **d))
        print(f"  [scVI] {dsid}: {len(keepstates)} states, {a.n_obs} cells x {len(genes_v)} genes, {time.time() - t0:.0f}s", flush=True)
        del a, m, Cd, Z, Craw; gc.collect(); torch.cuda.empty_cache()
    return _save(rows, "scvi")


# ------------------------------------------------------------------------------- scf (encoders_scfoundation objects)
def stage_scf():
    import real_counts
    import recast_centroid as fc
    from SIGnature.models.scfoundation import SCFoundationWrapper
    from encoder_lineages import CFG, ORDER
    SCF_PATH = f"{SB}/models/prepared/scfoundation"
    w = SCFoundationWrapper(model_path=SCF_PATH, use_gpu=True, pool_type="all")
    dev = next(w.model.parameters()).device; NG = w.n_genes

    def embed(Xln, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, Xln.shape[0], bs):
                xb = torch.tensor(Xln[i:i + bs], dtype=torch.float32, device=dev)
                gid = torch.arange(NG, device=dev).unsqueeze(0).repeat(xb.shape[0], 1)
                out.append(w._get_geneembmerge(xb, gid).detach().cpu().numpy())
        return np.vstack(out)

    def load_lin(stem, lab_file, states):
        Aa = ad.read_h5ad(f"{ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r"); n = Aa.n_obs
        lab_all = np.array([l.strip() for l in open(f"{ATLASES}/{lab_file}") if l.strip()]) if lab_file else (
            Aa.obs["label"].astype(str).to_numpy() if "label" in Aa.obs.columns else
            np.array([l.strip() for l in open(f"{ATLASES}/bench_labels_{stem}.txt") if l.strip()]))
        assert len(lab_all) == n
        mask = np.isin(lab_all, states); sub = Aa[mask].to_memory()
        cnt = real_counts.load(stem, sub.obs_names)
        sym = np.asarray(sub.var_names).astype(str); lab = lab_all[mask]; del Aa, sub
        return cnt, sym, lab

    gid1 = torch.arange(NG, device=dev).unsqueeze(0)
    fwd = lambda x: w._get_geneembmerge(x, gid1.repeat(x.shape[0], 1))
    rows = []
    for tag in ORDER:
        c = CFG[tag]; states = list(c["curated"])
        cnt_sc, sym, lab = load_lin(c["stem"], c["lab"], states)
        cl = [s for s in states if (lab == s).sum() >= 60]
        if len(cl) < 2:
            continue
        a_ = ad.AnnData(X=cnt_sc.copy()); a_.var_names = sym
        a_.obs_names = [f"c{i}" for i in range(cnt_sc.shape[0])]; a_.layers["counts"] = a_.X.copy()
        adh = w.preprocess_adata(a_); genes = np.asarray(adh.var_names).astype(str)
        Xln = (adh.X.toarray() if sp.issparse(adh.X) else np.asarray(adh.X)).astype("float32")
        C = real_counts.align_to(cnt_sc, sym, genes); CENT = fc.Centroider(C)
        t0 = time.time(); Z = embed(Xln); print(f"  [scF] {tag}: embedded {Xln.shape[0]} cells in {time.time() - t0:.0f}s", flush=True)
        for cc in cl:
            tgt = lab == cc
            d = diag(fwd, dev, Z[tgt], Z[~tgt], CENT(tgt), CENT(~tgt))
            rows.append(dict(cohort="crossfm6", encoder="scFoundation", lineage=tag, target=cc, reference="sibling", tier="crossenc", **d))
        del a_, adh, Xln, C, Z; gc.collect(); torch.cuda.empty_cache()
    return _save(rows, "scf")


def stage_merge():
    parts = [pd.read_csv(f"{RES}/pc_align_{s}_v1.csv") for s in ("fine65", "zhaoref", "ssl", "scvi", "scf")
             if os.path.exists(f"{RES}/pc_align_{s}_v1.csv")]
    df = pd.concat(parts, ignore_index=True)
    # the manuscript's 61 benchmark states = the per-state benchmark rows (states with a subtype-unique CellMarker
    # panel); load_lineage attributes every >=50-cell state of each lineage, so fine65 holds a few more contrasts
    # than the benchmark scores.  Flag them so the text can quote either cohort exactly.
    bench = pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv")
    bench = bench[bench.method.astype(str).str.fullmatch("RECAST_M0|recast_M0|RECAST_M0")][["lineage", "state"]].drop_duplicates()
    key = set(zip(bench.lineage, bench.state))
    df["in_benchmark"] = [(l, t) in key and c == "fine65" for l, t, c in zip(df.lineage, df.target, df.cohort)]
    df["disp_rel_T"] = df.dist_ECt_zt / df.norm_zt          # ||E(C_T) - zbar_T|| / ||zbar_T||: the Jensen gap of the encoder
    df.to_csv(f"{RES}/pc_align_v1.csv", index=False)
    g = df.groupby(["cohort", "encoder"])
    summ = pd.DataFrame({"n_contrasts": g.size(),
                         "cos_align_median": g.cos_align.median(), "cos_align_min": g.cos_align.min(),
                         "cos_align_q05": g.cos_align.quantile(0.05),
                         "frac_cos_ge_0.9": g.cos_align.apply(lambda s: (s >= 0.9).mean()),
                         "frac_cos_ge_0.95": g.cos_align.apply(lambda s: (s >= 0.95).mean()),
                         "frac_sign_pos": g.sign_pos.mean(),
                         "ratio_median": g.ratio.median(), "ratio_min": g.ratio.min(), "ratio_max": g.ratio.max(),
                         "completeness_rel_median": g.completeness_rel.median(),
                         "completeness_rel_max": g.completeness_rel.max(),
                         "completeness_abs_max": g.completeness_abs.max(),
                         "disp_rel_T_median": g.disp_rel_T.median()}).reset_index()
    summ.to_csv(f"{RES}/pc_align_summary_v1.csv", index=False)
    print(summ.to_string(), flush=True)
    return summ


# ------------------------------------------------------------------------------- steps (numerical error vs path resolution)
def stage_steps(top=8):
    """Recompute the SCimilarity contrasts with the largest 50-step completeness error at 50 / 200 / 1000 steps.
    If the residual is the discretization error of the path integral it must shrink with the step count."""
    import recast
    import select_recast as FS
    from recast.contrast import resolve_reference
    from recast.centroid import LabelProfiles
    from datasets import SCIMILARITY_MODEL
    from run_scoring_benchmark import _PerCellLognormEncoder
    f = pd.read_csv(f"{RES}/pc_align_fine65_v1.csv").sort_values("completeness_rel", ascending=False).head(top)
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda")); dev = torch.device("cuda")
    rows = []
    for dsid, grp in f.groupby("lineage", sort=False):
        Craw, lab, keepstates, go = FS.load_lineage(dsid)
        Z = np.asarray(enc.embed(Craw)); profiles = LabelProfiles(Craw, lab, "mean_lognorm")
        for s in grp.target:
            tmask, rmask = resolve_reference(lab, s, "rest")
            for n in (50, 200, 1000):
                d = diag(enc.torch_encode, dev, Z[tmask], Z[rmask], profiles.centroid(tmask), profiles.centroid(rmask), n_steps=n)
                rows.append(dict(cohort="fine65", encoder="SCimilarity", lineage=dsid, target=s, reference="sibling", **d))
                print(f"  {dsid}/{s} n_steps={n}: completeness_rel={d['completeness_rel']:.5f}", flush=True)
        del Craw, Z, profiles; gc.collect(); torch.cuda.empty_cache()
    out = pd.DataFrame(rows); out.to_csv(f"{RES}/pc_align_steps_v1.csv", index=False)
    print(out.pivot_table(index=["lineage", "target"], columns="n_steps", values="completeness_rel").to_string(), flush=True)
    return out



# ------------------------------------------------------------------------------- normfloor
# SCimilarity's Encoder ends with F.normalize(h(x), p=2): E(x) = h(x)/||h(x)||.  Supplementary Note 1's stability
# corollary for this encoder needs a floor ||h|| >= m > 0 on the attribution paths (the normalization Jacobian scales
# with 1/||h||).  This stage MEASURES m instead of assuming it: ||h|| at every cell of the fine-benchmark lineages and
# of the Zhao atlas, at every profile endpoint, and at 101 equally spaced points of every IG segment C_R -> C_T of the
# fine65 (74) and zhaoref (112) contrasts -- the same objects as the alignment stages above.
def _pre_net(mod):
    """h(x): the SCimilarity Encoder's layers before the output F.normalize (scimilarity.nn_models.Encoder.forward)."""
    for _ in range(5):
        if hasattr(mod, "network"):
            net = mod.network
            def pre(x, net=net):
                for layer in net:
                    x = layer(x)
                return x
            return pre
        nxt = None
        for a in ("_net", "_base", "model"):
            if getattr(mod, a, None) is not None:
                nxt = getattr(mod, a); break
        if nxt is None:
            break
        mod = nxt
    raise AttributeError("no scimilarity Encoder .network found under the encoder object")


def _hnorms(pre, dev, X, lognorm=False, bs=4096):
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            B = X[i:i + bs]; B = B.toarray() if sp.issparse(B) else np.asarray(B)
            B = B.astype("float32")
            if lognorm:                                                   # _PerCellLognormEncoder.embed's per-cell CP10k + log1p
                tot = B.sum(1, keepdims=True); tot[tot == 0] = 1.0
                B = np.log1p(1e4 * B / tot).astype("float32")
            out.append(pre(torch.as_tensor(B, device=dev)).norm(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype="float32")


def _path_hnorms(pre, dev, C_t, C_r, n=101):
    ct, cr = np.asarray(C_t, dtype="float32"), np.asarray(C_r, dtype="float32")
    ts = np.linspace(0.0, 1.0, n, dtype="float32")
    P = cr[None] + ts[:, None] * (ct - cr)[None]
    h = _hnorms(pre, dev, P)
    return dict(hnorm_path_min=float(h.min()), hnorm_path_median=float(np.median(h)), hnorm_path_max=float(h.max()),
                hnorm_R=float(h[0]), hnorm_T=float(h[-1]), n_path_points=int(n))


def _cellstats(h):
    return dict(n_cells=int(len(h)), hcells_min=float(h.min()), hcells_p01=float(np.percentile(h, 1)),
                hcells_median=float(np.median(h)), hcells_max=float(h.max()))


def stage_normfloor():
    import recast
    import select_recast as FS
    from recast.contrast import resolve_reference
    from recast.centroid import LabelProfiles
    from datasets import SCIMILARITY_MODEL
    from run_scoring_benchmark import _PerCellLognormEncoder
    enc = _PerCellLognormEncoder(recast.SCimilarityEncoder(SCIMILARITY_MODEL, device="cuda")); dev = torch.device("cuda")
    pre = _pre_net(enc)
    # sanity: normalize(pre(x)) must equal the encoder forward the alignment stages used
    n_in = next(mm for mm in enc._base._net.modules() if isinstance(mm, nn.Linear)).in_features
    x0 = torch.rand(2, n_in, device=dev)
    with torch.no_grad():
        e0 = enc.torch_encode(x0); h0 = pre(x0)
    assert torch.allclose(e0, torch.nn.functional.normalize(h0, p=2, dim=1), atol=1e-5), "pre-net does not reproduce the encoder"
    bench = pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv")
    bench = bench[bench.method.astype(str).str.fullmatch("RECAST_M0|recast_M0|RECAST_M0")][["lineage", "state"]].drop_duplicates()
    key = set(zip(bench.lineage, bench.state))
    rows, cells = [], []
    lineages = sorted(pd.read_csv(f"{SB}/subtype_benchmark/results/fs_uniq_perstate_v3.csv").lineage.unique())
    for dsid in lineages:
        t0 = time.time()
        Craw, lab, keepstates, go = FS.load_lineage(dsid)
        hc = _hnorms(pre, dev, Craw, lognorm=True); cs = _cellstats(hc)
        cells.append(dict(cohort="fine65", lineage=dsid, **cs))
        profiles = LabelProfiles(Craw, lab, "mean_lognorm")
        for s in keepstates:
            tmask, rmask = resolve_reference(lab, s, "rest")
            d = _path_hnorms(pre, dev, profiles.centroid(tmask), profiles.centroid(rmask))
            rows.append(dict(cohort="fine65", encoder="SCimilarity", lineage=dsid, target=s, reference="sibling",
                             tier="control" if dsid in CONTROL_LINEAGES else "headline", in_benchmark=(dsid, s) in key, **d, **cs))
        print(f"  {dsid}: {len(keepstates)} states, {Craw.shape[0]} cells, cell ||h|| min {cs['hcells_min']:.2f} median {cs['hcells_median']:.2f}, {time.time() - t0:.0f}s", flush=True)
        del Craw, profiles; gc.collect(); torch.cuda.empty_cache()
    # zhaoref: the Fig. 4 objects, mirroring stage_zhaoref (same RNG seed and draw order -> the same random references)
    import real_counts, contrastive_ig
    import recast_centroid as fc
    from reference_choice.definitions import BROAD, LIN
    att = ad.read_h5ad(f"{ATLASES}/zhao_attribution_scimilarity.h5ad")
    Xln = (att.X.toarray() if sp.issparse(att.X) else np.asarray(att.X)).astype("float32")
    lab = att.obs["label"].astype(str).to_numpy(); obs = att.obs_names; del att
    cnt = real_counts.load("zhao", obs)
    Z = contrastive_ig.embed(Xln); CENT = fc.Centroider(cnt); zdev = contrastive_ig.dev; zpre = _pre_net(contrastive_ig.enc)
    with torch.no_grad():
        xz = torch.as_tensor(Xln[:2], device=zdev)
        assert torch.allclose(contrastive_ig.enc(xz), torch.nn.functional.normalize(zpre(xz), p=2, dim=1), atol=1e-5)
    hz = _hnorms(zpre, zdev, Xln); cz = _cellstats(hz); cells.append(dict(cohort="zhaoref", lineage="zhao", **cz))
    print(f"  zhao: {len(hz)} cells, cell ||h|| min {cz['hcells_min']:.2f} median {cz['hcells_median']:.2f}", flush=True)
    RNG = np.random.default_rng(0)
    for tag, (sibs, broad, wrong) in LIN.items():
        in_lin = np.isin(lab, sibs)
        for t in sibs:
            tm = lab == t; tgt_idx = np.where(tm)[0]
            if len(tgt_idx) < 20:
                continue
            others = sibs[:]; others.remove(t)
            tvec = Xln[tm].mean(0)
            nearest = max(others, key=lambda s: np.corrcoef(tvec, Xln[lab == s].mean(0))[0, 1]) if others else t
            nsib = int(in_lin.sum() - tm.sum())
            cen = Z[tm].mean(0); d2 = ((Z - cen) ** 2).sum(1); d2[tm] = np.inf
            geo_idx = np.argsort(d2)[:max(nsib, 20)]
            nontgt = np.where(~tm)[0]
            rand_idx = RNG.choice(nontgt, size=max(nsib, 20), replace=False)
            REFS = {"nearest": np.where(lab == nearest)[0], "sibling": np.where(np.isin(lab, others))[0],
                    "broad": np.where(np.isin(lab, [s for s in BROAD[broad] if s != t]))[0], "global": nontgt,
                    "geometry": geo_idx, "random": rand_idx, "wrong": np.where(np.isin(lab, BROAD[wrong]))[0]}
            for rn, ridx in REFS.items():
                if len(ridx) < 5:
                    continue
                d = _path_hnorms(zpre, zdev, CENT(tgt_idx), CENT(ridx))
                rows.append(dict(cohort="zhaoref", encoder="SCimilarity", lineage=tag, target=t, reference=rn, tier="reference_choice",
                                 in_benchmark=False, **d, **cz))
    df = pd.DataFrame(rows); df.to_csv(f"{RES}/pc_align_normfloor_v1.csv", index=False)
    pd.DataFrame(cells).to_csv(f"{RES}/pc_align_cellnorms_v1.csv", index=False)
    print(f"[normfloor] {len(df)} contrasts -> {RES}/pc_align_normfloor_v1.csv", flush=True)
    print(df.groupby("cohort")[["hnorm_path_min", "hnorm_path_median", "hnorm_T", "hnorm_R"]].agg(["min", "median"]).round(3).to_string(), flush=True)
    print(pd.DataFrame(cells)[["cohort", "lineage", "n_cells", "hcells_min", "hcells_p01", "hcells_median", "hcells_max"]].round(3).to_string(), flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True,
                                                   choices=["fine65", "zhaoref", "ssl", "scvi", "scf", "steps", "merge", "normfloor"])
    ap.add_argument("--epochs", type=int, default=200)
    A = ap.parse_args()
    {"fine65": stage_fine65, "zhaoref": stage_zhaoref, "ssl": stage_ssl, "scf": stage_scf, "merge": stage_merge,
     "steps": stage_steps, "normfloor": stage_normfloor, "scvi": lambda: stage_scvi(A.epochs)}[A.stage]()
