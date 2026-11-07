"""Encoder swap on the 65-subtype cohort: RECAST panels rebuilt with other encoders (helpers of
validation/profile_centroid/alignment.py).

Cohort: the 21-lineage / 65-subtype CellMarker-unique recall@20 benchmark of fs_uniq_perstate_v3.csv.  For each lineage
the RECAST panels are rebuilt with three other encoders plus an inline SCimilarity version, scored with the recall of
marker_recall_bootstrap.py, and placed among the other methods.

Cells: select_recast.load_lineage (same cells, labels, >= 50-cell subtypes, SCimilarity 28,231-gene order).
Selection per (lineage, subtype), as encoders_recast.py / encoders_scvi.py:
  u         = unit(mean z_subtype - mean z_siblings), z from the swapped encoder over the lineage cells
  IG        = captum IntegratedGradients of f(x) = <enc(x), u>, 50 steps, baseline = sibling centroid
  endpoints : SCimilarity / SSL = mean-lognorm centroids (recast_centroid) from the integer counts in that encoder's
              gene space via real_counts.align_to; scVI = raw-count mean profiles (it takes counts and log1p's
              internally)
  gate      = dC > 0, negatives clipped; panel = top 50
Encoders: SCim_inline (inline recipe, compared with the package panels), scTab_SSL, PBMC_SSL (SSLWrapper), scVI
(per-lineage training, HVG4000 union marker genes, n_latent = 10, <= 200 epochs with early stopping, seed 0).

Outputs: panels/topN_<fsid>_recast_M0_enc-<name>.csv, results/encoder_swap_perstate.csv,
results/encoder_swap_leaderboard.csv.  Existing panel files are skipped (delete one to rebuild it).
Env SC, GPU (CUDA_VISIBLE_DEVICES=1).  Hours; scVI training dominates."""
import argparse
import gc
import json
import os
import sys
from collections import Counter

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SB = "/path/to/research/analysis"
FSB = f"{SB}/subtype_benchmark"
for p in (SB, f"{SB}/ans_replication", FSB, f"{SB}/attribution_methods", f"{SB}/validation",
          "/path/to/SIGnature/src",
          "/path/to/research/external/Signature/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import real_counts                      # noqa: E402
import recast_centroid as fc             # noqa: E402
import select_recast as FS            # noqa: E402

R = f"{FSB}/results"
TOPN = 50
NSTEPS = 50
K_SCORE = 20
SSL_PATHS = [("scTab_SSL", f"{SB}/models/prepared/ssl_sctab"),
             ("PBMC_SSL", f"{SB}/models/prepared/ssl_pbmc")]
ENCS = ["SCim_inline", "scTab_SSL", "PBMC_SSL", "scVI"]
COHORT = sorted(pd.read_csv(f"{R}/fs_uniq_perstate_v3.csv").lineage.unique())


def unit(v):
    return (v / (np.linalg.norm(v) + 1e-12)).astype("float32")


def panel_path(panels_dir, fsid, enc):
    return f"{panels_dir}/topN_{fsid}_recast_M0_enc-{enc}.csv"


def write_panel(panels_dir, fsid, enc, per_state):
    rows = [dict(cluster=str(c), rank=i + 1, gene=str(g))
            for c in sorted(per_state) for i, g in enumerate(per_state[c])]
    pd.DataFrame(rows, columns=["cluster", "rank", "gene"]).to_csv(
        panel_path(panels_dir, fsid, enc), index=False)


def embed_batched(enc, dev, mat, transform, bs=256):
    import torch
    Z = []
    with torch.no_grad():
        for i in range(0, mat.shape[0], bs):
            x = transform(mat[i:i + bs])
            Z.append(enc(torch.tensor(x, dtype=torch.float32, device=dev)).cpu().numpy())
    return np.vstack(Z)


def lognorm_dense(C):
    d = np.asarray(C.todense(), dtype="float32") if sp.issparse(C) else np.asarray(C, "float32")
    d = np.log1p(1e4 * d / np.clip(d.sum(1, keepdims=True), 1, None))
    return d


def densify(C):
    return np.asarray(C.todense(), dtype="float32") if sp.issparse(C) else np.asarray(C, "float32")


def ig_top(igobj, dev, Cc, Cr, u, genes):
    import torch
    inp = torch.tensor(Cc[None], device=dev).requires_grad_(True)
    a = igobj.attribute(inp, baselines=torch.tensor(Cr[None], device=dev), target=0,
                        additional_forward_args=(torch.tensor(u[None], device=dev),),
                        n_steps=NSTEPS).detach().cpu().numpy().ravel()
    a = np.where((Cc - Cr) > 0, np.clip(a, 0, None), -np.inf)
    order = np.argsort(-a)[:TOPN]
    return [genes[j] for j in order if np.isfinite(a[j])]


def hvg_union(C, sym, marker_names, n_top=4000):
    mean = np.asarray(C.mean(0)).ravel()
    sq = np.asarray(C.multiply(C).mean(0)).ravel()
    disp = np.clip(sq - mean ** 2, 0, None) / (mean + 1e-8)
    order = np.argsort(-np.where(mean > 0.01, disp, -np.inf))
    return np.array(sorted(set(order[:n_top].tolist())
                           | {i for i, g in enumerate(sym) if g in marker_names}))


def states_iter(lab, keepstates):
    for st in keepstates:
        tgt = lab == st
        if tgt.sum() < 5 or (~tgt).sum() < 5:
            continue
        yield st, tgt


def run_scim_inline(fsid, Craw, lab, keepstates, genes_go, panels_dir):
    import torch  # noqa: F401
    import contrastive_ig
    from captum.attr import IntegratedGradients
    assert list(map(str, contrastive_ig.go)) == genes_go, "contrastive_ig gene order != loader gene order"
    CENT = fc.Centroider(Craw)
    Z = embed_batched(contrastive_ig.enc, contrastive_ig.dev, Craw, lognorm_dense)
    ig = IntegratedGradients(contrastive_ig.CentroidTarget(contrastive_ig.enc), multiply_by_inputs=True)
    per = {}
    for st, tgt in states_iter(lab, keepstates):
        u = unit(Z[tgt].mean(0) - Z[~tgt].mean(0))
        per[st] = ig_top(ig, contrastive_ig.dev, CENT(tgt), CENT(~tgt), u, genes_go)
    write_panel(panels_dir, fsid, "SCim_inline", per)
    del Z
    return per


def run_ssl(fsid, name, w, Craw, lab, keepstates, genes_go, panels_dir):
    import torch
    import torch.nn as nn
    from captum.attr import IntegratedGradients

    class SSLCentroid(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.e = e

        def forward(self, x, u):
            s = (self.e(x) * u).sum(1)
            return torch.stack([s, s], 1)

    enc = w.model.eval()
    dev = next(w.model.parameters()).device
    a_ = ad.AnnData(X=Craw.copy())
    a_.var_names = genes_go
    a_.obs_names = [f"c{i}" for i in range(Craw.shape[0])]
    a_.layers["counts"] = a_.X.copy()
    adh = w.preprocess_adata(a_)
    genes_fm = list(map(str, adh.var_names))
    Xh = adh.X
    C_fm = real_counts.align_to(Craw, genes_go, genes_fm)
    CENTf = fc.Centroider(C_fm)
    Z = embed_batched(enc, dev, Xh, densify)
    ig = IntegratedGradients(SSLCentroid(enc), multiply_by_inputs=True)
    per = {}
    for st, tgt in states_iter(lab, keepstates):
        u = unit(Z[tgt].mean(0) - Z[~tgt].mean(0))
        per[st] = ig_top(ig, dev, CENTf(tgt), CENTf(~tgt), u, genes_fm)
    write_panel(panels_dir, fsid, name, per)
    del a_, adh, Xh, C_fm, Z
    gc.collect()
    torch.cuda.empty_cache()
    return per


def run_scvi(fsid, Craw, lab, keepstates, genes_go, gt_genes, panels_dir, epochs):
    import torch
    import torch.nn as nn
    import scvi
    from captum.attr import IntegratedGradients

    class SCVICentroid(nn.Module):
        def __init__(self, zenc, lv=True):
            super().__init__()
            self.z = zenc
            self.lv = lv

        def forward(self, x, u):
            x_ = torch.log1p(x) if self.lv else x
            out = self.z(x_)
            first = out[0] if isinstance(out, (tuple, list)) else out
            loc = first.loc if hasattr(first, "loc") else first
            return torch.stack([(loc * u).sum(1)] * 2, 1)

    scvi.settings.seed = 0
    sym = np.asarray(genes_go)
    keep = hvg_union(Craw, sym, gt_genes)
    genes_v = [str(g) for g in sym[keep]]
    a = ad.AnnData(X=Craw[:, keep].astype("float32").copy())
    a.var_names = genes_v
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.layers["counts"] = a.X.copy()
    scvi.model.SCVI.setup_anndata(a, layer="counts")
    m = scvi.model.SCVI(a, n_latent=10)
    # a trailing 1-cell minibatch breaks batchnorm ("Expected more than 1 value per channel",
    # hit on tf_cd4_ren whose train split is ==1 mod 128) — nudge the batch size off the residue
    bs = 128
    while a.n_obs % bs == 1 or int(a.n_obs * 0.9) % bs == 1:
        bs += 1
    m.train(max_epochs=epochs, early_stopping=True, accelerator="gpu", devices=1,
            enable_progress_bar=False, batch_size=bs)
    Z = m.get_latent_representation().astype("float32")
    zenc = m.module.z_encoder.eval()
    dev = next(zenc.parameters()).device
    ig = IntegratedGradients(SCVICentroid(zenc, bool(getattr(m.module, "log_variational", True))),
                             multiply_by_inputs=True)
    Cd = np.asarray(Craw[:, keep].todense(), dtype="float64")
    tot = Cd.sum(0)
    per = {}
    for st, tgt in states_iter(lab, keepstates):
        u = unit(Z[tgt].mean(0) - Z[~tgt].mean(0))
        cs = Cd[tgt].sum(0)
        Cc = (cs / max(int(tgt.sum()), 1)).astype("float32")
        Cr = ((tot - cs) / max(int((~tgt).sum()), 1)).astype("float32")
        per[st] = ig_top(ig, dev, Cc, Cr, u, genes_v)
    write_panel(panels_dir, fsid, "scVI", per)
    del a, m, Cd, Z
    gc.collect()
    torch.cuda.empty_cache()
    return per


def select_all(lineages, panels_dir, epochs):
    from SIGnature.models.ssl import SSLWrapper
    GT = json.load(open(f"{R}/cellmarker_gt_fine.json"))
    GO = set(l.strip() for l in open(f"{R}/go_genes.txt") if l.strip())
    ssl = [(name, SSLWrapper(model_path=path, model_filename="classifier", n_layers=5, use_gpu=True))
           for name, path in SSL_PATHS]
    for fsid in lineages:
        todo = [e for e in ENCS if not os.path.exists(panel_path(panels_dir, fsid, e))]
        if not todo:
            print(f"[{fsid}] all panels present, skip", flush=True)
            continue
        Craw, lab, keepstates, go = FS.load_lineage(fsid)
        genes_go = list(map(str, go))
        off = pd.read_csv(f"{FSB}/panels/topN_{fsid}_recast_M0.csv")
        official = sorted(off.cluster.astype(str).unique())
        assert set(official) == set(map(str, keepstates)), \
            f"{fsid}: cohort drift — official clusters {official} vs loader {keepstates}"
        print(f"\n### {fsid}: {Craw.shape[0]} cells, {len(keepstates)} states, todo={todo} ###",
              flush=True)
        for encname in todo:
            try:
                if encname == "SCim_inline":
                    per = run_scim_inline(fsid, Craw, lab, keepstates, genes_go, panels_dir)
                elif encname in ("scTab_SSL", "PBMC_SSL"):
                    w = dict(ssl)[encname]
                    per = run_ssl(fsid, encname, w, Craw, lab, keepstates, genes_go, panels_dir)
                else:
                    gt_genes = set().union(*[set(GT.get(s, [])) for s in keepstates]) & GO
                    per = run_scvi(fsid, Craw, lab, keepstates, genes_go, gt_genes, panels_dir,
                                   epochs)
                first = keepstates[0]
                print(f"  [{encname}] {len(per)} states | {first} top8: "
                      f"{', '.join(per.get(first, [])[:8])}", flush=True)
            except Exception as e:  # noqa: BLE001 — continue_on_error, scored panels decide
                print(f"  [{encname}] FAILED on {fsid}: {type(e).__name__}: {e}", flush=True)
        del Craw, lab
        gc.collect()


def score(panels_dir):
    v3 = pd.read_csv(f"{R}/fs_uniq_perstate_v3.csv")
    GT = json.load(open(f"{R}/cellmarker_gt_fine.json"))
    GO = set(l.strip() for l in open(f"{R}/go_genes.txt") if l.strip())
    rows = []
    for fsid in COHORT:
        off = pd.read_csv(f"{FSB}/panels/topN_{fsid}_recast_M0.csv")
        clusters = sorted(off.cluster.astype(str).unique())
        markers = {c: set(GT.get(c, [])) & GO for c in clusters}
        shared = {g for g, n in Counter(g for c in clusters for g in markers[c]).items() if n >= 2}
        markers = {c: markers[c] - shared for c in clusters}
        for encname in ENCS:
            fp = panel_path(panels_dir, fsid, encname)
            if not os.path.exists(fp):
                continue
            pnl = pd.read_csv(fp).sort_values(["cluster", "rank"])
            gs = {str(c): list(g.gene.astype(str)) for c, g in pnl.groupby("cluster")}
            for c in clusters:
                if not markers[c]:
                    continue
                top = gs.get(c, [])[:K_SCORE]
                rows.append(dict(lineage=fsid, state=c, method=f"recast_M0@{encname}",
                                 recall=len(set(top) & markers[c]) / len(markers[c])))
    dv = pd.DataFrame(rows)
    dv["unit"] = dv.lineage + "|" + dv.state
    tier_map = v3.drop_duplicates("unit").set_index("unit").tier
    dv["tier"] = dv.unit.map(tier_map)
    dv.to_csv(f"{R}/encoder_swap_perstate.csv", index=False)

    pivF = v3.pivot_table(index="unit", columns="method", values="recall")
    units = pivF.index
    U = pivF.values
    peF = np.nanmean(U, 0)
    field = list(pivF.columns)
    rng = np.random.default_rng(0)
    B = 2000
    out = []
    print(f"\n=== leaderboard: {len(units)} units, {len(field)} methods "
          f"(incl recast_M0={pivF['recast_M0'].mean():.4f} #"
          f"{int((peF > pivF['recast_M0'].mean()).sum()) + 1}) ===")
    for encname in ENCS:
        mcol = f"recast_M0@{encname}"
        sub = dv[dv.method == mcol].set_index("unit").recall.reindex(units)
        n_units = int(sub.notna().sum())
        if n_units == 0:
            print(f"  {mcol}: NO PANELS")
            continue
        sv = sub.values.astype(float)
        pe = float(np.nanmean(sv))
        rank = int((peF > pe).sum()) + 1
        ranks = np.empty(B, int)
        for b in range(B):
            idx = rng.integers(0, len(units), len(units))
            mu = np.nanmean(U[idx], 0)
            ranks[b] = int((mu > np.nanmean(sv[idx])).sum()) + 1
        tiers = dv[dv.method == mcol].groupby("tier").recall.mean()
        out.append(dict(encoder=encname, method=mcol, mean_recall20=round(pe, 4),
                        rank_in_field=rank, field_size=len(field) + 1, n_units=n_units,
                        boot_rank_median=int(np.median(ranks)),
                        boot_rank_lo=int(np.percentile(ranks, 2.5)),
                        boot_rank_hi=int(np.percentile(ranks, 97.5)),
                        fine_immune=round(float(tiers.get("fine_immune", np.nan)), 4),
                        fine_epithelial=round(float(tiers.get("fine_epithelial", np.nan)), 4),
                        distinct_control=round(float(tiers.get("distinct_control", np.nan)), 4)))
        print(f"  {mcol}: recall@20={pe:.4f} rank #{rank}/{len(field) + 1} "
              f"boot[#{out[-1]['boot_rank_lo']},#{out[-1]['boot_rank_hi']}] n={n_units}")
    lb = pd.DataFrame(out)
    lb.to_csv(f"{R}/encoder_swap_leaderboard.csv", index=False)
    top = np.argsort(-peF)[:8]
    print("  field top8:", [(field[i], round(float(peF[i]), 4)) for i in top])
    return lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--panels_dir", default=f"{FSB}/panels")
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.panels_dir, exist_ok=True)
    lineages = args.lineage or COHORT
    unknown = [l for l in lineages if l not in COHORT]
    assert not unknown, f"not in the benchmark cohort: {unknown}"
    if not args.score_only:
        select_all(lineages, args.panels_dir, args.epochs)
    score(args.panels_dir)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
