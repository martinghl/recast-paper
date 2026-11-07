"""Rankings behind Fig. 2c, d: trained, randomly initialised and reference-free attribution on the 97 benchmark units
(31 dataset-lineage groups).

Rankings per unit (top 20 saved; scored by code/figure02/01_marker_recovery.py on ground_truth/panels/marker_panels.json):
  trained         population RECAST through the code path of Fig. 2f (group_rankings.py, kind "full"): same cells,
                  labels and gene order (select_recast.load_lineage), centroid, direction, 50-step IG, dC > 0 gate and
                  positive channel; identical to out/orders/<lineage>.csv.gz (kind "full").
  random_0..19    the same code path with SCimilarity re-initialised: torch.manual_seed(seed), deepcopy,
                  reset_parameters on every module; the embedding, direction and IG all use the random network.
  reference_free  the per-cell integrated-gradients layer (attribution_ig of <stem>_attribution_scimilarity.h5ad),
                  averaged over the unit's target cells.
Analysis (01_marker_recovery.py): points are the 31 groups (mean over each group's units); RECAST minus comparator by a
two-sided group-level sign-flip test (1,000,000 random sign vectors) and a 10,000-resample group bootstrap percentile
95% CI; random encoders by the empirical one-sided P = (k + 1) / 21, k = seeds whose 31-group mean reaches the trained
value.  Unit-level results are reported beside.

Env SC, GPU.  Run the added and the original lineages in separate processes (select_recast module globals), as
group_rankings.py:
  CUDA_VISIBLE_DEVICES=0 python -I subtype_rankings.py LINEAGE [LINEAGE ...]
Writes out/cohort97/<lineage>.csv.gz (unit, encoder, top20); a lineage whose file exists is skipped.
"""
import copy
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import group_rankings as RM                 # noqa: E402  sys.path, M, P, S, D, UNIT_GT as in the Fig. 2f run

M, S, D, UNIT_GT = RM.M, RM.S, RM.D, RM.UNIT_GT
OUT = f"{HERE}/out/cohort97"
N_SEEDS = 20


def reference_free_layer(lin, lab):
    """attribution_ig rows of the cells load_lineage keeps (mirrors select_recast.load_lineage's selection)."""
    e = S.C.LINEAGES.get(lin)
    if e is not None:
        stem, states, min_cells = e["ds"], e["states"], e["min_cells"]
        lab_all = np.array([l.strip() for l in open(f"{S.ATLASES}/{e['label_file']}")])
    elif lin in S.EXT_ZHAO:
        stem, states, min_cells, lab_all = "zhao", S.EXT_ZHAO[lin], 50, None
    else:
        raise SystemExit(f"unknown id {lin}")
    att = ad.read_h5ad(f"{S.ATLASES}/{stem}_attribution_scimilarity.h5ad")
    if lab_all is None:
        lab_all = att.obs["label"].astype(str).to_numpy()
    keepstates = sorted([s for s in states if (lab_all == s).sum() >= min_cells])
    keep = np.isin(lab_all, keepstates)
    assert np.array_equal(lab_all[keep].astype(str), lab), f"{lin}: label order differs from load_lineage"
    van = att.layers["attribution_ig"][np.flatnonzero(keep)]
    van = (van.toarray() if sp.issparse(van) else np.asarray(van)).astype("float32")
    del att
    return van


def main(lineages):
    assert M.K == 20 and M.NSTEPS == 50
    added_lineages = [l for l in lineages if l in D.NEW_IDS]
    assert not added_lineages or len(added_lineages) == len(lineages), "run added and original lineages in separate processes"
    if added_lineages:
        import types
        S.C = types.SimpleNamespace(LINEAGES=D.LINEAGES, NEW_IDS=D.NEW_IDS)
        S.ATLASES = f"{RM.FULL_BENCHMARK}/data/atlases"
    import contrastive_ig
    from captum.attr import IntegratedGradients
    dev = contrastive_ig.dev

    def embed_with(enc, X, bs=512):
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], bs):
                out.append(enc(torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)).cpu().numpy())
        return np.vstack(out)

    def full_order(enc, Z, Craw, tgt, ref):          # group_rankings.run_groups([tgt]) with a given encoder
        ig = IntegratedGradients(contrastive_ig.CentroidTarget(enc).to(dev).eval(), multiply_by_inputs=True)
        C_R = M.fc.centroid(Craw, ref).astype("float32")
        cg = M.fc.centroid(Craw, tgt).astype("float32")[None]
        u = Z[tgt].mean(0) - Z[ref].mean(0)
        u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")[None]
        inp = torch.tensor(cg, device=dev).requires_grad_(True)
        base = torch.tensor(C_R[None], device=dev)
        a = ig.attribute(inp, baselines=base, target=0, additional_forward_args=(torch.tensor(u, device=dev),),
                         n_steps=M.NSTEPS, internal_batch_size=M.BATCH * 8).detach().cpu().numpy()
        a = np.where(cg - C_R[None] > 0, np.clip(a, 0, None), 0.0)
        ss = a.sum(1, keepdims=True); ss[ss == 0] = 1.0
        return np.argsort(-(a / ss * 1000.0).mean(0))

    os.makedirs(OUT, exist_ok=True)
    for lin in lineages:
        path = f"{OUT}/{lin}.csv.gz"
        if os.path.exists(path):
            print(f"[{lin}] exists, skipped", flush=True)
            continue
        t0 = time.time()
        Craw, lab, states, genes = S.load_lineage(lin)
        genes = list(genes)
        Xln = M.lognorm_rows(Craw)
        units = [c for c in states if f"{lin}|{c}" in UNIT_GT]
        assert set(units) == {u.split("|", 1)[1] for u in UNIT_GT if u.split("|", 1)[0] == lin}, lin
        idx = {c: (np.where(lab == c)[0], np.where(lab != c)[0]) for c in units}
        rows = []
        top = lambda order: ";".join(genes[j] for j in order[:M.K])       # noqa: E731
        # reference-free
        van = reference_free_layer(lin, lab)
        for c in units:
            rows.append(dict(unit=f"{lin}|{c}", encoder="reference_free", top20=top(np.argsort(-van[idx[c][0]].mean(0)))))
        del van
        # trained, then the 20 re-initialised encoders
        encs = [("trained", contrastive_ig.enc)]
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            rand = copy.deepcopy(contrastive_ig.enc)
            for m in rand.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            encs.append((f"random_{seed}", rand.eval()))
            if seed == 0:
                assert not torch.equal(next(rand.parameters()), next(contrastive_ig.enc.parameters()))
        for name, enc in encs:
            Z = embed_with(enc, Xln)
            for c in units:
                rows.append(dict(unit=f"{lin}|{c}", encoder=name, top20=top(full_order(enc, Z, Craw, *idx[c]))))
            if name != "trained":
                del enc
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip")
        os.replace(path + ".tmp", path)
        print(f"[{lin}] {len(units)} units, {len(rows)} rankings, {time.time() - t0:.0f}s", flush=True)
        del Craw, Xln
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
