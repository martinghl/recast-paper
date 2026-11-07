"""Cell-cycle comparisons defined by cell-cycle phase (Supplementary Fig. 4c, d):

    target    = cells the original authors call S/G2M   (bonemarrow_dev obs['CyclePhase'])
    reference = cells the authors call G0/G1 of the same AuthorCellType

so every comparison is "cycling X vs non-cycling X" with identity fixed; 33 identities have >= 50 cells in each
group.  Two more:

    bmf_global      all S/G2M vs all G0/G1 across the atlas, stratified subsample to 25k / 25k (seed 0)
    zhao_asc_exact  author clusters "cycling ASC" vs "noncycling ASC"

All groups come from the authors' labels.  Same output format as build_identity_contrasts.py:
phase_contrasts/<name>/{counts.npz, labels.txt, meta.json}.

Env: SC (SIGnature's align; no GPU).  Uses the loaders of build_identity_contrasts.py."""
import json
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

CR = "/path/to/research/analysis/cell_cycle"
OUT = f"{CR}/phase_contrasts"
BM = f"{CR}/data/bonemarrow_dev.h5ad"
MIN_SIDE = 50
GLOBAL_PER_SIDE = 25000
CYCLING = {"S", "G2M"}

import build_identity_contrasts as cc03  # noqa: E402  GO, _align, load_p2 -- module import only


def slug(name):
    return "bmf_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_contrast(name, counts, y, meta):
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    sp.save_npz(f"{d}/counts.npz", counts.tocsr())
    with open(f"{d}/labels.txt", "w") as fh:
        fh.write("\n".join(y) + "\n")
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    print(f"[{name}] {counts.shape[0]} cells ({meta['n_target']} target / "
          f"{meta['n_reference']} reference)", flush=True)
    return dict(contrast=name, status="ok", n_target=meta["n_target"],
                n_reference=meta["n_reference"], genes_measured_in_panel=meta["genes_measured_in_panel"])


def stratified_sample(rng, idx_by_ident, total):
    """Largest-remainder proportional allocation over identities, exact `total` rows."""
    n_all = sum(len(v) for v in idx_by_ident.values())
    quota = {k: total * len(v) / n_all for k, v in idx_by_ident.items()}
    base = {k: min(int(np.floor(q)), len(idx_by_ident[k])) for k, q in quota.items()}
    short = total - sum(base.values())
    for k in sorted(quota, key=lambda k: quota[k] - np.floor(quota[k]), reverse=True):
        if short <= 0:
            break
        if base[k] < len(idx_by_ident[k]):
            base[k] += 1
            short -= 1
    take = []
    for k, n in base.items():
        if n:
            take.append(rng.choice(idx_by_ident[k], size=n, replace=False))
    return np.sort(np.concatenate(take))


def main():
    import anndata as ad
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # ---------------- bonemarrow: one full aligned matrix, then row slices ----------------
    A = ad.read_h5ad(BM, backed="r")
    ct = A.obs["AuthorCellType"].astype(str).to_numpy()
    ph = A.obs["CyclePhase"].astype(str).to_numpy()
    R = A.raw if A.raw is not None else A
    print(f"materialising raw counts {R.shape} ...", flush=True)
    X = sp.csr_matrix(R.X[:])
    rv = R.var
    syms = (rv["feature_name"].astype(str).to_numpy() if "feature_name" in rv.columns
            else np.asarray(R.var_names, dtype=str))
    A.file.close()
    raw = ad.AnnData(X=X, var=pd.DataFrame(index=pd.Index(syms, name=None)))
    del X
    counts_all, n_meas = cc03._align(raw, "bonemarrow_full")   # integer guard inside
    del raw
    cyc = np.isin(ph, list(CYCLING))

    idents = sorted({c for c in np.unique(ct)
                     if (cyc & (ct == c)).sum() >= MIN_SIDE
                     and (~cyc & (ct == c)).sum() >= MIN_SIDE})
    slugs = [slug(c) for c in idents]
    assert len(set(slugs)) == len(slugs), "identity slug collision"
    print(f"{len(idents)} identities pass >= {MIN_SIDE}/{MIN_SIDE}", flush=True)

    for ident, sname in zip(idents, slugs):
        rows_i = np.flatnonzero(ct == ident)
        y = np.where(cyc[rows_i], "target", "reference")
        meta = dict(contrast=sname, arm="fair", dataset="bonemarrow_dev",
                    identity=ident, grouping="author CyclePhase (S/G2M vs G0/G1) within AuthorCellType",
                    n_target=int((y == "target").sum()), n_reference=int((y == "reference").sum()),
                    genes_measured_in_panel=int(n_meas))
        rows.append(write_contrast(sname, counts_all[rows_i], y, meta))

    # ---------------- bmf_global: pan-atlas cycling vs noncycling, stratified 25k/25k ----------------
    rng = np.random.default_rng(0)
    tgt_pool = {c: np.flatnonzero(cyc & (ct == c)) for c in np.unique(ct) if (cyc & (ct == c)).sum()}
    ref_pool = {c: np.flatnonzero(~cyc & (ct == c)) for c in np.unique(ct) if (~cyc & (ct == c)).sum()}
    n_t = min(GLOBAL_PER_SIDE, sum(len(v) for v in tgt_pool.values()))
    n_r = min(GLOBAL_PER_SIDE, sum(len(v) for v in ref_pool.values()))
    it = stratified_sample(rng, tgt_pool, n_t)
    ir = stratified_sample(rng, ref_pool, n_r)
    rows_g = np.concatenate([it, ir])
    y = np.array(["target"] * len(it) + ["reference"] * len(ir))
    meta = dict(contrast="bmf_global", arm="fair", dataset="bonemarrow_dev", identity="(all)",
                grouping="author CyclePhase, all identities pooled; stratified subsample by "
                         f"AuthorCellType, {GLOBAL_PER_SIDE}/side, seed 0",
                n_target=len(it), n_reference=len(ir), genes_measured_in_panel=int(n_meas))
    rows.append(write_contrast("bmf_global", counts_all[rows_g], y, meta))
    del counts_all

    # ---------------- zhao ASC exact pair (author cluster labels; fullgene counts fallback) ----------------
    ATLASES = "/path/to/research/analysis/atlases"
    counts, y, cov = cc03.load_p2(f"{ATLASES}/zhao_attribution_scimilarity.h5ad", "label",
                                  ["cycling ASC"], ["noncycling ASC"])
    meta = dict(contrast="zhao_asc_exact", arm="fair", dataset="zhao", identity="ASC",
                grouping="author clusters: cycling ASC (target) vs noncycling ASC (reference)",
                n_target=int((y == "target").sum()), n_reference=int((y == "reference").sum()),
                genes_measured_in_panel=int(cov))
    rows.append(write_contrast("zhao_asc_exact", counts, y, meta))

    df = pd.DataFrame(rows)
    df.to_csv(f"{CR}/phase_contrasts_built.csv", index=False)
    print("\n=== FAIR CONTRASTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
