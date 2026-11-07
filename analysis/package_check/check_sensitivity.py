"""Supplementary Fig. 5a-e: sensitivity of the 20-gene RECAST selection, recomputed with the package.

Sixteen Zhao subtypes, each against the other subtypes of its lineage, as in
validation/sensitivity/sensitivity_sweeps.py (a-d) and baseline_choice.py (e):
  a  n_steps in 8 / 16 / 32 / 50 / 100 / 200                    recast.attribute(n_steps=...)
  b  target cells subsampled to 20 / 50 / 100 / 200 (3 draws)    the same call on the subsampled cells
  c  reference cells subsampled to 50 / 100 / 200 / 500 (3 draws)
  d  ten joint draws of 100 target and 200 reference cells
  e  the IG endpoints replaced, with the contrast direction unchanged:
       ref_cpm         centroid="pseudobulk"
       median_lognorm  centroid="median_lognorm"
       global_cpm      baseline=<whole-dataset pseudobulk profile>, gate="path"
       zero            baseline="zero", gate="path"
The draws use the original generators (seeds 1000+s, 2000+s, 3000+s). Recall@20 against the curated panels and
the top-20 overlap with the default selection are compared with the original tables.

    python check_sensitivity.py
Writes sensitivity_<igsteps|refsize|cellnum|seeds|baseline>.csv."""
import os

import numpy as np
import pandas as pd

import common as C
import check_reference_choice as RC
import recast
from recast.centroid import pseudobulk_centroid

RIG = os.path.join(C.SB, "validation", "sensitivity")
K = 20
IGSTEPS = [8, 16, 32, 50, 100, 200]
REFSIZE = [50, 100, 200, 500]
CELLNUM = [20, 50, 100, 200]


def jac(a, b):
    a, b = set(a[:K]), set(b[:K])
    return len(a & b) / len(a | b)


def main():
    enc = C.scimilarity()
    a, _, obs = RC.dataset("zhao")
    lab = obs["label"].astype(str).to_numpy()
    genes = set(a.var_names)
    Z = enc.embed(a.X)
    glob = pseudobulk_centroid(a.X)                       # whole-dataset pseudobulk (the global_cpm baseline)

    def top(tidx, ridx, **kw):
        idx = np.concatenate([tidx, ridx])
        b = a[idx].copy()
        b.obs["grp"] = pd.Categorical(["T"] * len(tidx) + ["R"] * len(ridx))
        opts = dict(gate="dC", centroid="mean_lognorm", qc="off", n_steps=50)
        opts.update(kw)
        res = recast.attribute(enc, b, "grp", target="T", reference=["R"], embeddings=Z[idx], **opts)
        return res.genes["T"][:K]

    rows = {k: [] for k in ("igsteps", "refsize", "cellnum", "seeds", "baseline")}
    for tag, (sibs, _, _) in RC.LIN.items():
        for t in sibs:
            tgt = np.where(lab == t)[0]
            ref = np.where(np.isin(lab, [s for s in sibs if s != t]))[0]
            if len(tgt) < K or len(ref) < 5:
                continue
            cur = [g for g in RC.CURATED[tag][t] if g in genes]
            rec = lambda tp: len(set(tp) & set(cur)) / len(cur)
            R50 = top(tgt, ref)
            for ns in IGSTEPS:
                tp = R50 if ns == 50 else top(tgt, ref, n_steps=ns)
                rows["igsteps"].append(dict(lineage=tag, target=t, n_steps=ns, recall20=rec(tp), jac_vs50=jac(tp, R50)))
            for m in REFSIZE:
                if m >= len(ref):
                    continue
                rc, jc = [], []
                for s in range(3):
                    sub = np.random.default_rng(1000 + s).choice(ref, size=m, replace=False)
                    tp = top(tgt, sub); rc.append(rec(tp)); jc.append(jac(tp, R50))
                rows["refsize"].append(dict(lineage=tag, target=t, ref_size=m, recall20=float(np.mean(rc)),
                                            jac_vs_full=float(np.mean(jc))))
            for m in CELLNUM:
                if m >= len(tgt):
                    continue
                rc, jc = [], []
                for s in range(3):
                    sub = tgt[np.random.default_rng(2000 + s).choice(len(tgt), size=m, replace=False)]
                    tp = top(sub, ref); rc.append(rec(tp)); jc.append(jac(tp, R50))
                rows["cellnum"].append(dict(lineage=tag, target=t, n_cells=m, recall20=float(np.mean(rc)),
                                            jac_vs_full=float(np.mean(jc))))
            mr, mc = min(200, len(ref)), min(100, len(tgt))
            panels, recs = [], []
            for s in range(10):
                rng = np.random.default_rng(3000 + s)
                sub_t = tgt[rng.choice(len(tgt), size=mc, replace=False)]
                sub_r = rng.choice(ref, size=mr, replace=False)
                tp = top(sub_t, sub_r); panels.append(tp); recs.append(rec(tp))
            pj = [jac(panels[i], panels[j]) for i in range(10) for j in range(i + 1, 10)]
            rows["seeds"].append(dict(lineage=tag, target=t, recall20_mean=float(np.mean(recs)),
                                      recall20_sd=float(np.std(recs)), jac_pairwise_mean=float(np.mean(pj))))
            modes = {"mean_lognorm": R50,
                     "ref_cpm": top(tgt, ref, centroid="pseudobulk"),
                     "median_lognorm": top(tgt, ref, centroid="median_lognorm"),
                     "global_cpm": top(tgt, ref, baseline=glob, gate="path"),
                     "zero": top(tgt, ref, baseline="zero", gate="path")}
            for mode, tp in modes.items():
                rows["baseline"].append(dict(mode=mode, lineage=tag, target=t, recall20=rec(tp),
                                             jac_vs_default=jac(tp, R50)))
            print(f"  {tag} / {t}", flush=True)
            C.assert_package_only()

    stored_tables = {"igsteps": ("sensitivity_igsteps_v3.csv", ["lineage", "target", "n_steps"], ["recall20", "jac_vs50"]),
            "refsize": ("sensitivity_refsize_v3.csv", ["lineage", "target", "ref_size"], ["recall20", "jac_vs_full"]),
            "cellnum": ("sensitivity_cellnum_v3.csv", ["lineage", "target", "n_cells"], ["recall20", "jac_vs_full"]),
            "seeds": ("sensitivity_seeds_v3.csv", ["lineage", "target"],
                      ["recall20_mean", "recall20_sd", "jac_pairwise_mean"]),
            "baseline": ("ig_baseline_metrics_v3.csv", ["mode", "lineage", "target"], ["recall20", "jac_vs_default"])}
    for name, (f, keys, cols) in stored_tables.items():
        stored = pd.read_csv(os.path.join(RIG, f))
        if "is_full" in stored:
            stored = stored[~stored.is_full]
        M = pd.DataFrame(rows[name]).merge(stored[keys + cols], on=keys, how="left", suffixes=("", "_stored"),
                                           validate="one_to_one")
        assert M[f"{cols[0]}_stored"].notna().all(), (name, M[M[f"{cols[0]}_stored"].isna()][keys])
        diffs = {c: float((M[c] - M[f"{c}_stored"]).abs().max()) for c in cols}
        exact = int(np.all(np.isclose(M[cols].to_numpy(float), M[[f"{c}_stored" for c in cols]].to_numpy(float),
                                      atol=1e-12), axis=1).sum())
        print(f"[{name}] {len(M)} rows, equal {exact}/{len(M)}; max |package - original| "
              + ", ".join(f"{c} {v:.3g}" for c, v in diffs.items()), flush=True)
        if name == "baseline":
            print(M.groupby("mode")[["recall20", "recall20_stored", "jac_vs_default", "jac_vs_default_stored"]]
                  .mean().round(4).to_string(), flush=True)
        C.write(M, f"sensitivity_{name}.csv")


if __name__ == "__main__":
    main()
