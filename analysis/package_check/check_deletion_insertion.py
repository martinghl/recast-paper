"""Fig. 2e and Supplementary Fig. 1a: deletion and insertion curves of the projected encoder output.

For each of the 14 subtypes (the four immune lineages of Fig. 2c), genes of the target profile C_T are replaced
in ranked order by their reference values C_R (deletion), or genes of C_R by their target values (insertion), and
f = <enc(profile), u> is read off after every step, scaled so that f(C_R) = 0 and f(C_T) = 1. Everything RECAST
contributes comes from one package call per lineage: the ranking (result.genes), the direction u
(result.directions), the two profiles (result.target_profile, result.reference_profile) and the encoder that
evaluates f (enc.torch_encode). The two comparison orders are the expression difference C_T - C_R and a fixed
random permutation per subtype (seed 1234 + the subtype's position in the lineage's list), as in the original run.

Compared with results/faithfulness_delins_{auc,curves}_v3.csv (rounded to 4 and 5 decimals).
    python check_deletion_insertion.py
Writes deletion_insertion_auc.csv and deletion_insertion_curves.csv."""
import os

import numpy as np
import pandas as pd
import torch

import common as C
import recast
from b_fig2_cohort14 import CFG

RES = os.path.join(C.FSB, "results")
ZOOM = 512


def f_of(enc, profiles, u):
    with torch.no_grad():
        z = enc.torch_encode(torch.as_tensor(np.ascontiguousarray(profiles, dtype="float32")))
    return z.detach().cpu().numpy() @ u


def curves(enc, Ct, Cr, u, order, ks):
    Del = np.tile(Ct, (len(ks), 1)); Ins = np.tile(Cr, (len(ks), 1))
    for i, k in enumerate(ks):
        if k:
            idx = order[:k]
            Del[i, idx] = Cr[idx]
            Ins[i, idx] = Ct[idx]
    return f_of(enc, Del, u), f_of(enc, Ins, u)


def auc_norm(fvals, f_r, f_t, xs):
    n = (fvals - f_r) / (f_t - f_r + 1e-12)
    return float(np.sum((n[1:] + n[:-1]) / 2.0 * np.diff(xs))), n


def main():
    enc = C.scimilarity()
    aucs, crv = [], []
    for lin, (stem, label_file, states) in CFG.items():
        labels = C.read_labels(os.path.join(C.ATLASES, label_file))
        a, kept = C.lineage_adata(C.ATLASES, stem, labels, states, min_cells=60)
        res = recast.cluster_attribution(enc, a, "state", reference="rest", gate="dC",
                                         centroid="mean_lognorm", qc="silent")
        pos = {g: j for j, g in enumerate(a.var_names)}
        G = a.n_vars
        ks = sorted(set([0] + [int(round(x)) for x in np.geomspace(1, G, 44)] + [G]))
        fr = np.array(ks) / G
        zmask = np.array(ks) <= ZOOM
        frz = np.array(ks)[zmask] / ZOOM
        for si, s in enumerate([s for s in states if s in kept]):        # the lineage's own order, as in the original run
            u = res.directions.loc[s].to_numpy("float32")
            Ct = res.target_profile[s].to_numpy("float32")
            Cr = res.reference_profile[s].to_numpy("float32")
            f_t = float(f_of(enc, Ct[None], u)[0]); f_r = float(f_of(enc, Cr[None], u)[0])
            orders = {"RECAST": np.array([pos[g] for g in res.genes[s]]),
                      "DE": np.argsort(-(Ct - Cr)),
                      "random": np.random.default_rng(1234 + si).permutation(G)}
            for rk, order in orders.items():
                fdel, fins = curves(enc, Ct, Cr, u, order, ks)
                d, nd = auc_norm(fdel, f_r, f_t, fr)
                i, ni = auc_norm(fins, f_r, f_t, fr)
                dz, _ = auc_norm(fdel[zmask], f_r, f_t, frz)
                iz, _ = auc_norm(fins[zmask], f_r, f_t, frz)
                aucs.append(dict(lineage=lin, state=s, ranking=rk, f_c=f_t, f_o=f_r, del_auc=d, ins_auc=i,
                                 gap=i - d, del_auc_top512=dz, ins_auc_top512=iz))
                crv += [dict(lineage=lin, state=s, ranking=rk, k=k, del_f=float(x), ins_f=float(y))
                        for k, x, y in zip(ks, nd, ni)]
        print(f"[{lin}] {len(kept)} subtypes", flush=True)
        C.assert_package_only()

    A = pd.DataFrame(aucs)
    stored = pd.read_csv(os.path.join(RES, "faithfulness_delins_auc_v3.csv"))
    M = A.merge(stored, on=["lineage", "state", "ranking"], suffixes=("", "_stored"), validate="one_to_one")
    assert len(M) == len(A) == len(stored), (len(M), len(A), len(stored))
    cols = ["f_c", "f_o", "del_auc", "ins_auc", "gap", "del_auc_top512", "ins_auc_top512"]
    for c in cols:
        M[f"d_{c}"] = (M[c] - M[f"{c}_stored"]).abs()
    print("max |package - original| over the 42 rows (original values are rounded to 4 decimals):")
    print(M[[f"d_{c}" for c in cols]].max().round(6).to_string())
    print("means by ranking (package):")
    print(A.groupby("ranking")[["del_auc", "ins_auc", "gap"]].mean().round(4).to_string())
    print("means by ranking (original):")
    print(stored.groupby("ranking")[["del_auc", "ins_auc", "gap"]].mean().round(4).to_string())

    K = pd.DataFrame(crv)
    pc = pd.read_csv(os.path.join(RES, "faithfulness_delins_curves_v3.csv"))
    MC = K.merge(pc, on=["lineage", "state", "ranking", "k"], suffixes=("", "_stored"), validate="one_to_one")
    assert len(MC) == len(K) == len(pc), (len(MC), len(K), len(pc))
    for c in ("del_f", "ins_f"):
        MC[f"d_{c}"] = (MC[c] - MC[f"{c}_stored"]).abs()
    print("curves: max |package - original| (original rounded to 5 decimals):",
          MC[["d_del_f", "d_ins_f"]].max().round(6).to_dict())
    C.write(M, "deletion_insertion_auc.csv")
    C.write(MC, "deletion_insertion_curves.csv")


if __name__ == "__main__":
    main()
