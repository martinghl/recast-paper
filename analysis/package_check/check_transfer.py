"""Supplementary Fig. 9: RECAST panels behind the cross-atlas transfer comparisons, recomputed with the package.

Fine subtypes (42 directed comparisons): transfer_pairs.py read the RECAST lists from the benchmark panels
topN_tf_<compartment>_<atlas>_recast_M0.csv and used their first 3, 5, 10 and 20 genes. check_benchmark_panels.py
recomputes those 15 lineages with the package, so this script collects that comparison.

Broad populations (nine shared by Ren, Stephenson and Dominguez-Conde): transfer_summary.py selected the
top 20 genes of every shared population with at least 20 cells, against all other cells of the atlas, and stored
only the transfer AUROC of each atlas pair (the selected genes, in atlas B). Here the lists are one recast.attribute
call per atlas, and the AUROCs are recomputed with the original evaluation, replaying its random draws in the
original order (the 8,000-cell caps drawn while selecting, then the evaluations, RECAST first).

    python check_transfer.py
Writes transfer_subtypes.csv and transfer_broad.csv."""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import roc_auc_score

import common as C
import recast

# transfer.py
DATASETS = ["ren", "stephenson", "dominguez_conde"]
SHARED = ["CD16+ NK cells", "Classical monocytes", "Non-classical monocytes", "Tcm/Naive cytotoxic T cells",
          "Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Tem/Temra cytotoxic T cells",
          "Tem/Trm cytotoxic T cells", "gdT"]
K, CAP = 20, 8000
RES = os.path.join(C.FSB, "results")


def fine():
    A = pd.read_csv(os.path.join(C.OUT, "benchmark_panels.csv"))
    T = A[A.lineage.str.startswith("tf_")].copy()
    T["top20_order_identical"] = T.first_diff_rank.isna() | (T.first_diff_rank > K)
    print(f"[fine] {T.lineage.nunique()} lineages, {len(T)} subtypes; first 20 genes identical in order "
          f"{int(T.top20_order_identical.sum())}/{len(T)}", flush=True)
    return T


def broad():
    rng = np.random.default_rng(0)
    enc = C.scimilarity(); genes = C.scimilarity_genes()
    GL, XX, LL, GI = {}, {}, {}, {}
    for d in DATASETS:
        A = ad.read_h5ad(os.path.join(C.ATLASES, f"{d}_attribution_scimilarity.h5ad"))
        X = (A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)).astype("float32")
        lab = A.obs["label"].astype(str).to_numpy(); names = np.asarray(A.obs_names).astype(str)
        assert list(A.var_names.astype(str)) == genes
        del A
        for c in [t for t in np.unique(lab) if (lab == t).sum() >= 10]:   # caps drawn by genelists()
            if (lab == c).sum() > CAP:
                rng.choice(np.where(lab == c)[0], CAP, replace=False)
        a = ad.AnnData(C.counts(C.ATLASES, d, names, genes))
        a.var_names = genes
        a.obs["label"] = lab
        types = [t for t in SHARED if (lab == t).sum() >= 20]
        res = recast.attribute(enc, a, "label", target=types, reference="rest", gate="dC", centroid="mean_lognorm",
                               qc="off", embeddings=enc.embed(X, batch_size=512))
        GL[d] = {t: res.genes[t][:K] for t in types}
        XX[d], LL[d], GI[d] = X, lab, {g: i for i, g in enumerate(genes)}
        print(f"[broad] {d}: {len(types)} shared populations", flush=True)
        C.assert_package_only()

    def eval_in_B(markers, T, XB, labB, giB):      # transfer_summary.eval_in_B
        idx = [giB[g] for g in markers if g in giB]
        if not idx:
            return np.nan
        y = (labB == T).astype(int)
        if y.sum() < 5 or y.sum() == len(y):
            return np.nan
        sig = XB[:, idx].mean(1)
        if len(y) > CAP:
            pos = np.where(y == 1)[0]
            neg = rng.choice(np.where(y == 0)[0], min((y == 0).sum(), CAP), replace=False)
            sel = np.concatenate([pos, neg]); y, sig = y[sel], sig[sel]
        return float(roc_auc_score(y, sig))

    rows = []
    for T in SHARED:
        self_auc = {B: eval_in_B(GL[B].get(T, []), T, XX[B], LL[B], GI[B]) for B in DATASETS if T in GL[B]}
        for A_ in DATASETS:
            mk = GL[A_].get(T)
            if not mk:
                continue
            for B in DATASETS:
                if B == A_:
                    continue
                auc = eval_in_B(mk, T, XX[B], LL[B], GI[B])
                ret = auc / self_auc[B] if self_auc.get(B) and self_auc[B] > 0 else np.nan
                rows.append(dict(cell_type=T, src=A_, tgt=B, transfer_auc=auc, retention=ret))
    stored = pd.read_csv(os.path.join(RES, "transferability_currency_v3.csv"))
    stored = stored[stored.method == "M0"].drop(columns="method")
    M = pd.DataFrame(rows).merge(stored, on=["cell_type", "src", "tgt"], how="outer", suffixes=("", "_stored"),
                                 validate="one_to_one", indicator=True)
    assert (M._merge == "both").all(), M[M._merge != "both"]
    for c in ("transfer_auc", "retention"):
        d = (M[c] - M[c + "_stored"]).abs()
        print(f"[broad] {c}: equal {int((d <= 1e-12).sum())}/{len(M)}, max |d| {d.max():.3g}; "
              f"mean {M[c].mean():.4f} (original {M[c + '_stored'].mean():.4f})", flush=True)
    return M.drop(columns="_merge")


def main():
    C.write(fine(), "transfer_subtypes.csv")
    C.write(broad(), "transfer_broad.csv")


if __name__ == "__main__":
    main()
