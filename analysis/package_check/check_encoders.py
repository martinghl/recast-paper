"""Supplementary Fig. 2e: subtype separability and marker recovery with five encoders, recomputed with the package.

subtype_benchmark/encoders_recast.py (SCimilarity, scTab SSL, PBMC SSL), encoders_scfoundation.py and
encoders_scvi.py took the six lineages of encoder_lineages.py. For every subtype with at least 60 cells
they ranked genes by RECAST (the subtype against the rest of the lineage) and by the difference in mean log-normalized
expression, and scored recall of the curated markers among the top 10 and 20 genes. Separability is the accuracy and
macro-F1 of a 15-nearest-neighbour classifier (five folds) on the cell embeddings.

Here RECAST is recast.attribute with the package encoders (SCimilarityEncoder, SSLEncoder, SCFoundationEncoder, and
SCVIEncoder on an scVI model trained with the original recipe); the expression ranking and the classifier are the
original ones. Compared with recast_cross_fm_recall_multi_v3.csv, recast_on_scfoundation_multi_v3.csv,
recast_on_scvi_multi_v2.csv (k = 10 and 20) and fm_finestate_knn_ALL_multi_v3.csv.

The original scVI run kept no trained model, and retraining scVI on the GPU does not give the same weights, so the
scVI numbers can differ from the original ones although the computation is the same (the classifier, which does not
involve RECAST, differs too). The scVI stage therefore also runs the original computation (encoders_scvi.py)
on the same retrained model and compares its top-20 lists, recalls, latents and classifier with the package's.

    python check_encoders.py --stage scim_ssl | scf | scvi
Writes encoders_<stage>_recall.csv and encoders_<stage>_knn.csv (and encoders_scvi_same_model.csv)."""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

import common as C
import check_profile_centroid as I2
import recast
from recast.centroid import lognorm_rows

KS = (10, 20)
RES = os.path.join(C.FSB, "results")
STORED_RECALL = {"scim_ssl": "recast_cross_fm_recall_multi_v3.csv", "scf": "recast_on_scfoundation_multi_v3.csv",
              "scvi": "recast_on_scvi_multi_v2.csv"}
CC = C.config("encoder_lineages")


def lineage_data(tag):
    """Counts of every cell of the lineage's curated subtypes (SCimilarity gene order), and the subtypes with >= 60."""
    c = CC.CFG[tag]
    states = list(c["curated"])
    _, obs = C.attribution_obs(C.ATLASES, c["stem"])
    lab_all = C.read_labels(os.path.join(C.ATLASES, c["lab"])) if c["lab"] else obs["label"].astype(str).to_numpy()
    a, _ = C.lineage_adata(C.ATLASES, c["stem"], lab_all, states, min_cells=1)
    cl = [s for s in states if (a.obs["state"] == s).sum() >= 60]
    return a, cl, c["curated"]


def recall(top, mk, gset, K):
    mk = [g for g in mk if g in gset]
    return len(set(top[:K]) & set(mk)) / len(mk) if mk else np.nan


def recalls(enc, a, cl, curated, Z):
    """RECAST lists from the package, and the mean recalls of mean_recalls."""
    res = recast.attribute(enc, a, "state", target=cl, reference="rest", gate="dC", centroid="mean_lognorm",
                           qc="off", embeddings=Z)
    return res, mean_recalls(res.genes, a, cl, curated)


def mean_recalls(lists, a, cl, curated):
    """Mean recall over subtypes at K = 10, 20: of the given lists and of the mean-expression difference (original)."""
    genes = np.asarray(a.var_names).astype(str); gset = set(genes)
    Xln = lognorm_rows(sp.csr_matrix(a.X))
    lab = a.obs["state"].to_numpy()
    expr = {}
    for c in cl:
        t = lab == c
        md = np.asarray(Xln[t].mean(0)).ravel() - np.asarray(Xln[~t].mean(0)).ravel()
        expr[c] = [genes[j] for j in np.argsort(-md)[:max(KS)]]
    return [(K, round(float(np.nanmean([recall(lists[c], curated[c], gset, K) for c in cl])), 3),
             round(float(np.nanmean([recall(expr[c], curated[c], gset, K) for c in cl])), 3)) for K in KS]


def knn(Z, lab, cl, cosine=True):
    m = np.isin(lab, cl)
    Zc = normalize(np.asarray(Z[m], dtype="float32")) if cosine else Z[m]
    clf = KNeighborsClassifier(15, metric="cosine") if cosine else KNeighborsClassifier(15)
    yhat = cross_val_predict(clf, Zc, lab[m], cv=5)
    return round(float(accuracy_score(lab[m], yhat)), 3), round(float(f1_score(lab[m], yhat, average="macro")), 3)


def run(name, enc, a, cl, curated, tag, rec, kn, cosine=True):
    Z = enc.embed(a.X)
    res, rs = recalls(enc, a, cl, curated, Z)
    for K, rf, re in rs:
        rec.append(dict(encoder=name, lineage=tag, k=K, RECAST=rf, expression=re, n_states=len(cl)))
    acc, f1 = knn(Z, a.obs["state"].to_numpy(), cl, cosine)
    kn.append(dict(FM=name, lineage=tag, state_knn_acc=acc, state_macroF1=f1))
    print(f"  [{name}] {tag}: recall@20 {rec[-1]['RECAST']} (expression {rec[-1]['expression']}), kNN macro-F1 {f1}",
          flush=True)
    return res, Z


def stage_scim_ssl(rec, kn):
    data = {tag: lineage_data(tag) for tag in CC.ORDER}
    enc = C.scimilarity()
    for tag, (a, cl, cur) in data.items():
        run("SCimilarity", enc, a, cl, cur, tag, rec, kn)
    for name, path in I2.SSL:
        enc = recast.SSLEncoder(path, device="cuda")
        genes = pd.read_parquet(os.path.join(path, "var.parquet"))["feature_name"].astype(str).tolist()
        for tag, (a, cl, cur) in data.items():
            run(name, enc, I2.to_genes(a, genes), cl, cur, tag, rec, kn)
        del enc; torch.cuda.empty_cache()


def stage_scf(rec, kn):
    enc = recast.SCFoundationEncoder(I2.SCF, device="cuda")
    for tag in CC.ORDER:
        a, cl, cur = lineage_data(tag)
        run("scFoundation", enc, I2.to_genes(a, enc.gene_order), cl, cur, tag, rec, kn)


def original_scvi(m, a, cl):
    """encoders_scvi.py on a given model: posterior-mean latents (get_latent_representation), u the normalized
    difference of the latent means, IG (50 steps) from the mean raw-count profile of the other cells to that of the
    subtype through <z_encoder(log1p(x)).loc, u>, negative attributions set to 0, genes with C_T <= C_R excluded."""
    from captum.attr import IntegratedGradients
    Zp = m.get_latent_representation().astype("float32")
    zenc = m.module.z_encoder.eval()
    dev = next(zenc.parameters()).device
    lv = bool(getattr(m.module, "log_variational", True))

    class Centroid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z = zenc

        def forward(self, x, u):
            out = self.z(torch.log1p(x) if lv else x)
            first = out[0] if isinstance(out, (tuple, list)) else out
            loc = first.loc if hasattr(first, "loc") else first
            return torch.stack([(loc * u).sum(1)] * 2, 1)

    ig = IntegratedGradients(Centroid(), multiply_by_inputs=True)
    genes = np.asarray(a.var_names).astype(str)
    lab = a.obs["state"].to_numpy()
    Cd = np.asarray(sp.csr_matrix(a.X).todense(), dtype="float64")
    tot = Cd.sum(0)
    lists = {}
    for c in cl:
        t = lab == c
        u = Zp[t].mean(0) - Zp[~t].mean(0)
        u = (u / (np.linalg.norm(u) + 1e-12)).astype("float32")
        cs = Cd[t].sum(0)
        Cc = (cs / t.sum()).astype("float32")
        Cr = ((tot - cs) / (~t).sum()).astype("float32")
        at = ig.attribute(torch.tensor(Cc[None], device=dev).requires_grad_(True),
                          baselines=torch.tensor(Cr[None], device=dev), target=0,
                          additional_forward_args=(torch.tensor(u[None], device=dev),), n_steps=50)
        at = at.detach().cpu().numpy().ravel()
        at = np.where((Cc - Cr) > 0, np.clip(at, 0, None), -np.inf)
        lists[c] = [genes[j] for j in np.argsort(-at)[:max(KS)]]
    return Zp, lists


def same_model(m, a, cl, curated, tag, res, Z, rec, kn):
    """The package run and the original computation on one model: top-20 lists, recalls, latents, classifier."""
    Zp, lists = original_scvi(m, a, cl)
    rows = [dict(lineage=tag, state=c, **C.compare_lists(res.genes[c][:max(KS)], lists[c])) for c in cl]
    stored_rec = {K: rf for K, rf, _ in mean_recalls(lists, a, cl, curated)}
    pkg_rec = {r["k"]: r["RECAST"] for r in rec if r["lineage"] == tag}
    acc, f1 = knn(Zp, a.obs["state"].to_numpy(), cl, cosine=False)
    zd = float(np.abs(Zp - Z).max())
    print(f"  [scVI same model] {tag}: top-20 lists identical {sum(r['identical'] for r in rows)}/{len(rows)}; "
          f"recall@10/20 package {pkg_rec[10]}/{pkg_rec[20]}, original code {stored_rec[10]}/{stored_rec[20]}; "
          f"kNN acc/F1 package {kn[-1]['state_knn_acc']}/{kn[-1]['state_macroF1']}, original code {acc}/{f1}; "
          f"max |latent difference| {zd:.2g}", flush=True)
    for r in rows:
        r.update(recall10_package=pkg_rec[10], recall10_original_code=stored_rec[10], recall20_package=pkg_rec[20],
                 recall20_original_code=stored_rec[20], knn_acc_original_code=acc, knn_f1_original_code=f1,
                 max_abs_latent_diff=zd)
    return rows


def stage_scvi(rec, kn, epochs=200):
    """encoders_scvi.py: seed 0 set once, lineages trained in order, HVG union with all curated markers of the
    lineage, default batch size, posterior-mean latent; Euclidean kNN on the latent."""
    import scvi
    scvi.settings.seed = 0
    same = []
    for tag in CC.ORDER:
        a0, cl, cur = lineage_data(tag)
        sym = np.asarray(a0.var_names)
        markers = set().union(*[set(v) for v in cur.values()])
        cols = I2.hvg_union(sp.csr_matrix(a0.X), sym, markers)
        a = ad.AnnData(X=sp.csr_matrix(a0.X)[:, cols].astype("float32").copy())
        a.var_names = [str(g) for g in sym[cols]]
        a.obs_names = [f"c{i}" for i in range(a.n_obs)]
        a.layers["counts"] = a.X.copy()
        scvi.model.SCVI.setup_anndata(a, layer="counts")
        m = scvi.model.SCVI(a, n_latent=10)
        m.train(max_epochs=epochs, early_stopping=True, accelerator="gpu", devices=1, enable_progress_bar=False)
        a.obs["state"] = a0.obs["state"].to_numpy()
        res, Z = run("scVI", recast.SCVIEncoder(m), a, cl, cur, tag, rec, kn, cosine=False)
        same += same_model(m, a, cl, cur, tag, res, Z, rec, kn)
        del m; torch.cuda.empty_cache()
    S = pd.DataFrame(same)
    print(f"[scvi same model] top-20 lists identical {int(S.identical.sum())}/{len(S)}, same top-20 set "
          f"{int(S.same_set_top20.sum())}/{len(S)}")
    C.write(S, "encoders_scvi_same_model.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["scim_ssl", "scf", "scvi"])
    s = ap.parse_args().stage
    rec, kn = [], []
    {"scim_ssl": stage_scim_ssl, "scf": stage_scf, "scvi": stage_scvi}[s](rec, kn)
    C.assert_package_only()
    R = pd.DataFrame(rec).merge(pd.read_csv(os.path.join(RES, STORED_RECALL[s])), on=["encoder", "lineage", "k"],
                                how="left", suffixes=("", "_stored"), validate="one_to_one")
    K = pd.DataFrame(kn).merge(pd.read_csv(os.path.join(RES, "fm_finestate_knn_ALL_multi_v3.csv")),
                               on=["FM", "lineage"], how="left", suffixes=("", "_stored"), validate="one_to_one")
    for col in ("RECAST", "expression", "n_states"):
        print(f"[{s}] recall {col}: equal {int((R[col] == R[col + '_stored']).sum())}/{len(R)}, "
              f"max |d| {(R[col] - R[col + '_stored']).abs().max():.3g}", flush=True)
    for col in ("state_knn_acc", "state_macroF1"):
        print(f"[{s}] kNN {col}: equal {int((K[col] == K[col + '_stored']).sum())}/{len(K)}, "
              f"max |d| {(K[col] - K[col + '_stored']).abs().max():.3g}", flush=True)
    print(R.to_string(index=False), flush=True)
    C.write(R, f"encoders_{s}_recall.csv")
    C.write(K, f"encoders_{s}_knn.csv")


if __name__ == "__main__":
    main()
