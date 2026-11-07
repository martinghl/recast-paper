"""Supplementary Fig. 5f-h: RECAST selections under perturbed subtype labels, recomputed with the package.

validation/label_noise/label_noise_runs.py relabelled 2.5-40% of the cells of four lineages (boundary-focused
or random, three seeds each), wrote every perturbed label vector to ps_labels_<lineage>/<condition>_v3.txt and
the top 20 genes per subtype and condition to ps_genes_<lineage>_v3.json (top 50 for the clean labels). The
recovery and overlap numbers of the figure are computed from those stored lists.

Here every stored label vector is given to recast.attribute (each subtype with at least 20 cells against the
rest of its lineage) and the package's lists are compared with the stored ones. The perturbation generator is
also rerun on the package's embeddings, to show that it produces the same label vectors. The boundary mode ranks
cells by a difference of two distances, so the embedding is made as the original run made it: from the
attribution object's log-normalized matrix in batches of 512 cells.

    python check_label_noise.py
Writes label_noise.csv."""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import common as C
import recast

PS = os.path.join(C.SB, "validation", "label_noise")
EPS = [0.025, 0.05, 0.10, 0.20, 0.40]
SEEDS = [0, 1, 2]
LINEAGES = {
    "tf_cd8_dom": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                   ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
    "tf_b_yosh": ("yoshida", "bench_labels_yoshida.txt",
                  ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells", "Memory B cells"]),
    "tf_mono_steph": ("stephenson", "bench_labels_stephenson.txt", ["Classical monocytes", "Non-classical monocytes"]),
    "fs_lung_hlca": ("hlca_lung", "bench_labels_hlca_lung.txt", ["AT1", "AT2", "basal", "ciliated", "club", "ionocyte"]),
}


def generator(Z, lab0, states):
    """label_noise_runs.perturb: boundary mode relabels cells drawn from the 2k most ambiguous (own-centroid distance
    minus nearest-other distance) to their nearest other subtype; random mode relabels k random cells."""
    cen = {s: Z[lab0 == s].mean(0) for s in states}
    D = np.stack([np.linalg.norm(Z - cen[s], axis=1) for s in states], axis=1)
    own = np.array([states.index(s) for s in lab0])
    d_own = D[np.arange(len(lab0)), own]
    D[np.arange(len(lab0)), own] = np.inf
    near = np.argmin(D, axis=1)
    amb = np.argsort(-(d_own - D[np.arange(len(lab0)), near]))

    def perturb(mode, eps, seed):
        rng = np.random.default_rng(seed)
        n = len(lab0); k = int(round(eps * n))
        lab = lab0.copy()
        if mode == "boundary":
            for i in rng.choice(amb[:min(2 * k, n)], size=k, replace=False):
                lab[i] = states[near[i]]
        else:
            for i in rng.choice(n, size=k, replace=False):
                lab[i] = states[rng.choice([j for j in range(len(states)) if j != own[i]])]
        return lab
    return perturb


def main():
    enc = C.scimilarity()
    rows = []
    for lin, (stem, labfile, states) in LINEAGES.items():
        _, obs = C.attribution_obs(C.ATLASES, stem)
        lab_all = (obs["label"].astype(str).to_numpy() if "label" in obs
                   else C.read_labels(os.path.join(C.ATLASES, labfile)))
        a, _ = C.lineage_adata(C.ATLASES, stem, lab_all, states, min_cells=1)
        lab0 = a.obs["state"].to_numpy()
        A = ad.read_h5ad(os.path.join(C.ATLASES, f"{stem}_attribution_scimilarity.h5ad"), backed="r")
        rows_ = A.obs_names.astype(str).get_indexer(a.obs_names)
        assert (rows_ >= 0).all() and (np.diff(rows_) > 0).all()
        Xln = A.X[rows_]
        Xln = (Xln.toarray() if sp.issparse(Xln) else np.asarray(Xln)).astype("float32")
        A.file.close()
        Z = enc.embed(Xln, batch_size=512)
        stored = json.load(open(os.path.join(PS, f"ps_genes_{lin}_v3.json")))
        perturb = generator(Z, lab0, states)

        def run(lab, depth):
            a.obs["lab"] = lab
            tg = [s for s in states if (lab == s).sum() >= 20]
            res = recast.attribute(enc, a, "lab", target=tg, reference="rest", gate="dC", centroid="mean_lognorm",
                                   qc="off", embeddings=Z)
            return {s: res.genes[s][:depth] for s in tg}

        for s, g in run(lab0, 50).items():
            rows.append(dict(lineage=lin, condition="clean", state=s, labels_regenerated=True,
                             **C.compare_lists(g, stored["baseline"][s])))
        n_same = 0
        for mode in ("boundary", "random"):
            for eps in EPS:
                for seed in SEEDS:
                    tag = f"{mode}_e{int(eps * 1000):03d}_s{seed}"
                    labels = np.array([l.rstrip("\n") for l in open(os.path.join(PS, f"ps_labels_{lin}", f"{tag}_v3.txt"))
                                       if l.rstrip("\n")])
                    assert len(labels) == len(lab0), (lin, tag)
                    regen = bool((perturb(mode, eps, seed) == labels).all())
                    n_same += regen
                    for s, g in run(labels, 20).items():
                        rows.append(dict(lineage=lin, condition=tag, state=s, labels_regenerated=regen,
                                         **C.compare_lists(g, stored[f"{tag}|{s}"])))
        R = pd.DataFrame([r for r in rows if r["lineage"] == lin])
        print(f"[{lin}] {len(R)} lists, identical {int(R.identical.sum())}, same top-20 set "
              f"{int(R.same_set_top20.sum())}; perturbed label vectors regenerated {n_same}/30", flush=True)
        C.assert_package_only()
    R = pd.DataFrame(rows)
    print(f"\nall: {len(R)} lists, identical {int(R.identical.sum())}, same top-20 set {int(R.same_set_top20.sum())}")
    if not R.identical.all():
        print(R[~R.identical][["lineage", "condition", "state", "first_diff_rank", "overlap_top20"]].to_string(index=False))
    C.write(R, "label_noise.csv")


if __name__ == "__main__":
    main()
