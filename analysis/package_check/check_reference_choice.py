"""Fig. 4: reference choice.

Part 1 (Fig. 4b-e and the filled points of 4f): 16 Zhao subtypes, seven references each (nearest sibling,
siblings, broad lineage, all other cells, embedding neighbours, random cells, an unrelated lineage). The reference
cells are chosen exactly as in validation/reference_choice/reference_choice.py (the random reference with the same
generator and the same sequence of draws); each target-reference pair is one recast.attribute call. Compared: the
top 20 genes (ref_robust_genes_v3.json) and the marker recovery and within-lineage AUROC (ref_robust_metrics_v3.csv).
Fig. 4e is computed from these lists alone.

Part 2 (open points of Fig. 4f): 14 subtypes in four immune lineages, references made of whole annotated clusters
chosen by embedding similarity (subtype_benchmark/embedding_reference.py). Each rule is one recast.attribute call
with a list of reference labels. Compared: marker recovery (labelfree_recovery_v3.csv, which kept no gene lists).

    python check_reference_choice.py
Writes reference_choice_lists.csv, reference_choice_metrics.csv and reference_choice_cluster_references.csv."""
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import common as C
import recast

RR = os.path.join(C.SB, "validation", "reference_choice")
RES = os.path.join(C.FSB, "results")
K = 20

# ---- configuration copied from reference_choice.py
CURATED = {
    "zhao_Mono": {"CD14+ Mo": ["CD14", "S100A8", "S100A9", "S100A12", "VCAN", "FCN1", "LYZ", "MNDA", "CSF3R", "S100A6"],
                  "CD16+ Mo": ["FCGR3A", "CDKN1C", "LST1", "MS4A7", "CX3CR1", "RHOC", "SIGLEC10", "IFITM3", "HES4", "APOBEC3A"],
                  "MerTK+ Mo": ["MERTK", "C1QA", "C1QB", "C1QC", "CLEC10A", "F13A1", "SELENOP", "CD163", "STAB1", "FOLR2"]},
    "zhao_CD8": {"CCR7+ CD8": ["CCR7", "SELL", "TCF7", "LEF1", "MAL", "NOSIP", "IL7R", "TRABD2A"],
                 "CRCR3+ CD8": ["CXCR3", "GZMK", "IL7R", "CD27", "GPR183", "LTB", "CD28", "SLAMF1"],
                 "CX3CR1+ CD8": ["CX3CR1", "FGFBP2", "GNLY", "GZMB", "GZMH", "PRF1", "NKG7", "FCGR3A", "KLRD1", "ZNF683"],
                 "CXCR6+ CD8": ["CXCR6", "ITGAE", "ZNF683", "CD69", "XCL1", "XCL2", "CD160", "KLRC1", "TRDC", "KLRF1"]},
    "zhao_CD4": {"CCR7+ CD4": ["CCR7", "SELL", "TCF7", "LEF1", "MAL", "ADTRP", "TSHZ2", "NOSIP"],
                 "CXCR3+ CD4": ["CXCR3", "GZMK", "GZMA", "CCL5", "NKG7", "KLRG1", "HOPX", "CST7"],
                 "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT", "RTKN2", "LAIR2", "TNFRSF18"]},
    "zhao_B": {"Naive B": ["IGHD", "TCL1A", "FCER2", "IL4R", "YBX3", "BACH2", "IGHM"],
               "TCL1A+ naive B": ["TCL1A", "IGHD", "FCER2", "PLPP5", "CD72", "IL4R"],
               "Classical mem B": ["CD27", "TNFRSF13B", "AIM2", "CD82", "S100A10", "COCH"],
               "CD11c+ mem B": ["ITGAX", "FCRL5", "TBX21", "ZEB2", "FGR", "NKG7", "SLC11A1"],
               "ZBTB32+ mem B": ["ZBTB32", "NKG7", "GZMA", "CCL5", "MT1F"],
               "MZB": ["CR2", "CD1C", "CD24", "PLD4", "RALGPS2", "DNASE1L3"]},
}
BROAD = {
    "T": ["CCR7+ CD4", "CXCR3+ CD4", "Treg", "CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8", "MAIT"],
    "B": ["CD11c+ mem B", "Classical mem B", "MZB", "Naive B", "TCL1A+ naive B", "ZBTB32+ mem B"],
    "Mono": ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo", "Macrophage", "CD1c+ DCs"],
}
LIN = {
    "zhao_Mono": (["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"], "Mono", "T"),
    "zhao_CD8": (["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"], "T", "B"),
    "zhao_CD4": (["CCR7+ CD4", "CXCR3+ CD4", "Treg"], "T", "B"),
    "zhao_B": (["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"], "B", "Mono"),
}
BSTAB = 40      # bootstrap draws per reference in reference_choice.py (replayed only to keep the generator in step)

# ---- configuration copied from recast_subtypes.py (labels, markers) for the label-free rules
CFG = {
    "pbmcbmn_l2": ("pbmcbmn", "bench_labels_pbmcbmn_l2.txt", ["B naive", "B intermediate", "B memory"]),
    "yoshida": ("yoshida", "bench_labels_yoshida.txt",
                ["Naive B cells", "Non-switched memory B cells", "Switched memory B cells", "Memory B cells"]),
    "helperT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
                ["Tcm/Naive helper T cells", "Tem/Effector helper T cells", "Regulatory T cells"]),
    "cytoT": ("dominguez_conde", "bench_labels_dominguez_conde.txt",
              ["Tcm/Naive cytotoxic T cells", "Tem/Temra cytotoxic T cells", "Tem/Trm cytotoxic T cells", "gdT"]),
}
MARKERS = {
    "naive": {"TCL1A", "IGHD", "FCER2", "IL4R", "YBX3", "TSPAN13", "PLPP5", "BACH2", "SELL", "VPREB3", "CD72", "IGHM", "APBB2", "PCDH9"},
    "memory": {"CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH", "CRTAM", "CD80", "ITGB1", "TEX9", "GRIP2", "LINC01781", "BANK1"},
    "switched": {"IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGHGP", "CD27", "TNFRSF13B", "AIM2", "S100A10"},
    "nonswitched": {"IGHM", "IGHD", "CD27", "TNFRSF13B", "AIM2", "S100A10", "COCH"},
    "intermediate": {"TNFRSF13B", "COCH", "AIM2", "CRTAM", "MARCKS", "RALGPS2", "SSPN", "SCIMP", "CD72"},
    "plasma": {"MZB1", "XBP1", "PRDM1", "DERL3", "JCHAIN", "TNFRSF17", "CD38", "SLAMF7", "ITM2C", "POU2AF1", "SEC11C", "FKBP11"},
    "naiveT": {"CCR7", "SELL", "TCF7", "LEF1", "IL7R", "TXK", "NOSIP", "MAL", "TRABD2A", "ADTRP", "PLAC8", "LDHB", "CD27", "ACTN1"},
    "eff_helper": {"GZMK", "GZMA", "CCL5", "KLRB1", "ANXA1", "S100A4", "IL32", "AHNAK", "CCR6", "GPR183", "CRTAM", "TNFRSF4", "CCR4"},
    "treg": {"FOXP3", "IL2RA", "IKZF2", "CTLA4", "TNFRSF4", "TNFRSF18", "TIGIT", "RTKN2", "TBC1D4", "LAYN", "IL32", "IL10RA"},
    "eff_cyto": {"GZMB", "GZMH", "GNLY", "NKG7", "PRF1", "FGFBP2", "FCGR3A", "KLRD1", "CX3CR1", "SPON2", "KLRG1", "CMC1", "GZMA", "KLRF1"},
    "trm": {"ITGAE", "ITGA1", "ZNF683", "CXCR6", "XCL1", "XCL2", "CD69", "GZMK", "DUSP2", "CD8A", "CD160", "KLRC1"},
    "gdt": {"TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "KLRC1", "KLRD1", "TRGV9", "TRDV3", "KLRG1", "NKG7"},
}
TMAP = {"Regulatory T cells": "treg", "Tcm/Naive helper T cells": "naiveT", "Tem/Effector helper T cells": "eff_helper",
        "Tcm/Naive cytotoxic T cells": "naiveT", "Tem/Temra cytotoxic T cells": "eff_cyto",
        "Tem/Trm cytotoxic T cells": "trm", "gdT": "gdt"}


def state_key(lbl):
    if lbl in TMAP:
        return TMAP[lbl]
    low = lbl.lower()
    if "plasma" in low:
        return "plasma"
    if "switched" in low and "non" not in low and "unswitched" not in low:
        return "switched"
    if "non-switched" in low or "unswitched" in low:
        return "nonswitched"
    if "intermediate" in low:
        return "intermediate"
    if "naive" in low:
        return "naive"
    return "memory"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def dataset(stem):
    """All cells of a phase-2 dataset: raw counts in SCimilarity's gene order, the attribution object's lognorm
    matrix (used only to score gene sets, as the original scripts did) and its obs."""
    genes = C.scimilarity_genes()
    att = ad.read_h5ad(os.path.join(C.ATLASES, f"{stem}_attribution_scimilarity.h5ad"))
    names = np.asarray(att.obs_names).astype(str)
    assert list(att.var_names) == genes
    Xln = sp.csr_matrix(att.X)
    obs = att.obs.copy()
    del att
    a = ad.AnnData(C.counts(C.ATLASES, stem, names, genes))
    a.var_names = genes
    a.obs_names = names
    return a, Xln, obs


def top_vs(enc, a, Z, tgt_idx, ref_idx, n=K):
    """RECAST top genes for one target cell set against one reference cell set (one package call)."""
    grp = np.full(a.n_obs, "other", dtype=object)
    grp[ref_idx] = "R"
    grp[tgt_idx] = "T"
    a.obs["grp"] = pd.Categorical(grp.astype(str))
    res = recast.attribute(enc, a, "grp", target="T", reference=["R"], gate="dC", centroid="mean_lognorm",
                           qc="off", embeddings=Z)
    return res.genes["T"][:n]


def part1(enc):
    a, Xln, obs = dataset("zhao")
    genes = np.asarray(a.var_names); gi = {g: i for i, g in enumerate(genes)}
    lab = obs["label"].astype(str).to_numpy()
    Z = enc.embed(a.X)
    stored_genes = json.load(open(os.path.join(RR, "ref_robust_genes_v3.json")))
    rng = np.random.default_rng(0)
    rows, lists = [], []
    for tag, (sibs, broad, wrong) in LIN.items():
        in_lin = np.isin(lab, sibs)
        Xl = Xln[in_lin]
        for t in sibs:
            tm = lab == t
            tgt_idx = np.where(tm)[0]
            if len(tgt_idx) < 20:
                continue
            others = [s for s in sibs if s != t]
            tvec = np.asarray(Xln[tm].mean(0)).ravel()
            nearest = max(others, key=lambda s: np.corrcoef(tvec, np.asarray(Xln[lab == s].mean(0)).ravel())[0, 1])
            nsib = int(in_lin.sum() - tm.sum())
            cen = Z[tm].mean(0)
            d2 = ((Z - cen) ** 2).sum(1); d2[tm] = np.inf
            nontgt = np.where(~tm)[0]
            refs = {"nearest": np.where(lab == nearest)[0],
                    "sibling": np.where(np.isin(lab, others))[0],
                    "broad": np.where(np.isin(lab, [s for s in BROAD[broad] if s != t]))[0],
                    "global": nontgt,
                    "geometry": np.argsort(d2)[:max(nsib, 20)],
                    "random": rng.choice(nontgt, size=max(nsib, 20), replace=False),
                    "wrong": np.where(np.isin(lab, BROAD[wrong]))[0]}
            refs = {k: v for k, v in refs.items() if len(v) >= 5}
            for rn, ridx in refs.items():                    # the original bootstrap draws, replayed unused
                for _ in range(BSTAB):
                    rng.integers(0, len(tgt_idx), len(tgt_idx)); rng.integers(0, len(ridx), len(ridx))
            ymask = lab[in_lin] == t
            for rn, ridx in refs.items():
                top = top_vs(enc, a, Z, tgt_idx, ridx)
                stored = stored_genes[f"{tag}|{t}"][rn]
                lists.append(dict(lineage=tag, target=t, reference=rn, n_ref=len(ridx), **C.compare_lists(top, stored)))
                mk = [g for g in CURATED[tag][t] if g in gi]
                idx = [gi[g] for g in top]
                Xs = Xl[:, idx].toarray()
                ot = np.mean([Xs[ymask, j].mean() / (Xs[ymask, j].mean() + Xs[~ymask, j].mean() + 1e-12)
                              for j in range(len(idx))])
                rows.append(dict(lineage=tag, target=t, reference=rn, n_ref=len(ridx), nearest=nearest,
                                 recall20=len(set(top) & set(mk)) / len(mk), ontarget=float(ot),
                                 auroc=float(roc_auc_score(ymask.astype(int), Xs.mean(1)))))
            print(f"  [{tag}/{t}] " + " ".join(f"{x['reference']}={'=' if x['identical'] else x['overlap_top20']}"
                                               for x in lists[-len(refs):]), flush=True)
    a.obs.drop(columns="grp", inplace=True)
    L, M = pd.DataFrame(lists), pd.DataFrame(rows)
    stored = pd.read_csv(os.path.join(RR, "ref_robust_metrics_v3.csv"))
    M = M.merge(stored[["lineage", "target", "reference", "n_ref", "nearest", "recall20", "ontarget", "auroc"]],
                on=["lineage", "target", "reference"], suffixes=("", "_stored"), validate="one_to_one")
    assert len(M) == len(stored) == len(L), (len(M), len(stored), len(L))
    for c in ("recall20", "ontarget", "auroc"):
        M[f"d_{c}"] = (M[c] - M[f"{c}_stored"]).abs()
    print(f"\nPart 1: top-20 lists identical {int(L.identical.sum())}/{len(L)}, same set {int(L.same_set_top20.sum())}/"
          f"{len(L)}; reference sizes equal {int((M.n_ref == M.n_ref_stored).sum())}/{len(M)}; nearest equal "
          f"{int((M.nearest == M.nearest_stored).sum())}/{len(M)}")
    print("max |package - original|:", M[["d_recall20", "d_ontarget", "d_auroc"]].max().to_dict())
    print(M.groupby("reference")[["recall20", "recall20_stored", "auroc", "auroc_stored"]].mean().round(4).to_string())
    C.write(L, "reference_choice_lists.csv")
    C.write(M, "reference_choice_metrics.csv")


def part2(enc):
    t2sib, src = {}, {}
    for key, (stem, finef, states) in CFG.items():
        for c in states:
            t2sib[(stem, c)] = [s for s in states if s != c]
            src.setdefault(stem, {"finef": finef, "targets": []})
            if c not in src[stem]["targets"]:
                src[stem]["targets"].append(c)
    rows = []
    for stem in ["pbmcbmn", "yoshida", "dominguez_conde"]:
        a, _, _ = dataset(stem)
        genes = set(a.var_names)
        fine = C.read_labels(os.path.join(C.ATLASES, src[stem]["finef"]))
        assert len(fine) == a.n_obs
        a.obs["fine"] = pd.Categorical(fine)
        Z = enc.embed(a.X)
        states = [s for s in np.unique(fine) if (fine == s).sum() >= 20]
        cen = {s: Z[fine == s].mean(0) for s in states}
        ucen = {s: unit(cen[s]) for s in states}
        for c in src[stem]["targets"]:
            tgt = fine == c
            others = [s for s in states if s != c]
            order = sorted(others, key=lambda s: -float(ucen[c] @ ucen[s]))
            rho = float(np.linalg.norm(cen[c] - cen[order[0]]) / (np.linalg.norm(cen[c] - Z[~tgt].mean(0)) + 1e-12))
            thr = float(ucen[c] @ unit(Z.mean(0)))
            auto = [s for s in order if float(ucen[c] @ ucen[s]) > thr]
            rules = {"anno_sib": t2sib[(stem, c)], "vs_all": others, "near_auto": (auto or order[:1])}
            for k in [1, 2, 3, 5]:
                rules[f"near{k}"] = order[:k]
            rules["phase0"] = order[:1] if rho < 0.7 else others
            mk = MARKERS[state_key(c)] & genes
            for rule, rlist in rules.items():
                if np.isin(fine, rlist).sum() < 10:
                    continue
                res = recast.attribute(enc, a, "fine", target=c, reference=list(rlist), gate="dC",
                                       centroid="mean_lognorm", qc="off", embeddings=Z)
                top = res.genes[c][:K]
                rows.append(dict(source=stem, target=c, rule=rule, recall20=round(len(set(top) & mk) / len(mk), 4),
                                 rho=round(rho, 3),
                                 ref_states=";".join(rlist[:6]) + ("..." if len(rlist) > 6 else ""),
                                 top20=";".join(top)))
            print(f"  [{stem}] {c}", flush=True)
        del a
    L = pd.DataFrame(rows)
    stored = pd.read_csv(os.path.join(RES, "labelfree_recovery_v3.csv"))
    M = L.merge(stored, on=["source", "target", "rule"], suffixes=("", "_stored"), validate="one_to_one")
    assert len(M) == len(L) == len(stored), (len(M), len(L), len(stored))
    print(f"\nPart 2: recall equal {int((M.recall20 == M.recall20_stored).sum())}/{len(M)}, reference clusters equal "
          f"{int((M.ref_states == M.ref_states_stored).sum())}/{len(M)}, rho equal {int((M.rho == M.rho_stored).sum())}/{len(M)}")
    print(M.groupby("rule")[["recall20", "recall20_stored"]].mean().round(4).to_string())
    C.write(M, "reference_choice_cluster_references.csv")


if __name__ == "__main__":
    enc = C.scimilarity()
    part1(enc)
    C.assert_package_only()
    part2(enc)
    C.assert_package_only()
