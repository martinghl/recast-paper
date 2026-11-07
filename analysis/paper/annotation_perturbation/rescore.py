"""Marker recovery of the annotation-perturbation runs scored on the subtype marker panels (Supplementary Fig. 5f-h).

Nothing is re-run.  The stored top-20 selections of validation/label_noise are scored against the marker panels of
the benchmark (full_benchmark/ground_truth/panels/marker_panels.json; panel genes in the GO universe, as the benchmark
scores them):
  RECAST    validation/label_noise/ps_genes_<lin>_v3.json   (perturbed top 20; 'baseline' = clean top 50)
  Wilcoxon  validation/label_noise/ps_de_<lin>_top50_v2.csv (tie-corrected, clean and perturbed)
Recall = |top20 & panel| / |panel|, panel restricted to the lineage's genes; NaN when the panel is empty.  Jaccard,
cosine, eps_eff and the other columns are copied from the stored tables unchanged; the drop column of the effective-
shift table is recomputed from the new recall (as effective_shift.py computes it).

Out (this folder): out/annotation_perturbation_recast.csv, out/annotation_perturbation_comparators.csv,
out/annotation_perturbation_effective_shift.csv, out/summary.json (the numbers quoted in the text).
Env analysis:  python -I rescore.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SB = "/path/to/research/analysis"
PS = f"{SB}/validation/label_noise"
FSB = f"{SB}/subtype_benchmark"
PANELS = f"{FSB}/full_benchmark/ground_truth/panels/marker_panels.json"
LIN = ["tf_cd8_dom", "tf_b_yosh", "tf_mono_steph", "fs_lung_hlca"]
K = 20
KEY = ["lineage", "state", "mode", "eps", "seed"]

GO = set(l.strip() for l in open(f"{FSB}/results/go_genes.txt") if l.strip())
UNIT = json.load(open(PANELS))


def panels(lin, states, genes):
    gi = set(genes)
    assert {k.split("|", 1)[1] for k in UNIT if k.split("|", 1)[0] == lin} <= set(states), lin
    return {c: {g for g in set(UNIT.get(f"{lin}|{c}", [])) & GO if g in gi} for c in states}


def rec(top, mk):
    return len(set(top[:K]) & mk) / len(mk) if mk else np.nan


def parse(tag):                                          # as score_comparators_label_noise.parse_condition
    mode, e, s = tag.rsplit("_", 2)
    return mode, int(e[1:]) / 1000.0, int(s[1:])


def score():
    rf, rd = [], []
    for lin in LIN:
        meta = np.load(f"{PS}/ps_export_{lin}_meta_v3.npz", allow_pickle=True)
        genes, states = list(meta["genes"].astype(str)), [str(s) for s in meta["states"]]
        # the Wilcoxon lists were computed on the _v2 export, whose gene list is the same
        assert genes == list(np.load(f"{PS}/ps_export_{lin}_meta_v2.npz", allow_pickle=True)["genes"].astype(str))
        P = panels(lin, states, genes)
        j = json.load(open(f"{PS}/ps_genes_{lin}_v3.json"))
        for s in states:
            rf.append(dict(lineage=lin, state=s, mode="none", eps=0.0, seed=-1, recall20=rec(j["baseline"][s], P[s])))
        for k, top in j.items():
            if k == "baseline":
                continue
            cond, s = k.split("|", 1)
            mode, eps, seed = parse(cond)
            assert len(top) == K
            rf.append(dict(lineage=lin, state=s, mode=mode, eps=eps, seed=seed, recall20=rec(top, P[s])))
        d = pd.read_csv(f"{PS}/ps_de_{lin}_top50_v2.csv").sort_values(["condition", "state", "rank"])
        for (cond, s), g in d.groupby(["condition", "state"]):
            mode, eps, seed = parse(cond)
            rd.append(dict(lineage=lin, state=s, mode=mode, eps=eps, seed=seed,
                           recall20=rec(list(g.gene.astype(str)), P.get(s, set()))))
        yield lin, P
    score.F, score.D = pd.DataFrame(rf), pd.DataFrame(rd)


def swap(stored, new):
    """stored table with recall20 replaced by `new` (same rows, every other column untouched)"""
    stored = stored.copy()
    stored["_eps"] = stored.eps.round(3)
    new = new.assign(_eps=new.eps.round(3)).drop(columns="eps")
    m = stored.merge(new, on=["lineage", "state", "mode", "_eps", "seed"], how="left", suffixes=("_stored", ""),
                     validate="one_to_one")
    assert len(m) == len(stored)
    return m.drop(columns=["recall20_stored", "_eps"])[stored.columns]


def drops(F, E):
    """effective_shift.py's drop: clean recall - seed-mean boundary recall at eps (10% and 20%)"""
    E = E.copy()
    b = F[F["mode"] == "none"].set_index(["lineage", "state"]).recall20
    for eps in (0.10, 0.20):
        d = F[(F["mode"] == "boundary") & np.isclose(F.eps, eps)].groupby(["lineage", "state"]).recall20.mean()
        m = np.isclose(E.eps, eps)
        E.loc[m, "drop"] = [b.get((l, s), np.nan) - d.get((l, s), np.nan) for l, s in zip(E[m].lineage, E[m].state)]
    return E


def main():
    Fs, Ds, Es = (pd.read_csv(f"{PS}/{f}") for f in ("ps_recast_all_v3.csv", "ps_de_all_metrics_v2.csv",
                                                     "ps_epseff_v3.csv"))
    assert len(Fs) == len(Ds) == 496 and len(Es) == 32
    sizes = {}
    for lin, P in score():
        sizes.update({f"{lin}|{c}": len(v) for c, v in P.items()})
    F, D = swap(Fs, score.F), swap(Ds, score.D)
    E = drops(F, Es)[Es.columns]
    for t in (F, D):
        assert t.dropna(subset=["recall20"]).groupby(["lineage", "state"]).ngroups == 14
    os.makedirs(f"{HERE}/out", exist_ok=True)
    F.to_csv(f"{HERE}/out/annotation_perturbation_recast.csv", index=False)
    D.to_csv(f"{HERE}/out/annotation_perturbation_comparators.csv", index=False)
    E.to_csv(f"{HERE}/out/annotation_perturbation_effective_shift.csv", index=False)

    # ---------------------------------------------------------------- numbers the text quotes
    def pooled(T, mode, eps):
        return float(T[(T["mode"] == mode) & np.isclose(T.eps, eps)].groupby(["lineage", "state"])
                     .recall20.mean().dropna().mean())

    r = {"panel_sizes": sizes, "n_evaluable": int(F.dropna(subset=["recall20"]).groupby(["lineage", "state"]).ngroups)}
    for nm, T in (("RECAST", F), ("Wilcoxon", D)):
        r[nm] = {"clean": float(T[T["mode"] == "none"].recall20.mean()),
                 **{f"{m}_{int(round(e * 1000)):03d}": pooled(T, m, e) for m in ("boundary", "random")
                    for e in (0.025, 0.05, 0.10, 0.20, 0.40)}}
    for eps in (0.10, 0.20):
        C = E[np.isclose(E.eps, eps)].dropna(subset=["drop"])
        r[f"drop_{int(eps * 100)}"] = {f"{l}|{s}": [round(float(x), 4), round(float(d), 4)]
                                       for l, s, x, d in zip(C.lineage, C.state, C.eps_eff, C["drop"])}
        r[f"unchanged_{int(eps * 100)}"] = int((C["drop"].abs() < 1e-12).sum())
        r[f"no_loss_{int(eps * 100)}"] = int((C["drop"] <= 1e-12).sum())
        r[f"loss_gt5pp_{int(eps * 100)}"] = sorted(f"{l}|{s}" for l, s, d in zip(C.lineage, C.state, C["drop"])
                                                   if d > 0.05)
        r[f"max_drop_epseff_lt0.2_{int(eps * 100)}"] = float(C[C.eps_eff < 0.2]["drop"].max())
    json.dump(r, open(f"{HERE}/out/summary.json", "w"), indent=1)
    print(f"RECAST clean {r['RECAST']['clean']:.4f} | 10% boundary {r['RECAST']['boundary_100']:.4f} | "
          f"Wilcoxon clean {r['Wilcoxon']['clean']:.4f} | 10% boundary {r['Wilcoxon']['boundary_100']:.4f}")
    for e in (10, 20):
        print(f"  {e}%: unchanged {r[f'unchanged_{e}']}/14, no loss {r[f'no_loss_{e}']}/14, "
              f">5 pp loss {r[f'loss_gt5pp_{e}']}, max drop at eps_eff<0.2 {r[f'max_drop_epseff_lt0.2_{e}']:.4f}")
        for k, (x, d) in r[f"drop_{e}"].items():
            print(f"     {k:45s} eps_eff {x:.3f}  drop {d:+.4f}")

if __name__ == "__main__":
    main()
