"""Per-subtype effective relabelling fraction of the label-noise runs (Supplementary Fig. 5f-h).

eps_eff = |T triangle T'| / |T| for every subtype: the lineage-level fraction eps is amplified about
n_lineage / n_state for small subtypes under boundary-weighted relabelling.  Large drops of recovery should occur only
where eps_eff is large.  Computed from the stored perturbed-label files and the clean labels (lab0 in the meta npz).
Out: ps_epseff<tag>.csv, a Spearman correlation and the cases where a large drop has a small eps_eff."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import os as _os
# suffix of the input tables (README: result tags); the pooled table is written by merge_recast.py
VTAG = _os.environ.get("RECAST_RESULT_TAG", "")

OUT = "/path/to/research/analysis/validation/label_noise"
LINS = ["tf_cd8_dom", "tf_b_yosh", "tf_mono_steph", "fs_lung_hlca"]

rows = []
for lin in LINS:
    meta = np.load(f"{OUT}/ps_export_{lin}_meta.npz", allow_pickle=True)
    lab0 = meta["lab0"].astype(str); states = meta["states"].astype(str)
    for eps_tag, eps in [("e100", 0.10), ("e200", 0.20)]:
        for s in states:
            T0 = lab0 == s
            effs = []
            for seed in range(3):
                labp = np.array([l.rstrip("\n") for l in
                                 open(f"{OUT}/ps_labels_{lin}/boundary_{eps_tag}_s{seed}.txt")])
                T1 = labp == s
                effs.append((T0 & ~T1).sum() + (~T0 & T1).sum())
            rows.append(dict(lineage=lin, state=s, eps=eps, n_state=int(T0.sum()),
                             frac_of_lineage=float(T0.mean()),
                             eps_eff=float(np.mean(effs)) / T0.sum()))
E = pd.DataFrame(rows)

F = pd.read_csv(f"{OUT}/ps_recast_all{VTAG}.csv")
b = F[F["mode"] == "none"].set_index(["lineage", "state"]).recall20
for eps in [0.10, 0.20]:
    d = F[(F["mode"] == "boundary") & (F.eps == eps)].groupby(["lineage", "state"]).recall20.mean()
    m = E.eps == eps
    E.loc[m, "drop"] = [b.get((l, s), np.nan) - d.get((l, s), np.nan)
                        for l, s in zip(E[m].lineage, E[m].state)]
E.to_csv(f"{OUT}/ps_epseff{VTAG}.csv", index=False)
print(E[E.eps == 0.10].round(3).to_string(index=False))
for eps in [0.10, 0.20]:
    V = E[(E.eps == eps)].dropna(subset=["drop"])
    r = spearmanr(V.eps_eff, V["drop"])
    print(f"\nboundary {int(eps*100)}%: Spearman(eps_eff, recall drop) rho={r.statistic:.3f} "
          f"p={r.pvalue:.4f} (n={len(V)})")
    quiet = V[V.eps_eff < 0.2]
    print(f"             bound check: max drop among eps_eff<0.2 states = {quiet['drop'].max():.3f} "
          f"({len(quiet)} states); drops>0.05 all at eps_eff = "
          f"{sorted(V[V['drop'] > 0.05].eps_eff.round(2).tolist())}")
