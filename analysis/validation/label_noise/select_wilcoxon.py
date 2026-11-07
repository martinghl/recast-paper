"""Tie-corrected Wilcoxon (one subtype against its siblings) under the perturbed labels RECAST saw
(Supplementary Fig. 5f-h).

Per-gene ranks over the lineage cells do not depend on the labels, so the rank matrix and tie terms are computed once;
each condition x state is then a rank sum -> z -> top 50.

Usage: python select_wilcoxon.py [lineage]        (reads ps_export_<lin>_*.npz + ps_labels_<lin>/)
Out:   ps_de_<lin>_top50.csv   (condition, state, rank, gene); scored by score_comparators_label_noise.py"""
import os, sys, glob
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata

OUT = os.environ.get("LABEL_NOISE_DIR", "/path/to/research/analysis/validation/label_noise")
LIN = sys.argv[1] if len(sys.argv) > 1 else "tf_cd8_dom"
TOPN = 50

# RECAST_RESULT_TAG: read the counts and the perturbed labels written with the same tag by label_noise_runs.py
# (boundary-weighted relabelling depends on the embedding, so the label sets go with the run).
VTAG = os.environ.get("RECAST_RESULT_TAG", "")
cnt = sp.load_npz(f"{OUT}/ps_export_{LIN}_counts{VTAG}.npz").tocsr()
meta = np.load(f"{OUT}/ps_export_{LIN}_meta{VTAG}.npz", allow_pickle=True)
genes = meta["genes"].astype(str); lab0 = meta["lab0"].astype(str); STATES = list(meta["states"])
N = cnt.shape[0]
tot = np.asarray(cnt.sum(1)).ravel()
Xln = cnt.multiply(1e4 / np.clip(tot, 1, None)[:, None]).log1p().toarray().astype("float32")
print(f"[{LIN}] lognorm {Xln.shape}; ranking once …", flush=True)

try:
    R = rankdata(Xln, axis=0).astype("float64")             # scipy >= 1.10
except TypeError:
    R = np.apply_along_axis(rankdata, 0, Xln).astype("float64")

# tie term per gene: sum(t^3 - t) over tied groups. Sparse lognorm ties are dominated by zeros;
# compute exactly per gene via sorting.
def tie_terms(X):
    T = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        _, counts = np.unique(X[:, j], return_counts=True)
        c = counts[counts > 1].astype("float64")
        T[j] = (c ** 3 - c).sum()
    return T

TIE = tie_terms(Xln)
print(f"[{LIN}] tie terms done", flush=True)


def wilcoxon_top(mask):
    n1 = int(mask.sum()); n2 = N - n1
    R1 = R[mask].sum(0)
    mu = n1 * (N + 1) / 2.0
    var = (n1 * n2 / 12.0) * ((N + 1) - TIE / (N * (N - 1.0)))
    z = (R1 - mu) / np.sqrt(np.clip(var, 1e-12, None))
    return np.argsort(-z)[:TOPN]


rows = []
conds = [("none_e000_s-1", lab0)]
_lf = sorted(glob.glob(f"{OUT}/ps_labels_{LIN}/*{VTAG}.txt"))
if not VTAG:
    _lf = [f for f in _lf if not f.endswith("_v2.txt")]      # an untagged run reads only untagged label files
assert len(_lf) == 30, f"{len(_lf)} label files for tag {VTAG!r}"
for f in _lf:
    tag = os.path.basename(f)[:-4]
    if VTAG:
        tag = tag[:-len(VTAG)]                               # condition names stay arm-independent
    lab = np.array([l.rstrip("\n") for l in open(f)])
    assert len(lab) == N
    conds.append((tag, lab))

for tag, lab in conds:
    for s in STATES:
        mask = lab == s
        if mask.sum() < 20:
            continue
        top = wilcoxon_top(mask)
        for r, j in enumerate(top):
            rows.append((tag, s, r + 1, genes[j]))
    print(f"[{LIN}] {tag} done", flush=True)

pd.DataFrame(rows, columns=["condition", "state", "rank", "gene"]).to_csv(
    f"{OUT}/ps_de_{LIN}_top50{VTAG}.csv", index=False)
print(f"[wrote] ps_de_{LIN}_top50{VTAG}.csv  ({len(conds)} conditions)")
