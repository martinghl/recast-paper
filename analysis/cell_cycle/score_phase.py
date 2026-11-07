"""Score every method's panel for the phase-defined cell-cycle comparisons against the cell-cycle ground
truth (Supplementary Fig. 4c, d).

The ground truth is a pathway (1,815 genes), so Recall@20 could not exceed 20/1815 = 0.011.  The primary metric
is therefore

    precision@K = |top-K panel  n  cell-cycle genes| / K

at K = 20 (main), 50, 100 and 200, with a hypergeometric enrichment test against the genes measured in the
comparison; Recall@K is written as well.  All quantities use the genes measured in the comparison (>= 3 cells, the
filter the comparator methods saw), so RECAST and the comparators share one denominator.

Ground truth: the full Reactome / GO-BP cell-cycle union.  The groups come from the original authors' annotation,
not from a cell-cycle score, so the Tirosh S/G2M genes (MKI67, TOP2A, UBE2C, TPX2, BIRC5, TYMS, RRM2, GTSE1, ...)
stay in it; the set without them is a sensitivity variant.

Inputs
  phase_recast/<contrast>/panel.csv  RECAST panels (select_recast_phase.py)
  phase_genesets/<contrast>.csv     comparator panels (run_comparators_phase.sh -> export_comparator_genes_phase.R)
  cc_gt_full.txt                      ground truth;  cc_gt_tirosh_out.txt / cc_gt_core.txt = sensitivity
Outputs
  phase_scores_long.csv           comparison x method x K
  phase_leaderboard.csv           method means over comparisons, RECAST's rank
Env: SC (no GPU)."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

CR = "/path/to/research/analysis/cell_cycle"
KS = [20, 50, 100, 200]
RECAST = "RECAST_M0"


def load_gt(f):
    return {l.strip() for l in open(f) if l.strip()}


def measured_genes(contrast):
    f = f"{CR}/phase_sce_export/{contrast}_genes.txt"
    return [l.strip() for l in open(f) if l.strip()]


def panels_for(contrast):
    """method -> ranked gene list, RECAST and the DE field in one dict."""
    out = {}
    fp = f"{CR}/phase_recast/{contrast}/panel.csv"
    if os.path.exists(fp):
        out[RECAST] = pd.read_csv(fp).sort_values("rank").gene.astype(str).tolist()
    dp = f"{CR}/phase_genesets/{contrast}.csv"
    if os.path.exists(dp):
        d = pd.read_csv(dp)
        for m, g in d.sort_values("rank").groupby("method", sort=False):
            out[str(m)] = g.gene.astype(str).tolist()
    return out


def score(contrast, gt_all, tag):
    meas = measured_genes(contrast)
    meas_set = set(meas)
    gt = gt_all & meas_set                       # fair per-contrast denominator
    N, K_gt = len(meas_set), len(gt)
    rows = []
    for method, genes in panels_for(contrast).items():
        genes = [g for g in genes if g in meas_set]
        for K in KS:
            top = genes[:K]
            if not top:
                continue
            hit = len({g for g in top} & gt)
            # P(X >= hit) under the hypergeometric null of drawing len(top) genes from the measured set
            p = float(hypergeom.sf(hit - 1, N, K_gt, len(top))) if hit else 1.0
            exp = len(top) * K_gt / N
            rows.append(dict(gt=tag, contrast=contrast, method=method, K=K, n_top=len(top),
                             hits=hit, precision=hit / len(top), expected=exp,
                             enrichment=(hit / exp) if exp else np.nan,
                             recall=hit / K_gt if K_gt else np.nan, p_hyper=p,
                             n_measured=N, n_gt_measured=K_gt))
    return rows


def main():
    contrasts = sorted(n for n in os.listdir(f"{CR}/phase_contrasts")
                       if os.path.exists(f"{CR}/phase_contrasts/{n}/counts.npz"))
    # GT variants. See the "Which ground truth" note in the module docstring for why `full` is primary.
    gts = [("primary", load_gt(f"{CR}/cc_gt_full.txt")),
           ("tirosh_out", load_gt(f"{CR}/cc_gt_tirosh_out.txt"))]
    if os.path.exists(f"{CR}/cc_gt_core.txt"):
        gts.append(("core", load_gt(f"{CR}/cc_gt_core.txt")))

    rows = []
    for tag, gt in gts:
        for c in contrasts:
            rows += score(c, gt, tag)
    long = pd.DataFrame(rows)
    long.to_csv(f"{CR}/phase_scores_long.csv", index=False)

    # leaderboard on the primary GT at the headline depth.
    # NB: bracket access on "gt" is required, not stylistic -- `long.gt` resolves to DataFrame.gt(),
    # the greater-than method, so `long.gt == "primary"` is a plain False and silently masks
    # everything out. That made this function report "no DE panels yet" even with panels present.
    # COVERAGE GATE. A method's mean is only comparable to another's if both were scored on the same
    # contrasts. In a 58-method battery some methods fail on some contrasts, and a method scored only
    # where it happens to do well would outrank one scored everywhere -- a pure artefact. So the
    # ranked leaderboard admits only methods present in every contrast, and partial methods are
    # written out separately, ranked against nobody. (A probe carrying RECAST's own panel on its single
    # best contrast scored 0.850 against RECAST's 12-contrast mean of 0.612, which is how this was found.)
    n_methods = long.method.nunique()
    all_contrasts = set(long.contrast.unique())
    cover = long[long["gt"] == "primary"].groupby("method").contrast.nunique()
    complete = set(cover[cover == len(all_contrasts)].index)
    partial = sorted(set(cover.index) - complete)
    if partial:
        print(f"[coverage] {len(partial)} of {n_methods} methods missing contrasts, "
              f"excluded from the ranked leaderboard -> phase_leaderboard_partial.csv")
        for m in partial[:10]:
            print(f"           {m}: {cover[m]}/{len(all_contrasts)} contrasts")
        if len(partial) > 10:
            print(f"           ... and {len(partial) - 10} more")
        if RECAST in partial:
            print(f"           WARNING: {RECAST} itself is incomplete -- fix before reading any rank")

    board, board_partial = [], []
    for K in KS:
        s = long[(long["gt"] == "primary") & (long["K"] == K)]
        if s.empty:
            continue
        agg = dict(mean_precision=("precision", "mean"), median_precision=("precision", "median"),
                   mean_enrichment=("enrichment", "mean"), n_contrasts=("contrast", "nunique"))
        g = s[s.method.isin(complete)].groupby("method").agg(**agg).reset_index()
        g["K"] = K
        g["rank"] = g.mean_precision.rank(ascending=False, method="min").astype(int)
        board.append(g)
        if partial:
            p = s[s.method.isin(partial)].groupby("method").agg(**agg).reset_index()
            p["K"] = K
            board_partial.append(p)
    if board_partial:
        pd.concat(board_partial, ignore_index=True).sort_values(["K", "mean_precision"],
                                                                ascending=[True, False]).to_csv(
            f"{CR}/phase_leaderboard_partial.csv", index=False)
    if not board:
        # Only legitimate when nothing scored at all. If rows exist, an empty board means a masking
        # bug like the .gt collision above, not a missing input -- fail loudly rather than mislead.
        assert long.empty, f"leaderboard empty but {len(long)} score rows exist -- masking bug"
        print("no scores at all -- check phase_recast/ and phase_genesets/ inputs")
        return
    if n_methods == 1:
        print("RECAST-only: no DE panels in phase_genesets/ yet, so the leaderboard is a single row.")
    bd = pd.concat(board, ignore_index=True).sort_values(["K", "rank"])
    bd.to_csv(f"{CR}/phase_leaderboard.csv", index=False)

    for K in KS:
        s = bd[bd.K == K]
        if s.empty:
            continue
        n = len(s)
        f = s[s.method == RECAST]
        head = ", ".join(f"{r.method} {r.mean_precision:.3f}" for r in s.head(5).itertuples())
        if len(f):
            r = f.iloc[0]
            print(f"[K={K:3d}] RECAST precision {r.mean_precision:.3f} "
                  f"(enrichment {r.mean_enrichment:.1f}x) rank {int(r['rank'])}/{n} "
                  f"over {int(r.n_contrasts)} contrasts (all-{len(all_contrasts)} field) | top5: {head}")
        else:
            print(f"[K={K:3d}] RECAST absent | top5: {head}")


if __name__ == "__main__":
    main()
