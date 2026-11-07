"""Tables of the subtype marker panels (Supplementary Data sheets 12-14).

Panels
  subtype benchmark   full_benchmark/ground_truth/panels/marker_panels.json (97 units)
  pancreatic control  the same rules on the four Baron endocrine types (panel_rules.build, as build_panels.py calls it;
                      Baron is not part of the benchmark); its subtype -> CellMarker name lists are in candidate_markers.XW
  unfiltered          the same candidate walk without the sibling rule: genes in the vocabulary, top 20 by support
Counts (sheets 12, 14) walk each unit's full candidate pool in the order of the rules, as counts/construction_counts.py does
(candidates -> outside the vocabulary -> assigned to or tied with a sibling -> beyond the 20 best-supported -> final);
the walk must reproduce the stored panels and construction counts.
Rankings: the benchmark's inputs (scoring/in/panels, scoring/in/genesets50; the marker_recall_all loaders and `pick`).
Metrics (sheet 13): ranking_metrics.metrics; rank among RECAST and the 50 method configurations (the five panel
selection tools and Seurat ROC are outside this set).
Check: recall@20 per unit equals the benchmark table (scoring/out/recall_by_subtype.csv).
Env analysis.  Usage: python -I panel_construction.py
  -> out/panel_construction_by_unit.csv, out/alternative_metrics_cellmarker.csv, out/panel_construction_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
OUT = f"{HERE}/out"
for p in (f"{FULL_BENCHMARK}/ground_truth", FSB):
    sys.path.insert(0, p)
import candidate_markers as B            # noqa: E402
import panel_rules as V         # noqa: E402
from ranking_metrics import metrics  # noqa: E402
from cluster_key import pick       # noqa: E402

PANELS, GS50 = f"{FULL_BENCHMARK}/scoring/in/panels", f"{FULL_BENCHMARK}/scoring/in/genesets50"
W = "scanpy_wilcoxontiecorrect_rankby_raw"
OUT_OF_FIELD = {"seurat_roc", "scmarkergene", "nsforest", "rankcorr", "scmags", "scgenefit"}
METRICS = [("recall@10", "Recall@10", "Marker recovery", 10), ("recall@20", "Recall@20", "Marker recovery", 20),
           ("recall@50", "Recall@50", "Marker recovery", 50), ("AP@20", "AP@20", "Average precision", 20),
           ("NDCG@20", "NDCG@20", "Normalized discounted cumulative gain", 20)]
BARON = "fs_panc_baron"
# display names of the best other method; a method outside this table is reported by its id
NAME = {"scran_scoreMarkers_rank_logFC_cohen": "scran scoreMarkers (rank, Cohen logFC)",
        "scran_scoreMarkers_min_logFC_cohen": "scran scoreMarkers (min, Cohen logFC)",
        "scran_scoreMarkers_mean_logFC_cohen": "scran scoreMarkers (mean, Cohen logFC)",
        "scran_scoreMarkers_max_logFC_cohen": "scran scoreMarkers (max, Cohen logFC)",
        "scran_scoreMarkers_rank_AUC": "scran scoreMarkers (rank, AUC)",
        "scanpy_wilcoxontiecorrect_rankby_raw": "Wilcoxon (tie-corrected)", "cosg": "COSG",
        "seurat_wilcox": "Seurat Wilcoxon", "lm_two_sample": "two-sample linear model",
        "logfc_raw": "Log fold-change"}   # as labelled in Fig. 3


def lists(lin):
    """{method: {state: ranked genes}} as marker_recall_all reads them (M0 panel + the DE top-50 lists)."""
    m0 = pd.read_csv(f"{PANELS}/topN_{lin}_recast_M0.csv").sort_values(["cluster", "rank"])
    out = {"recast_M0": {str(c): list(g.gene.astype(str)) for c, g in m0.groupby("cluster")}}
    de = pd.read_csv(f"{GS50}/{lin}.csv").sort_values(["method", "cluster", "rank"])
    for m, gm in de.groupby("method"):
        out[str(m)] = {str(c): list(g.gene.astype(str)) for c, g in gm.groupby("cluster")}
    return out


def walk(lineages, sup, spell, go, sibling=True):
    """Full-pool walk in the order of the rules; per unit the step counts and the final (top 20) list."""
    S = lambda s, g: sup(V.EV[s][g]) if g in V.EV.get(s, {}) else 0
    res = {}
    for lin, states in lineages.items():
        for s in states:
            cand = [g for g in V.EV.get(s, {}) if S(s, g) >= B.MIN_PMID]
            if len(cand) < B.MIN_GENES:
                continue
            n_out = n_sh = 0
            surv = []
            for g in sorted(cand, key=lambda g: -S(s, g)):               # stable: CellMarker file order in ties
                e, n = spell(g), S(s, g)
                sib = [S(t, g) for t in states if t != s and S(t, g) >= B.MIN_PMID]
                if e not in go:
                    n_out += 1
                elif sibling and sib and max(sib) >= n:
                    n_sh += 1
                else:
                    surv.append(e)
            if surv:
                res[f"{lin}|{s}"] = dict(candidates=len(cand), outside=n_out, shared=n_sh,
                                         beyond=max(0, len(surv) - B.TOPK), final=surv[:B.TOPK])
    return res


def hits(L, unit, gt, k=20):
    lin, s = unit.split("|", 1)
    return len(set(list(pick(L[lin]["recast_M0"], s))[:k]) & gt), len(set(list(pick(L[lin][W], s))[:k]) & gt)


def board(L, gts, field):
    """gts {unit: set}: RECAST mean, rank and best other per metric over units with a nonempty panel."""
    rows = []
    for u, gt in gts.items():
        if not gt:
            continue
        lin, s = u.split("|", 1)
        for m in field:
            mt = metrics(list(pick(L[lin][m], s)), gt)
            rows += [dict(unit=u, method=m, metric=k, value=v) for k, v in mt.items()]
    D = pd.DataFrame(rows)
    out = {}
    for key, *_ in METRICS:
        mean = D[D.metric == key].groupby("method").value.mean()
        oth = mean.drop("recast_M0").sort_values(ascending=False, kind="stable")
        out[key] = dict(n=int(D[(D.metric == key) & (D.method == "recast_M0")].unit.nunique()),
                        recast=float(mean["recast_M0"]), rank=int((mean > mean["recast_M0"]).sum()) + 1,
                        best=oth.index[0], best_value=float(oth.iloc[0]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    V.EV = V.read_evidence()
    spell, go = B.encoder_spelling(), set(l.strip() for l in open(B.GO) if l.strip())
    V.SPELL, V.GO = spell, go
    sup = V.support_fn()
    frozen = json.load(open(f"{FULL_BENCHMARK}/ground_truth/panels/marker_panels.json"))
    flow = json.load(open(f"{FULL_BENCHMARK}/ground_truth/counts/construction_steps.json"))
    bench = sorted({u.split("|")[0] for u in frozen})
    lin_states = {lin: sorted(pd.read_csv(f"{PANELS}/topN_{lin}_recast_M0.csv").cluster.astype(str).unique())
                  for lin in bench + [BARON]}
    fine = {l: lin_states[l] for l in bench}

    # ---- the walk reproduces the stored panels and the construction counts; Baron through the same builder
    wf = walk(fine, sup, spell, go)
    assert {u: d["final"] for u, d in wf.items()} == frozen
    tot = lambda w, k: sum(d[k] for d in w.values())
    assert (tot(wf, "candidates"), tot(wf, "outside"), tot(wf, "shared"), tot(wf, "beyond")) == \
        (flow["pool0"], flow["n_vocab"], flow["n_shared"], flow["n_cut"])
    wb = walk({BARON: lin_states[BARON]}, sup, spell, go)
    built, _ = V.build({BARON: lin_states[BARON]})
    assert {u: d["final"] for u, d in wb.items()} == built, "Baron walk != panel_rules.build"
    unf = {u: set(d["final"]) for u, d in walk(fine, sup, spell, go, sibling=False).items()}
    assert set(unf) == set(frozen)

    L = {lin: lists(lin) for lin in lin_states}
    field = ["recast_M0"] + sorted(set(L[bench[0]]) - OUT_OF_FIELD - {"recast_M0"})
    assert len(field) == 51 and all(set(field) <= set(L[l]) for l in L), len(field)

    # ---- check: recall@20 per unit equals the benchmark table
    bt = pd.read_csv(f"{FULL_BENCHMARK}/scoring/out/recall_by_subtype.csv").set_index(["unit", "method"]).recall
    for u, gt in frozen.items():
        hr, hw = hits(L, u, set(gt))
        assert abs(hr / len(gt) - bt[(u, "recast_M0")]) < 1e-12 and abs(hw / len(gt) - bt[(u, W)]) < 1e-12, u
    print("check: recall@20 per unit equals the benchmark table (RECAST, Wilcoxon), 97 units")

    # ---- outputs
    rows = []
    for cohort, w in (("fine_subtype_benchmark", wf), ("pancreatic_control", wb)):
        for u, d in sorted(w.items()):
            hr, hw = hits(L, u, set(d["final"]))
            n = len(d["final"])
            rows.append(dict(Lineage=u.split("|")[0], Subtype=u.split("|", 1)[1], Candidates=d["candidates"],
                             Outside=d["outside"], Shared=d["shared"], Beyond_top20=d["beyond"], Final=n,
                             RECAST=f"{hr}/{n} ({hr / n:.2f})", Wilcoxon=f"{hw}/{n} ({hw / n:.2f})", cohort=cohort,
                             RECAST_hits=hr, RECAST_recovery=hr / n, Wilcoxon_hits=hw, Wilcoxon_recovery=hw / n))
    BY_UNIT = pd.DataFrame(rows)
    BY_UNIT.to_csv(f"{OUT}/panel_construction_by_unit.csv", index=False, encoding="utf-8-sig")
    counts_by_cohort = {}
    for cohort, a in BY_UNIT.groupby("cohort", sort=False):
        counts_by_cohort[cohort] = {"Dataset–subtype units": f"{len(a):,}", "Candidate markers": f"{a.Candidates.sum():,}",
                        "Removed outside the evaluation vocabulary": f"{a.Outside.sum():,}",
                        "Removed because shared with sibling panels": f"{a.Shared.sum():,}",
                        "Removed beyond the 20 best-supported genes": f"{a.Beyond_top20.sum():,}",
                        "Total markers in final panels": f"{a.Final.sum():,}",
                        "Final panel size, median (range)": f"{np.median(a.Final):g} ({a.Final.min()}–{a.Final.max()})",
                        "Panels containing fewer than eight genes": f"{int((a.Final < 8).sum())}",
                        "Markers recovered by RECAST": f"{a.RECAST_hits.sum():,}",
                        "Markers recovered by Wilcoxon": f"{a.Wilcoxon_hits.sum():,}",
                        "Mean marker recovery, RECAST": f"{a.RECAST_recovery.mean():.3f}",
                        "Mean marker recovery, Wilcoxon": f"{a.Wilcoxon_recovery.mean():.3f}"}
    f2 = BY_UNIT[BY_UNIT.cohort == "fine_subtype_benchmark"]
    assert len(f2) == 97 and f2.Final.sum() == flow["n_final"]

    b_f = board(L, {u: set(g) for u, g in frozen.items()}, field)
    b_u = board(L, unf, field)
    names = NAME
    new = []
    for variant, bo, sib in (("sibling-unique", b_f, "removed"), ("plain", b_u, "retained")):
        for i, (key, dname, ep, k) in enumerate(METRICS):
            new.append({"Ground truth": "CellMarker 3.0", "Panels": variant, "Metric": dname, "n": bo[key]["n"],
                        "RECAST": round(bo[key]["recast"], 3), "Rank": bo[key]["rank"],
                        "Best other method": names.get(bo[key]["best"], bo[key]["best"]),
                        "Value": round(bo[key]["best_value"], 3), "display_resource": "Human-cell-marker export",
                        "shared_sibling_markers": sib, "endpoint": ep, "selected_genes": k})
    METRICS_TABLE = pd.DataFrame(new)
    METRICS_TABLE.to_csv(f"{OUT}/alternative_metrics_cellmarker.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    res = dict(counts_by_cohort=counts_by_cohort, metrics={"sibling_unique": b_f, "plain": b_u}, unmapped_best=sorted(
        {bo[k]["best"] for bo in (b_f, b_u) for k in bo} - set(names)))
    json.dump(res, open(f"{OUT}/panel_construction_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
