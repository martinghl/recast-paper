"""Step-by-step counts of the marker-panel construction (Supplementary Data sheet 12; construction schematic).

Walks the rules of panel_rules.py in their fixed order without the builder's stop at the 20th kept gene, so every
filter is counted on the full candidate pool; cutting each unit to its first 20 survivors gives the panels of
panels/marker_panels.json (asserted).  Record-level counts use the row filter of panel_rules.read_evidence.

Writes counts/construction_steps.csv (one row per step: level, criterion, n_in, n_removed, n_out, note) and
counts/construction_steps.json (the same numbers, keyed for paper/ground_truth_tables/panel_construction.py).
Env analysis.  Usage: python -I counts/construction_counts.py
"""
import collections
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.dirname(HERE)
sys.path.insert(0, GT)
import candidate_markers as B  # noqa: E402
import panel_rules as V  # noqa: E402

TISSUE = [("_lung_", "lung"), ("_gut_", "gut"), ("_glia_", "brain"), ("_neuron_", "brain")]


def tissue(lin):
    return next((t for k, t in TISSUE if k in lin), "blood")


def main():
    lineages = {}
    for f in sorted(glob.glob(f"{V.PANELS}/topN_*_recast_M0.csv")):
        lin = f.split("topN_")[1].split("_recast_M0")[0]
        states = sorted(pd.read_csv(f).cluster.astype(str).unique())
        if lin in V.SKIP or not any(B.XW.get(s) for s in states):
            continue
        lineages[lin] = states
    bench = sorted({s for ss in lineages.values() for s in ss})
    mapped = [s for s in bench if B.XW.get(s)]
    names = {s: {n.lower() for n in B.XW[s]} for s in mapped}
    allnames = set().union(*names.values())

    # ---- record level (the reader's filter, in its order)
    r = collections.Counter()
    nkt_matched = 0
    with open(B.DB, encoding="utf-8", errors="ignore") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            r["all"] += 1
            if len(f) < 20:
                continue
            r["fields"] += 1
            if f[0] != "Human":
                continue
            r["human"] += 1
            if f[4] != "Normal":
                continue
            r["normal"] += 1
            cn, sym = f[6].lower(), f[9].strip()
            if not sym or sym == "NA":
                continue
            r["symbol"] += 1
            if "natural killer t" in cn or "nkt" in cn:
                nkt_matched += cn in allnames
                continue
            r["no_nkt"] += 1
            r["matched"] += cn in allnames
    assert nkt_matched == 0

    # ---- subtype x gene evidence (dataset-independent)
    ev = V.read_evidence()
    sup = V.support_fn()
    lit = lambda d: {p for p in d["lit"] if p}
    pairs = [(s, g) for s in mapped for g in ev.get(s, {})]
    no_author = sum(1 for s, g in pairs if not lit(ev[s][g]))
    one_pub = sum(1 for s, g in pairs if lit(ev[s][g]) and sup(ev[s][g]) < B.MIN_PMID)
    cand = {s: [g for g in ev.get(s, {}) if sup(ev[s][g]) >= B.MIN_PMID] for s in mapped}
    n_cand = sum(len(v) for v in cand.values())
    assert len(pairs) - no_author - one_pub == n_cand
    kept_states = [s for s in mapped if len(cand[s]) >= B.MIN_GENES]
    small_states = [s for s in mapped if len(cand[s]) < B.MIN_GENES]

    # ---- units: a subtype inside one dataset lineage; siblings = every state of that lineage's panel
    S = lambda s, g: sup(ev[s][g]) if g in ev.get(s, {}) else 0
    spell = B.encoder_spelling()
    go = set(l.strip() for l in open(B.GO) if l.strip())
    units = [(lin, s) for lin, ss in lineages.items() for s in ss if s in kept_states]
    pool0 = sum(len(cand[s]) for _, s in units)
    n_vocab = n_shared = n_cut = 0
    final = {}
    for lin, s in units:
        survivors = []
        for g in sorted(cand[s], key=lambda g: -S(s, g)):                  # stable: CellMarker file order in ties
            e, n = spell(g), S(s, g)
            sib = [S(t, g) for t in lineages[lin] if t != s and S(t, g) >= B.MIN_PMID]
            if e not in go:
                n_vocab += 1
            elif sib and max(sib) >= n:
                n_shared += 1
            else:
                survivors.append(e)
        n_cut += max(0, len(survivors) - B.TOPK)
        if survivors:
            final[f"{lin}|{s}"] = survivors[:B.TOPK]
    built = json.load(open(f"{GT}/panels/marker_panels.json"))
    assert final == built, "full-pool walk does not give the panels of build_panels.py"
    n_final = sum(len(v) for v in final.values())
    fin_states = sorted({u.split("|")[1] for u in final})
    fin_lins = sorted({u.split("|")[0] for u in final})
    by_tissue = collections.Counter(tissue(u.split("|")[0]) for u in final)
    sub_by_tissue = {t: len({u.split("|")[1] for u in final if tissue(u.split("|")[0]) == t}) for t in by_tissue}
    per_unit = pd.Series({u: len(v) for u, v in final.items()})

    steps = [
        ("records", "CellMarker 3.0 records", None, None, r["all"], ""),
        ("records", "Human", r["all"], r["all"] - r["human"], r["human"],
         f"{r['all'] - r['fields']} malformed lines inside the non-human count" if r["all"] != r["fields"] else ""),
        ("records", "Normal tissue", r["human"], r["human"] - r["normal"], r["normal"], ""),
        ("records", "Gene symbol given", r["normal"], r["normal"] - r["symbol"], r["symbol"], ""),
        ("records", "NKT cell names skipped", r["symbol"], r["symbol"] - r["no_nkt"], r["no_nkt"],
         "none of these names is in the name table"),
        ("records", f"Cell name in the name table ({len(allnames)} CellMarker names for {len(mapped)} of "
                    f"{len(bench)} panel subtypes)", r["no_nkt"], r["no_nkt"] - r["matched"], r["matched"], ""),
        ("subtype x gene", "Subtype-gene pairs in the matched records", None, None, len(pairs), ""),
        ("subtype x gene", ">= 2 publications (distinct PMIDs), >= 1 reported by the authors",
         len(pairs), no_author + one_pub, n_cand,
         f"{no_author} with CellMarker-computed (Method) records only; {one_pub} with one publication"),
        ("subtypes", ">= 8 candidate genes", len(mapped), len(small_states), len(kept_states),
         "dropped: " + "; ".join(f"{s} ({len(cand[s])})" for s in small_states)),
        ("unit x gene", f"Units = subtype x dataset lineage ({len(units)} units in {len(lineages)} lineages)",
         None, None, pool0, ""),
        ("unit x gene", "Gene in the encoder vocabulary", pool0, n_vocab, pool0 - n_vocab, ""),
        ("unit x gene", "Not supported equally or more by a sibling subtype of the same lineage",
         pool0 - n_vocab, n_shared, pool0 - n_vocab - n_shared, ""),
        ("unit x gene", "Top 20 by publication count", pool0 - n_vocab - n_shared, n_cut, n_final,
         f"{len(units) - len(final)} units left without genes"),
        ("final", f"{len(final)} units, {len(fin_states)} subtypes, {len(fin_lins)} lineages", None, None, n_final,
         "units by tissue: " + ", ".join(f"{t} {by_tissue[t]} ({sub_by_tissue[t]} subtypes)"
                                         for t in ("blood", "lung", "gut", "brain"))
         + f"; genes per unit median {per_unit.median():g}, min {per_unit.min()}"),
    ]
    t = pd.DataFrame(steps, columns=["level", "criterion", "n_in", "n_removed", "n_out", "note"])
    t.to_csv(f"{HERE}/construction_steps.csv", index=False)
    out = dict(records=dict(r), nkt_matched=nkt_matched, n_names=len(allnames), n_bench_states=len(bench),
               n_mapped=len(mapped), pairs=len(pairs), no_author=no_author, one_pub=one_pub, n_cand=n_cand,
               n_kept_states=len(kept_states), small_states={s: len(cand[s]) for s in small_states},
               n_lineages_panels=len(lineages), n_units=len(units), pool0=pool0, n_vocab=n_vocab,
               n_shared=n_shared, n_cut=n_cut, n_final=n_final, final_units=len(final),
               final_subtypes=len(fin_states), final_lineages=len(fin_lins), units_no_gene=len(units) - len(final),
               by_tissue=dict(by_tissue), subtypes_by_tissue=sub_by_tissue,
               genes_per_unit_median=float(per_unit.median()), genes_per_unit_min=int(per_unit.min()))
    json.dump(out, open(f"{HERE}/construction_steps.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
