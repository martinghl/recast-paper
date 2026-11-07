"""Rules of the subtype marker panels (Supplementary Data sheets 12-14), shared by build_panels.py,
counts/construction_counts.py and paper/ground_truth_tables/panel_construction.py.

Source: CellMarker 3.0 (human_cell_marker_new.txt), rows with species Human and disease Normal, matched to subtypes by
the exact cell-name lists of candidate_markers.XW; NKT names skipped; symbols upper-cased, then written in the
encoder's spelling.
  1. Contrast: a lineage's siblings are all subtype labels of its RECAST panel (scoring/in/panels), with or without a
     marker panel of their own; FAMILY gives the compartment.
  2. Evidence: support = distinct non-empty PMIDs; a candidate needs support >= 2 including at least one PMID whose
     record is not CellMarker's computational "Method" source (support_fn).
  3. Shared: a candidate that a sibling also supports (rule 2) stays only with the subtype whose support is strictly
     highest; a tie removes it from all.
  4. The candidate pool is filtered (encoder vocabulary, rule 3) and cut to the top 20 by support; ties keep
     CellMarker file order.
  5. A subtype gets a panel only if >= 8 candidates pass rule 2 (before rules 3 and 4); it is scored on what survives.
read_evidence() reads the evidence and build() applies the rules.
"""
import collections
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidate_markers as B  # noqa: E402

FULL_BENCHMARK = os.path.dirname(HERE)
PANELS = f"{FULL_BENCHMARK}/scoring/in/panels"
FAMILY = [("tf_cd8_", "CD8 T"), ("tf_cd4_", "CD4 T"), ("tf_b_", "B"), ("tf_mono_", "monocyte"), ("_nk_", "NK"),
          ("_asc_", "ASC"), ("_lung_", "lung epithelium"), ("_gut_", "gut epithelium"), ("_glia_", "glia"), ("_neuron_", "neuron")]
SKIP = {"fs_panc_baron"}                       # the endocrine control is not part of this benchmark


def family(lin):
    hits = [f for k, f in FAMILY if k in lin]
    return hits[0] if len(hits) == 1 else None


def read_evidence():
    """subtype -> gene -> {lit, met: PMIDs; src: marker_sources}, genes in first-appearance order."""
    names = {s: {n.lower() for n in v} for s, v in B.XW.items()}
    ev = collections.defaultdict(dict)
    with open(B.DB, encoding="utf-8", errors="ignore") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 20 or f[0] != "Human" or f[4] != "Normal":
                continue
            cn, sym, pmid = f[6].lower(), f[9].strip(), f[16].strip()
            if not sym or sym == "NA" or "natural killer t" in cn or "nkt" in cn:
                continue
            for s, ns in names.items():
                if cn in ns:
                    d = ev[s].setdefault(sym.upper(), {"lit": set(), "met": set(), "src": set()})
                    d["met" if f[15] == "Method" else "lit"].add(pmid)
                    d["src"].add(f[15])
    return ev


def support_fn():
    """support of one evidence record: distinct PMIDs, or 0 when none of them is a literature (non-Method) PMID."""
    def sup(d):
        lit = {p for p in d["lit"] if p}
        return len(lit | {p for p in d["met"] if p}) if lit else 0
    return sup


def fmt(pmids):
    return "; ".join(sorted(p[:-2] if p.endswith(".0") else p for p in pmids if p))


def build(lineages):
    """{lineage: [subtypes]} -> ({lineage|subtype: [genes]}, one row per candidate scanned with its status).
    Reads the module globals EV (read_evidence()), SPELL (candidate_markers.encoder_spelling()) and GO (the encoder
    gene list), which the caller sets."""
    sup = support_fn()
    S = lambda s, g: sup(EV[s][g]) if g in EV.get(s, {}) else 0
    units, rows = {}, []
    for lin, states in lineages.items():
        fam = family(lin)
        for s in states:
            pool = sorted((g for g in EV.get(s, {}) if S(s, g) >= B.MIN_PMID), key=lambda g: -S(s, g))  # stable
            if len(pool) < B.MIN_GENES:
                continue                         # rule 5 (a subtype without names has no pool)
            keep = []
            for i, g in enumerate(pool, 1):
                e, n = SPELL(g), S(s, g)
                sib = {t: S(t, g) for t in states if t != s and S(t, g) >= B.MIN_PMID}
                if e not in GO:
                    st = "dropped: not in encoder vocabulary"
                elif sib and max(sib.values()) >= n:
                    t = max(sib, key=sib.get)
                    st = f"dropped: shared with {t} ({n} vs {sib[t]})"
                else:
                    st = "in GT"
                    keep.append(e)
                d = EV[s][g]
                rows.append(dict(compartment=fam, lineage=lin, subtype=s, pool_rank=i, gene=e, gene_cellmarker=g,
                                 support=n, n_literature=len({p for p in d["lit"] if p}),
                                 pmids_literature=fmt(d["lit"]), pmids_method_only=fmt(d["met"] - d["lit"]),
                                 marker_sources="; ".join(sorted(d["src"])),
                                 sibling_support="; ".join(f"{t}={v}" for t, v in sorted(sib.items(), key=lambda kv: -kv[1])),
                                 status=st, kept_rank=len(keep) if st == "in GT" else ""))
                if len(keep) == B.TOPK:
                    break
            assert len(set(keep)) == len(keep), (lin, s)
            if keep:
                units[f"{lin}|{s}"] = keep
    return units, pd.DataFrame(rows)
