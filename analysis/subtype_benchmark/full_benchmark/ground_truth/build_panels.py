"""Build the subtype marker panels of the benchmark (rules: panel_rules.py; inputs: candidate_markers.py).  The
only hand-made input is the subtype -> CellMarker cell-name table (candidate_markers.XW); every gene decision follows
from CellMarker by the rules.

Writes panels/: marker_panels.json {lineage|subtype: [genes]} (97 units), marker_panels.csv (every candidate scanned
until the 20th kept gene, with its status) and contrasts.csv (each lineage's siblings, with and without a panel).
Env analysis.  Usage: python -I build_panels.py
"""
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidate_markers as B  # noqa: E402
import panel_rules as V  # noqa: E402

OUT = f"{HERE}/panels"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lineages = {}
    for f in sorted(glob.glob(f"{V.PANELS}/topN_*_recast_M0.csv")):
        lin = f.split("topN_")[1].split("_recast_M0")[0]
        states = sorted(pd.read_csv(f).cluster.astype(str).unique())
        if lin in V.SKIP or not any(B.XW.get(s) for s in states):
            print(f"[skip] {lin}: {states}")
            continue
        assert V.family(lin), lin
        lineages[lin] = states
    V.EV = V.read_evidence()                     # build() reads EV / SPELL / GO from panel_rules
    V.SPELL = B.encoder_spelling()
    V.GO = set(l.strip() for l in open(B.GO) if l.strip())
    units, t = V.build(lineages)
    json.dump(units, open(f"{OUT}/marker_panels.json", "w"), indent=1, ensure_ascii=False)
    t.to_csv(f"{OUT}/marker_panels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([dict(lineage=l, compartment=V.family(l), siblings=", ".join(ss),
                       with_panel=", ".join(s for s in ss if f"{l}|{s}" in units),
                       without_panel=", ".join(s for s in ss if f"{l}|{s}" not in units))
                  for l, ss in lineages.items()]).to_csv(f"{OUT}/contrasts.csv", index=False, encoding="utf-8-sig")
    n = pd.Series({u: len(g) for u, g in units.items()})
    print(f"units {len(units)} | subtypes {len({u.split('|')[1] for u in units})} | genes per unit "
          f"median {n.median():g} min {n.min()} | unit x gene {n.sum()}")
    print(t.status.str.replace(r"shared with .*", "shared with a sibling", regex=True).value_counts().to_string())
