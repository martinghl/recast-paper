"""Fig. 2b and Fig. 5: the top-50 RECAST panels in subtype_benchmark/panels/topN_<lineage>_recast_M0.csv.

Fig. 2b ranks genes from the Zhao CD8 panel, and Fig. 5 tests the top 30 genes of 25 lineages for program
enrichment. Four of the lineages are Zhao atlas lineages (labels from the attribution object's obs['label'],
subtypes with at least 50 cells); they are recomputed here with recast.cluster_attribution. The other 21 are
benchmark lineages whose package panels check_benchmark_panels.py has already written (run it first); their top 50
are compared directly.

    python check_top50_panels.py
Writes top50_panels.csv (one row per subtype)."""
import os

import pandas as pd

import common as C
import recast

PANELS = os.path.join(C.FSB, "panels")
ZHAO = {
    "ext_zhao_CD8": ["CCR7+ CD8", "CRCR3+ CD8", "CX3CR1+ CD8", "CXCR6+ CD8"],
    "ext_zhao_CD4": ["CCR7+ CD4", "CXCR3+ CD4", "Treg"],
    "ext_zhao_B": ["Naive B", "TCL1A+ naive B", "Classical mem B", "CD11c+ mem B", "ZBTB32+ mem B", "MZB"],
    "ext_zhao_Mono": ["CD14+ Mo", "CD16+ Mo", "MerTK+ Mo"],
}


def main():
    rows = []
    enc = C.scimilarity()
    _, obs = C.attribution_obs(C.ATLASES, "zhao")
    labels = obs["label"].astype(str).to_numpy()
    for lin, states in ZHAO.items():
        a, kept = C.lineage_adata(C.ATLASES, "zhao", labels, states, min_cells=50)
        res = recast.cluster_attribution(enc, a, "state", reference="rest", gate="dC",
                                         centroid="mean_lognorm", qc="silent")
        stored = C.panel_lists(os.path.join(PANELS, f"topN_{lin}_recast_M0.csv"))
        assert sorted(stored) == sorted(kept), (lin, sorted(stored), kept)
        for s in kept:
            rows.append(dict(lineage=lin, state=s, source="recomputed",
                             same_set_top30=set(res.genes[s][:30]) == set(stored[s][:30]),
                             **C.compare_lists(res.genes[s], stored[s])))
        print(f"[{lin}] " + ", ".join(f"{r['state']}: {'identical' if r['identical'] else r['first_diff_rank']}"
                                      for r in rows if r["lineage"] == lin), flush=True)
        C.assert_package_only()
    for f in sorted(os.listdir(PANELS)):
        if not (f.startswith("topN_") and f.endswith("_recast_M0.csv")):
            continue
        lin = f[len("topN_"):-len("_recast_M0.csv")]
        if lin in ZHAO:
            continue
        pkg = C.panel_lists(os.path.join(C.OUT, "panels", f"topN200_{lin}_recast.csv"))
        stored = C.panel_lists(os.path.join(PANELS, f))
        assert sorted(stored) == sorted(pkg), (lin, sorted(stored), sorted(pkg))
        for s in stored:
            rows.append(dict(lineage=lin, state=s, source="check_benchmark_panels",
                             same_set_top30=set(pkg[s][:30]) == set(stored[s][:30]), **C.compare_lists(pkg[s], stored[s])))
    R = pd.DataFrame(rows)
    print(f"\n{R.lineage.nunique()} lineages, {len(R)} subtypes: top-50 lists identical {int(R.identical.sum())}/{len(R)}; "
          f"same top-30 set {int(R.same_set_top30.sum())}/{len(R)}, "
          f"same top-50 set {int(R.same_set_top50.sum())}/{len(R)}")
    if not R.identical.all():
        print(R[~R.identical][["lineage", "state", "first_diff_rank", "overlap_top50"]].to_string(index=False))
    C.write(R, "top50_panels.csv")


if __name__ == "__main__":
    main()
