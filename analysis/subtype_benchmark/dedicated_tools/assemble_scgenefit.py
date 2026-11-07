#!/usr/bin/env python
"""Merge the per-subtype scGeneFit parts into one gene-list file and one timing file per lineage.

run_scgenefit.py runs one subtype per process so each can be held to one thread, the condition under which the other
methods' selection times were measured.  The per-lineage time is the sum over the lineage's subtypes, i.e. what a
single-threaded run of the whole lineage costs.

Also reports whether the top-20 lists are in alphabetical order and how much they overlap between subtypes.
Alphabetical order is expected (scGeneFit returns a set, so there is no order within the first 20), but the overlap
between subtypes should be low, because each subtype gets its own 20-gene solution.

->  genesets200/<lin>__scgenefit.csv, timing/<lin>__scgenefit.json"""
import argparse, glob, json, os
import numpy as np, pandas as pd

LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="/path/to/cpu_server/dedicated_tools/sgf_parts")
    ap.add_argument("--sce", default="/path/to/cpu_server/dedicated_tools/sce_export_v2")
    ap.add_argument("--out", default="/path/to/cpu_server/dedicated_tools/genesets200")
    ap.add_argument("--timing", default="/path/to/cpu_server/dedicated_tools/timing")
    ap.add_argument("--allow-partial", action="store_true",
                    help="emit a lineage even if some subtypes are missing; the missing subtype "
                         "names and a lower bound on their cost are recorded in the timing json, "
                         "and fill_unfinished.py scores them as the worst value of any method")
    ap.add_argument("--stall-seconds", type=float, default=0.0,
                    help="elapsed single-thread seconds each missing subtype had already consumed "
                         "when the run was frozen; a lower bound, not a measurement")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.timing, exist_ok=True)

    print(f"{'lineage':16s} {'subtypes':>8s} {'depth':>9s} {'seconds':>8s} {'alpha':>6s} {'overlap':>7s}")
    for lin in LIN20:
        js = sorted(glob.glob(f"{a.part}/{lin}__*.json"))
        labs = sorted(set(l.strip() for l in open(f"{a.sce}/{lin}_labels.txt") if l.strip()))
        want = len(labs)
        have_idx = {int(os.path.basename(f).split("__")[1][:-5]) for f in js}
        missing = [labs[i] for i in range(want) if i not in have_idx]
        if missing and not a.allow_partial:
            print(f"{lin:16s} {len(js):3d}/{want:<4d} INCOMPLETE -- skipped")
            continue
        meta = [json.load(open(f)) for f in js]
        parts = [pd.read_csv(f.replace(".json", ".csv")) for f in js]
        d = pd.concat(parts, ignore_index=True).sort_values(["cluster", "rank"])
        d.to_csv(f"{a.out}/{lin}__scgenefit.csv", index=False)

        tops = {c: list(g.sort_values("rank").gene[:20]) for c, g in d.groupby("cluster")}
        ks = list(tops)
        alpha = float(np.mean([tops[k] == sorted(tops[k]) for k in ks]))
        ov = [len(set(tops[p]) & set(tops[q])) / len(set(tops[p]) | set(tops[q]))
              for i, p in enumerate(ks) for q in ks[i + 1:]]
        depths = {m["cluster"]: m["depth"] for m in meta}
        secs = float(sum(m["seconds"] for m in meta))
        json.dump({"method": "scgenefit", "lineage": lin, "seconds": secs,
                   "seconds_budget20": float(sum(m["seconds_budget20"] for m in meta)),
                   "seconds_budget200": float(sum(m["seconds_budget200"] for m in meta)),
                   "n_cells": meta[0]["n_cells"], "n_genes": meta[0]["n_genes"],
                   "n_clusters": len(meta), "n_clusters_expected": want,
                   "missing_clusters": missing,
                   "seconds_missing_lower_bound": float(a.stall_seconds) * len(missing),
                   "min_depth": min(depths.values()),
                   "max_depth": max(depths.values()), "depth_per_cluster": depths,
                   "threads": 1, "scgenefit_method": meta[0]["scgenefit_method"],
                   "epsilon": meta[0]["epsilon"],
                   "alphabetical_top20_fraction": alpha,
                   "mean_top20_overlap_between_subtypes": float(np.mean(ov)) if ov else None,
                   "protocol": "ranks 1-20 = get_markers(num_markers=20); ranks 21-200 = "
                               "get_markers(num_markers=200) minus those 20. scGeneFit's LP returns "
                               "tied unit weights, so it yields a set of the requested size and no "
                               "order within it; each subtype was run separately on one thread so "
                               "the time is on the same boundary as the other methods' selection times.",
                   }, open(f"{a.timing}/{lin}__scgenefit.json", "w"), indent=1)
        flag = f"  PARTIAL {len(meta)}/{want}, missing {missing}" if missing else ""
        print(f"{lin:16s} {len(meta):8d} {min(depths.values()):4d}-{max(depths.values()):<4d} "
              f"{secs:8.0f} {alpha:6.2f} {np.mean(ov) if ov else float('nan'):7.3f}{flag}")


if __name__ == "__main__":
    main()
