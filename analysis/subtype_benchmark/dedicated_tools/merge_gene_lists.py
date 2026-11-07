#!/usr/bin/env python
"""Merge the dedicated tools' gene lists into copies of the comparator gene-list directories.

Writes fs_genesets_v2_plus/ (depth 50: recall@20 and AUROC) and fs_genesets200_v2_plus/ (depth 200: marker rank and
contamination), each holding every row of the original file plus the new methods' rows; the original directories are
not written to.

Cluster names must match the originals, because the scorers look subtypes up by name.  The tools were given the same
<lin>_labels.txt as the harness, so the labels agree by construction; the check below would catch a mismatch.  R's
make.names renames some labels inside the harness ("Naive B cells" -> "Naive.B.cells"); the scorers use
cluster_key.pick for that, so a new method's plain label is matched the same way.

->  fs_genesets_v2_plus/<lin>.csv, fs_genesets200_v2_plus/<lin>.csv
->  dedicated_tools/geneset_merge_report.csv   per lineage x method: depth written, cluster match"""
import argparse, os, shutil, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FSB = os.path.dirname(HERE)
sys.path.insert(0, FSB)
from cluster_key import pick   # noqa: E402

LIN20 = ["fs_asc_zhao", "fs_gut_smillie", "fs_lung_hlca", "fs_nk_pbmcbmn", "fs_nk_zhao",
         "tf_b_ren", "tf_b_steph", "tf_b_yosh", "tf_cd4_dom", "tf_cd4_ren", "tf_cd4_steph",
         "tf_cd4_yosh", "tf_cd8_dom", "tf_cd8_ren", "tf_cd8_steph", "tf_cd8_yosh",
         "tf_mono_dom", "tf_mono_ren", "tf_mono_steph", "tf_mono_yosh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=f"{HERE}/genesets200", help="dedicated_tools per-lineage method CSVs")
    ap.add_argument("--src50", default=f"{FSB}/fs_genesets_v2")
    ap.add_argument("--src200", default=f"{FSB}/fs_genesets200_v2")
    ap.add_argument("--dst50", default=f"{FSB}/fs_genesets_v2_plus")
    ap.add_argument("--dst200", default=f"{FSB}/fs_genesets200_v2_plus")
    a = ap.parse_args()
    os.makedirs(a.dst50, exist_ok=True)
    os.makedirs(a.dst200, exist_ok=True)

    # The depth-50 directory holds every lineage the fine-state harness ever scored (41), not just
    # the 20 with canonical fine subtypes, and the recall producer walks all of them.  So the copy
    # has to be complete: every base lineage is carried across, and the dedicated_tools methods are appended
    # only to the 20 they were run on.  (The depth-200 directory is already the 20.)
    all50 = sorted(f[:-4] for f in os.listdir(a.src50) if f.endswith(".csv"))
    all200 = sorted(f[:-4] for f in os.listdir(a.src200) if f.endswith(".csv"))
    for lin in sorted(set(all50) - set(LIN20)):
        shutil.copyfile(f"{a.src50}/{lin}.csv", f"{a.dst50}/{lin}.csv")
    for lin in sorted(set(all200) - set(LIN20)):
        shutil.copyfile(f"{a.src200}/{lin}.csv", f"{a.dst200}/{lin}.csv")
    print(f"[copy] {len(set(all50) - set(LIN20))} depth-50 and "
          f"{len(set(all200) - set(LIN20))} depth-200 lineages carried across untouched")

    report = []
    for lin in LIN20:
        base50 = pd.read_csv(f"{a.src50}/{lin}.csv")
        base200 = pd.read_csv(f"{a.src200}/{lin}.csv")
        # the scorers group the base file by cluster, so its own labels are the reference set
        base_clusters = {str(c) for c in base50["cluster"].unique()}
        base_by_name = {c: c for c in base_clusters}

        add50, add200 = [], []
        for f in sorted(os.listdir(a.new)):
            if not f.startswith(f"{lin}__") or not f.endswith(".csv"):
                continue
            d = pd.read_csv(f"{a.new}/{f}")
            meth = str(d["method"].iloc[0])
            for c in d["cluster"].unique():
                hit = pick(base_by_name, str(c))
                report.append(dict(lineage=lin, method=meth, cluster=str(c),
                                   matched=str(hit) if hit else "",
                                   depth200=int((d.cluster == c).sum()),
                                   depth50=int(min(50, (d.cluster == c).sum()))))
            d = d.sort_values(["method", "cluster", "rank"])
            add200.append(d)
            add50.append(d[d["rank"] <= 50])

        if not add50:
            print(f"[skip] {lin}: no dedicated_tools methods yet -- base copied unchanged")
            shutil.copyfile(f"{a.src50}/{lin}.csv", f"{a.dst50}/{lin}.csv")
            shutil.copyfile(f"{a.src200}/{lin}.csv", f"{a.dst200}/{lin}.csv")
            continue
        out50 = pd.concat([base50] + add50, ignore_index=True)
        out200 = pd.concat([base200] + add200, ignore_index=True)
        out50.to_csv(f"{a.dst50}/{lin}.csv", index=False)
        out200.to_csv(f"{a.dst200}/{lin}.csv", index=False)
        n_new = out50.method.nunique() - base50.method.nunique()
        print(f"[ok  ] {lin}: {base50.method.nunique()} + {n_new} methods -> "
              f"{out50.method.nunique()}  ({len(out50)} / {len(out200)} rows)")

    rep = pd.DataFrame(report)
    rep.to_csv(f"{HERE}/geneset_merge_report.csv", index=False)
    bad = rep[rep.matched == ""]
    print(f"\nmerge report: {len(rep)} lineage x method x cluster rows, "
          f"{len(bad)} with no matching cluster in the base file")
    if len(bad):
        print(bad.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
