"""Integer-count matrices of the added lineages for the comparator methods, written with the functions of
subtype_benchmark/export_lineage_counts.py (real_counts / subset / write_ds) and the added-lineage paths and lineage
table: inputs from full_benchmark/data/atlases, outputs to full_benchmark/data/sce_export_v2.  Same cells (attribution
h5ad order), same >= min_cells subtype rule, same SCimilarity gene space, genes detected in >= 3 cells, same file
layout.  export_lineage_counts imports extra_lineages only for its lineage table (which loads the encoder), so that
import is replaced by an empty table.

Env SC.  Usage: python -I export_added_counts.py [dsid ...]   (default: all added lineages)"""
import os
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd

FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
sys.path.insert(0, FSB)
sys.path.insert(0, f"{FULL_BENCHMARK}/added_lineages")
sys.modules["extra_lineages"] = types.SimpleNamespace(LINEAGES={})
import export_lineage_counts as R  # noqa: E402
import added_datasets as D  # noqa: E402

R.ATLASES = f"{FULL_BENCHMARK}/data/atlases"
R.OUT = f"{FULL_BENCHMARK}/data/sce_export_v2"


def main():
    want = sys.argv[1:] or D.NEW_IDS
    by_stem = {}
    for dsid in want:
        by_stem.setdefault(D.LINEAGES[dsid]["ds"], []).append(dsid)
    rows = []
    for stem, ids in sorted(by_stem.items()):
        print(f"\n=== stem {stem} ({len(ids)} lineages) ===", flush=True)
        C = R.real_counts(stem)
        A = ad.read_h5ad(f"{R.ATLASES}/{stem}_attribution_scimilarity.h5ad", backed="r")
        att_lab = A.obs["label"].astype(str).to_numpy()
        A.file.close()
        for dsid in ids:
            e = D.LINEAGES[dsid]
            lab = np.array([l.strip() for l in open(f"{R.ATLASES}/{e['label_file']}")])
            assert (lab == att_lab).all(), f"{dsid}: label file disagrees with the attribution h5ad"
            Csub, labs = R.subset("labelfile", C, stem, e["label_file"], e["states"], e["min_cells"])
            dropped = [s for s in e["states"] if s not in set(labs)]
            print(f"  [{dsid}] states kept {sorted(set(labs))}; dropped (< {e['min_cells']} cells) {dropped}", flush=True)
            rows.append(R.write_ds(dsid, Csub, labs))
    df = pd.DataFrame(rows)
    os.makedirs(R.OUT, exist_ok=True)
    df.to_csv(f"{R.OUT}/rebuild_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
