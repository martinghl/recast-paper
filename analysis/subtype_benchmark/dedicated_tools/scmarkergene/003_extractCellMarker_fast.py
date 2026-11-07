#!/usr/bin/env python
"""Vectorised data layer for scMarkerGene's 003_extractCellMarker.py.

The original `collect_contribution_data` builds `gene_model_data[gene][model] -> list of floats` and
`cell_model_data[cell] -> list of (n_genes,) arrays`, an n_models x n_cells x n_genes pile of Python objects.  Its two
consumers need less: `build_cell_gene_matrix` reduces it to the per-cell mean over models, and
`save_aggregated_contributions` writes the same mean (`gene_model_arrays` is only used through len()).

This module computes those two products directly from the float32 attribution arrays that the modified 002 writes,
then calls the original statistics and selection code (`compute_fdr_by_permutation`, `extractPotentialGenes`,
`extractMarkerGenes`, imported from 003_extractCellMarker.py) with the arguments of the original main block.

The accept rule is kept: a cell contributes to the mean only from the models that predicted its label correctly (the
`if not predList[cell_index]: continue` gate), and a cell with no accepting model drops out.
`readAggregatedExplanation` is replaced for the same reason (it built one dict per cell with one key per gene); the
DataFrame it returns has the same columns, dtypes and row order."""
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd


def infoLine(message, infoType="info"):
    import datetime
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType = infoType + " "
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[" + infoType + " " + str(t) + "] " + message, flush=True)


def load_authors_module(src_path):
    spec = importlib.util.spec_from_file_location("scmg003", src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scmg003"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ vectorised replacements
def collect_and_average(inDIR, A):
    """Per-cell mean over the models that classified that cell correctly.

    Returns X (n_kept x n_genes), the kept cell ids, and cellId -> group name.
    """
    cellIdList = A.readCellSample(inDIR)              # sampleList.pool.dat, pool.dat row order
    gene_names = A.readGeneList(inDIR)
    codeHash = A.readCellCode(inDIR)                  # code -> group name
    exp_dir = inDIR + "/002_geneContribution_output/explain_dir"
    model_files = sorted(f for f in os.listdir(exp_dir) if f.endswith("_explanation.npy"))
    n_models = len(model_files)
    n_cells, n_genes = len(cellIdList), len(gene_names)

    total = np.zeros((n_cells, n_genes), dtype=np.float64)
    count = np.zeros(n_cells, dtype=np.int64)
    first = np.full(n_cells, n_models, dtype=np.int64)   # first model that accepted this cell
    labels = None
    for model_index in range(n_models):
        predList = np.array(A.readPredictionInfo(inDIR, model_index), dtype=bool)
        att = np.load(f"{exp_dir}/model_{model_index}_explanation.npy")
        lab = np.load(f"{exp_dir}/model_{model_index}_labels.npy")
        assert att.shape == (n_cells, n_genes), (att.shape, n_cells, n_genes)
        if labels is None:
            labels = lab
        total[predList] += att[predList]
        count[predList] += 1
        first[predList & (first == n_models)] = model_index

    # `cell_model_data` is an insertion-ordered dict: a cell enters when the first model that
    # classifies it correctly reaches it, and each model walks the pool in file order.  Row order
    # feeds rng.permutation() inside the FDR, so it is reproduced rather than assumed irrelevant.
    keep = np.flatnonzero(count > 0)
    keep = keep[np.argsort(first[keep], kind="stable")]
    X = total[keep] / count[keep][:, None]
    kept_ids = [cellIdList[i] for i in keep]
    groups = {cellIdList[i]: codeHash.get(str(labels[i]), "unknown") for i in keep}
    return X, kept_ids, groups, gene_names, n_models


def write_aggregated(X, kept_ids, groups, output_dir):
    """save_aggregated_contributions(): cellId <TAB> group <TAB> 6-dp values joined by '|'."""
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/aggregated_explanation.dat", "wt") as fo:
        for cid, row in zip(kept_ids, X):
            fo.write(f"{cid}\t{groups.get(cid, 'unknown')}\t"
                     + "|".join(f"{k:.6f}" for k in row) + "\n")


def read_aggregated_fast(agg_file, geneList):
    """readAggregatedExplanation(), without the per-cell dict comprehension."""
    ids, grps, rows = [], [], []
    with open(agg_file, "rt") as fi:
        for line in fi:
            r = line.rstrip("\n").split("\t")
            if len(r) < 3:
                continue
            ids.append(r[0]); grps.append(r[1])
            rows.append(np.fromstring(r[2], sep="|")[:len(geneList)])
    df = pd.DataFrame(np.vstack(rows), columns=list(geneList))
    df.insert(0, "group", grps)
    df.insert(0, "cell_id", ids)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", dest="inDIR", required=True)
    ap.add_argument("-o", dest="outDIR", required=True)
    ap.add_argument("-e", dest="exp_data", required=True,
                    help="the original dense expression TSV, as the original 003 takes")
    ap.add_argument("-p", dest="n_perm", type=int, default=1000)
    ap.add_argument("-n", dest="marker_num", type=int, default=20)
    ap.add_argument("-t", dest="TopN", type=int, default=50)
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "scmarkergene_src",
                                                  "003_extractCellMarker.py"))
    a = ap.parse_args()
    A = load_authors_module(os.path.abspath(a.src))
    # the original extractPotentialGenes calls readAggregatedExplanation on the module namespace
    A.readAggregatedExplanation = read_aggregated_fast

    infoLine("Aggregating all models")
    X, kept_ids, groups, gene_names, n_models = collect_and_average(a.inDIR, A)
    infoLine(f"{X.shape[0]} of {X.shape[0]} kept cells x {X.shape[1]} genes over {n_models} models")

    gene_stats = A.compute_fdr_by_permutation(
        X=X, cell_ids=kept_ids, cellId2groupHash=groups,
        gene_names=gene_names, n_perm=a.n_perm)

    output_dir = f"{a.outDIR}/aggregatedData"
    A.save_statistical_results(gene_stats, output_dir)
    write_aggregated(X, kept_ids, groups, output_dir)
    significant_gene_names = gene_stats.loc[gene_stats["significant"], "gene"].unique().tolist()
    infoLine(f"Analysis completed! Found {len(significant_gene_names)} significant genes (FDR<0.05) "
             f"out of {len(gene_names)} total genes")

    infoLine("Extract candidate genes")
    candidate_genes = A.extractPotentialGenes(a.inDIR, a.outDIR, a.TopN, significant_gene_names)

    infoLine("Select marker genes")
    A.extractMarkerGenes(a.exp_data, candidate_genes, a.outDIR, a.marker_num)
    infoLine("Done!")


if __name__ == "__main__":
    main()
