#!/usr/bin/env python
"""Vectorised replacement for scMarkerGene's 001_prepareData.py.

The original holds the matrix as `expHash[cellId][geneName]`, a dict of dicts of Python floats; the original demo
(3,006 cells) fits, the benchmark's lineages (up to 16,737 cells x 18,603 genes) do not.

The method is unchanged.  Every rule follows the original, in the original order, and the outputs are the same files
with the same formatting:

  filterGenes      keep gene g if max over cells >= threshold AND some group has
                   percentile(g within that group, 80) > threshold   (threshold 0.01 for counts; for
                   'normalized' input, the 25th percentile of all non-zero values)
  reference        per gene, mean/std of log2(CP10K + 1) over all cells; 256 rows of
                   |N(mean, std)|, rounded to 3 dp, plus a 4th/5th-decimal jitter
  pool/train/test  log2(1 + CP10K) at 3 dp, "|"-joined; 90/10 split per group (or last 10 when the
                   group has <= 50 cells), shuffled with random.seed(42)

Two differences:
  * `--seed` seeds numpy before the reference draw (the original is unseeded, so its universal.dat differs on every
    run);
  * the input may be a Matrix Market triple (--mtx/--genes/--labels) instead of the dense TSV, which would be
    20-90 GB of text for these lineages.  Values are read identically either way.

`percentile` uses numpy's default linear interpolation, as the original."""
import argparse, datetime, gzip, os, random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def infoLine(message, infoType="info"):
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType = infoType + " "
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[" + infoType + " " + str(t) + "] " + message, flush=True)


# ------------------------------------------------------------------ input
def read_tsv(infile):
    """The original format: header 'cellId<TAB>group<TAB>gene1<TAB>...', one row per cell."""
    op = gzip.open if infile.endswith(".gz") else open
    with op(infile, "rt") as fi:
        genes = np.array(fi.readline().rstrip("\n").split("\t")[2:], dtype=object)
        cells, groups, rows = [], [], []
        for line in fi:
            r = line.rstrip("\n").split("\t")
            cells.append(r[0]); groups.append(r[1])
            rows.append(np.asarray(r[2:], dtype=np.float64))
    return np.vstack(rows), genes, np.array(cells, dtype=object), np.array(groups, dtype=object)


def read_mtx(mtx, genes_f, labels_f, prefix):
    op = gzip.open if mtx.endswith(".gz") else open
    with op(mtx, "rb") as fh:
        m = sio.mmread(fh)                       # genes x cells
    X = sp.csr_matrix(m.T).astype(np.float64)    # cells x genes
    genes = np.array([l.strip() for l in open(genes_f)], dtype=object)
    groups = np.array([l.strip() for l in open(labels_f)], dtype=object)
    cells = np.array([f"{prefix}{i}" for i in range(X.shape[0])], dtype=object)
    return X, genes, cells, groups


def to_cp10k(X, file_type):
    """readData(): CP10K per cell for 'count' input; 'normalized' input passes through."""
    if file_type == "normalized":
        return X
    if sp.issparse(X):
        lib = np.asarray(X.sum(1)).ravel()
        lib[lib == 0] = 1.0
        return sp.csr_matrix(sp.diags(10000.0 / lib) @ X)
    lib = X.sum(1)
    lib[lib == 0] = 1.0
    return X * (10000.0 / lib)[:, None]


# ------------------------------------------------------------------ the method's rules
def filter_genes(Xn, groups, file_type, threshold=0.01):
    """filterGenes(), vectorised.  Same two conditions, same order, same threshold rule."""
    dense = not sp.issparse(Xn)
    if file_type == "normalized":
        nz = Xn[Xn > 0] if dense else Xn.data
        threshold = np.percentile(nz, 25)
    colmax = Xn.max(axis=0)
    colmax = np.asarray(colmax).ravel() if dense else np.asarray(colmax.todense()).ravel()
    idx = np.flatnonzero(colmax >= threshold)          # `if max < threshold: continue`
    if idx.size == 0:
        return idx, threshold
    special = np.zeros(idx.size, dtype=bool)
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        sub = Xn[rows][:, idx]
        sub = sub if dense else np.asarray(sub.todense())
        special |= np.percentile(sub, 80, axis=0) > threshold
    return idx[special], threshold


def gene_stats_log(Xn, idx, file_type):
    """generateReference(): mean/std over log2(CP10K+1) ('count') or the raw values."""
    sub = Xn[:, idx]
    sub = np.asarray(sub.todense()) if sp.issparse(sub) else np.asarray(sub)
    v = np.log2(sub + 1.0) if file_type == "count" else sub
    return v.mean(axis=0), v.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", dest="infile", default=None, help="dense TSV, the original format")
    ap.add_argument("--mtx"); ap.add_argument("--genes"); ap.add_argument("--labels")
    ap.add_argument("--cell-prefix", default="cell")
    ap.add_argument("-o", dest="outDIR", required=True)
    ap.add_argument("-t", dest="file_type", choices=["count", "normalized"], default="count")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    refNum = 256                                       # the original's fixed reference count
    os.makedirs(a.outDIR, exist_ok=True)

    infoLine("Read data ... ")
    if a.mtx:
        X, genes, cells, groups = read_mtx(a.mtx, a.genes, a.labels, a.cell_prefix)
    else:
        X, genes, cells, groups = read_tsv(a.infile)
    Xn = to_cp10k(X, a.file_type)
    for g, n in zip(*np.unique(groups, return_counts=True)):
        if n < 30:
            print("Cell number from " + str(g) + " is lower than 30")

    infoLine("Filter informative genes ... ")
    keep_idx, thr = filter_genes(Xn, groups, a.file_type)
    keep_idx = keep_idx[np.argsort(genes[keep_idx].astype(str), kind="stable")]  # sorted(goodGeneList)
    good = genes[keep_idx].astype(str)
    print(str(len(good)), "out of ", str(len(genes)), "genes will be used for further analysis")

    infoLine("Save gene expression data ... ")
    # saveGeneExpressionByGroup(): {group: {gene: [CP10K per cell, in input order]}}
    sub = Xn[:, keep_idx]
    sub = np.asarray(sub.todense()) if sp.issparse(sub) else np.asarray(sub)
    expByGroup = {}
    for g in np.unique(groups):
        blk = sub[np.flatnonzero(groups == g)]
        expByGroup[str(g)] = {good[j]: list(blk[:, j]) for j in range(len(good))}
    np.save(a.outDIR + "/expression_by_group_gene.npy", expByGroup)

    infoLine("Generate references ... ")
    np.random.seed(a.seed)
    mu, sd = gene_stats_log(Xn, keep_idx, a.file_type)
    refData = np.array([np.absolute(np.random.normal(mu[j], sd[j], refNum)) for j in range(len(good))])
    addData = np.round(np.random.randint(1, 9, size=refData.shape) * 0.0001
                       + np.random.randint(1, 9, size=refData.shape) * 0.00001, 5)
    refData = np.round(np.round(refData, 3) + addData, 5)
    with open(a.outDIR + "/universal.dat", "wt") as fo:
        for sv in np.transpose(refData):
            fo.write("universal\t9999\t" + "|".join("{:.5f}".format(k) for k in sv) + "\n")

    infoLine("Fromat gene expression data ... ")
    V = np.log2(1.0 + sub) if a.file_type == "count" else sub
    groupList = sorted(set(str(g) for g in groups))                    # group2codeHash order
    code = {g: str(i) for i, g in enumerate(groupList)}
    lines = {}
    for i, cid in enumerate(cells):
        lines[str(cid)] = (str(groups[i]) + "\t" + code[str(groups[i])] + "\t"
                           + "|".join("{:.3f}".format(k) for k in V[i]))

    infoLine("Prepare pool data ... ")
    cellIdList = sorted(lines.keys())
    with open(a.outDIR + "/pool.dat", "wt") as fo:
        fo.write("\n".join(lines[c] for c in cellIdList) + "\n")
    with open(a.outDIR + "/sampleList.pool.dat", "wt") as fo:
        fo.write("\n".join(cellIdList) + "\n")

    infoLine("Prepare train and test data ... ")
    # prepareTrainTestData(): iterates group2cellHash, which the original builds in input order
    random.seed(a.seed)
    seen, order_groups = set(), []
    for g in groups:
        if str(g) not in seen:
            seen.add(str(g)); order_groups.append(str(g))
    list_train, list_test = [], []
    for g in order_groups:
        dlist = [lines[str(cells[i])] for i in np.flatnonzero(groups == g)]
        random.shuffle(dlist)
        if len(dlist) > 50:
            list_train.extend(dlist[:int(len(dlist) * 0.9)])
            list_test.extend(dlist[int(len(dlist) * 0.9):])
        else:
            list_train.extend(dlist[:-10]); list_test.extend(dlist[-10:])
    random.seed(a.seed)
    random.shuffle(list_train); random.shuffle(list_test)
    with open(a.outDIR + "/train.dat", "wt") as fo:
        fo.write("\n".join(list_train) + "\n")
    with open(a.outDIR + "/test.dat", "wt") as fo:
        fo.write("\n".join(list_test) + "\n")

    infoLine("Save cell information ... ")
    with open(a.outDIR + "/cellCode.dat", "wt") as fo:
        for g in groupList:
            fo.write(code[g] + "\t" + g + "\n")

    infoLine("Save gene information ... ")
    with open(a.outDIR + "/orderedGeneList.dat", "wt") as fo:
        fo.write("\n".join(good) + "\n")

    infoLine("Save meta ... ")
    with open(a.outDIR + "/meta.dat", "wt") as fo:
        fo.write("class\t%d\ngene\t%d\nreference\t%d\npool\t%d\ntrain\t%d\ntest\t%d\n"
                 % (len(groupList), len(good), refNum, len(lines), len(list_train), len(list_test)))
    infoLine("Done!")


if __name__ == "__main__":
    main()
