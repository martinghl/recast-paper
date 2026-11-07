"""Per-subtype expression of the RECAST-specific genes (Supplementary Fig. 6).

A gene that RECAST ranks for a subtype and that COSG's top 30 does not contain should be expressed in that subtype and
not in its siblings (the comparison both methods were run on).

Counts come from sce_export_v2 (export_lineage_counts.py: integer counts from <stem>_fullgene.h5ad; checked to be
integer here too).

Writes:
  example_gene_expression.csv   gene x lineage x state: n_cells, mean log1p CP10K, fraction > 0,
                                   and a specificity margin (target state minus the best sibling)"""
import gzip, io, sys
import numpy as np
import pandas as pd
import scipy.io as sio

BV = "/path/to/research/analysis/validation/biology"
OUT = f"{BV}/programs"
SCE = "/path/to/research/analysis/subtype_benchmark/sce_export_v2"

D = pd.read_csv(f"{OUT}/method_specific_hits.csv")
D["lin_list"] = D.lineages.str.split(";")

rows = []
for lin in sorted({l for ls in D.lin_list for l in ls}):
    want = sorted({r.gene for r in D.itertuples() if lin in r.lin_list})
    genes = pd.read_csv(f"{SCE}/{lin}_genes.txt", header=None)[0].astype(str).to_numpy()
    lab = pd.read_csv(f"{SCE}/{lin}_labels.txt", header=None)[0].astype(str).to_numpy()
    with gzip.open(f"{SCE}/{lin}_counts.mtx.gz", "rb") as f:
        M = sio.mmread(io.BytesIO(f.read())).tocsr()          # genes x cells
    assert M.shape == (len(genes), len(lab)), (M.shape, len(genes), len(lab))
    assert np.allclose(M.data, np.round(M.data)) and M.data.min() >= 1, \
        f"{lin}: counts are not integers -- this is the lognorm-truncation bug, do not plot it"
    tot = np.asarray(M.sum(0)).ravel().astype(float)          # per-cell library size
    gi = {g: i for i, g in enumerate(genes)}
    states = sorted(pd.unique(lab))
    for g in want:
        if g not in gi:
            continue
        c = np.asarray(M[gi[g]].todense()).ravel().astype(float)
        ln = np.log1p(c / tot * 1e4)                          # log1p CP10K, the field's default
        for s in states:
            m = lab == s
            rows.append(dict(lineage=lin, gene=g, state=s, n_cells=int(m.sum()),
                             mean_lognorm=float(ln[m].mean()), frac_expressing=float((c[m] > 0).mean())))
    print(f"[{lin}] {M.shape[1]} cells, {len(states)} states, {sum(g in gi for g in want)}/{len(want)} genes",
          flush=True)

E = pd.DataFrame(rows)

# the specificity margin: how far the gene's own subtype sits above the best sibling in the SAME lineage.
# Computed per (lineage, gene, target state) so a gene can be exemplary in one lineage and not another.
tgt = D.explode("lin_list").rename(columns={"lin_list": "lineage"})[
    ["lineage", "gene", "subtype", "comp", "program", "recast_rank", "n_datasets"]].drop_duplicates()
marg = []
for r in tgt.itertuples():
    sub = E[(E.lineage == r.lineage) & (E.gene == r.gene)]
    if sub.empty:
        continue
    own = sub[sub.state == r.subtype]
    if own.empty:                       # the state label in the cohort table is not the export's label
        continue
    sib = sub[sub.state != r.subtype]
    marg.append(dict(lineage=r.lineage, gene=r.gene, state=r.subtype, program=r.program, comp=r.comp,
                     recast_rank=r.recast_rank, n_datasets=r.n_datasets,
                     own_mean=own.mean_lognorm.iloc[0], own_frac=own.frac_expressing.iloc[0],
                     best_sib_mean=sib.mean_lognorm.max(), best_sib_frac=sib.frac_expressing.max(),
                     margin_mean=own.mean_lognorm.iloc[0] - sib.mean_lognorm.max(),
                     margin_frac=own.frac_expressing.iloc[0] - sib.frac_expressing.max()))
G = pd.DataFrame(marg).sort_values("margin_mean", ascending=False)

E.to_csv(f"{OUT}/example_gene_expression.csv", index=False)
G.to_csv(f"{OUT}/example_gene_margins.csv", index=False)
print(f"\n{len(E)} gene x lineage x state rows; {len(G)} (lineage, gene, target state) candidates")
print(G.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"[wrote] example_gene_expression.csv, example_gene_margins.csv")
