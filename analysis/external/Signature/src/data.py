"""Harmonize per-study expression to the SCimilarity gene space (Option A, SC env).

A raw-count .h5ad from ``census_fetch`` (var_names = gene symbols, var['measured'],
obs['target'], obs['dataset_id']) becomes an AnnData aligned to the 28,231-gene SCimilarity
space and log-normalized OVER that space (SCimilarity convention: align THEN lognorm). The
SAME resulting ``X`` feeds both IG attribution and the DE / neutral evaluator, so attributions
and expression are computed on identical cells & genes (PLAN guardrail: one gene universe).
"""
from __future__ import annotations

import os


def read_gene_order(model_dir: str) -> list:
    """The frozen SCimilarity input gene space (list of symbols), read like the model does."""
    with open(os.path.join(model_dir, "gene_order.tsv")) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def harmonize(adata_raw, gene_order):
    """Align a raw-count AnnData to ``gene_order`` and log-normalize over that space.

    Returns an AnnData with:
      - ``X``            = log-normalized expression (n_cells x len(gene_order)), var_names == gene_order
      - ``layers['counts']`` = raw counts aligned to gene_order (zero-padded for unmeasured genes)
      - ``obs``          = preserved (incl. 'target', 'dataset_id')
      - ``var['present']`` = whether each gene was TRULY measured in this study AND is in the panel.
        Uses ``adata_raw.var['measured']`` (Census presence matrix) when available; otherwise
        falls back to gene-list membership (only correct when the source already omits unassayed
        genes). Genes absent from a study's assay must NOT count as "present" — they are
        structural zeros, not biological silence.
    """
    from SIGnature.utils import align_dataset, lognorm_counts

    gene_order = list(gene_order)
    a = adata_raw.copy()
    a.var_names_make_unique()

    if "measured" in a.var.columns:
        meas = a.var["measured"].to_numpy().astype(bool)
        measured_syms = {str(s) for s in a.var_names[meas]}
    else:
        measured_syms = set(map(str, a.var_names))
    present = measured_syms.intersection(gene_order)

    a = align_dataset(a, gene_order)               # X -> raw counts aligned (zero-padded), keeps counts layer
    if "counts" not in a.layers:
        a.layers["counts"] = a.X.copy()
    a = lognorm_counts(a)                           # X -> lognorm over the 28,231-gene space
    a.var["present"] = [g in present for g in a.var_names]
    return a


def frozen_gene_universe(adatas, require_all: bool = True) -> list:
    """Genes (subset of gene_order) measured across studies, in gene_order order.

    ``require_all=True`` → intersection of truly-measured genes across all studies (the fair,
    shared universe for selection + evaluation). ``False`` → union. All inputs must have been
    harmonized against the identical gene_order (asserted).
    """
    if not adatas:
        return []
    ref = list(adatas[0].var_names)
    for a in adatas[1:]:
        if list(a.var_names) != ref:
            raise ValueError(
                "frozen_gene_universe: all adatas must share the same gene_order (var_names)."
            )
    present_sets = [set(a.var_names[a.var["present"].to_numpy()]) for a in adatas]
    keep = set.intersection(*present_sets) if require_all else set.union(*present_sets)
    return [g for g in ref if g in keep]
