#!/usr/bin/env python
"""Merge the five dedicated marker-selection tools into the comparator gene lists of the added lineages, with
dedicated_tools/merge_gene_lists.py (only its lineage list and paths change):

    data/fs_genesets_v2/<lin>.csv       depth-50 lists of the 51 comparator methods (from the CPU server)
    data/fs_genesets200_v2/<lin>.csv    depth-200 lists of the same methods
    data/dedicated_tools/genesets200/<lin>__<m>.csv  NS-Forest, scMAGS, RankCorr, scGeneFit (CPU server) + scMarkerGene (GPU)
->  data/genesets50_plus/<lin>.csv, data/genesets200_plus/<lin>.csv, data/geneset_merge_report.csv

Env analysis.  Usage: python -I gene_lists.py
"""
import os
import sys

FSB = "/path/to/research/analysis/subtype_benchmark"
FULL_BENCHMARK = f"{FSB}/full_benchmark"
DATA = f"{FULL_BENCHMARK}/data"
sys.path.insert(0, f"{FSB}/dedicated_tools")
sys.path.insert(0, f"{FULL_BENCHMARK}/added_lineages")
import merge_gene_lists as G  # noqa: E402
import added_datasets as D  # noqa: E402

G.LIN20 = list(D.NEW_IDS)      # the merger's lineage list
G.HERE = DATA                  # where it writes geneset_merge_report.csv
sys.argv = [sys.argv[0], "--new", f"{DATA}/dedicated_tools/genesets200",
            "--src50", f"{DATA}/fs_genesets_v2", "--src200", f"{DATA}/fs_genesets200_v2",
            "--dst50", f"{DATA}/genesets50_plus", "--dst200", f"{DATA}/genesets200_plus"]
G.main()
