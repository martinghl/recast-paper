# Analysis scripts

The scripts that computed the result tables in `../code/`. They need the raw datasets and, for most analyses, the
SCimilarity model and a GPU, so they do not run from this repository alone. They are here so that each result in the
paper can be traced to the code that produced it.

## Finding the script behind a result

`MAP.tsv` has one row per figure panel and per Supplementary Data sheet:

| Column | Contents |
|---|---|
| `item` | Figure panel or Supplementary Data sheet |
| `drawn_by` | The script in `code/` that draws it |
| `data` | The tables in `code/` it reads |
| `computed_by` | The analysis scripts that wrote those tables, with the output file in brackets where it helps |
| `package_check` | The script in `package_check/` that recomputes its RECAST results with the released package |
| `notes` | Anything the scripts alone do not say |

Paths are relative to the repository root. Three rows describe inputs that many results share: `[datasets]`,
`[marker panels]` (the subtype marker panels of the benchmark) and `[marker benchmark]` (the table of the 56-method
benchmark, Fig. 3 and Supplementary Fig. 3).

## What is here

| Path | Contents |
|---|---|
| `MAP.tsv` | Figure panel or data sheet → scripts and tables (above) |
| `EDITED.tsv` | Scripts whose behaviour differs from the version that was run, and how |
| `package_check/` | Recomputes the RECAST results with the released package and compares them with the stored tables; results in its README |
| `gene_sets/` | The gene-set snapshots and background gene list of the enrichment analyses, with SHA-256 checksums (`sha256sum -c SHA256SUMS` after decompressing) |
| `environments/` | The conda environments the analyses ran in |
| `data_sources.tsv` | The datasets and their accessions |
| `subtype_benchmark/` | The fine-subtype marker benchmark: RECAST and comparator selections, marker panels (`full_benchmark/ground_truth/`), scoring (`full_benchmark/scoring/`), the marker-panel tools (`dedicated_tools/`), the added lung, gut and brain lineages (`full_benchmark/added_lineages/`), cell replacement, cross-atlas transfer and the encoder comparison |
| `broad_populations/` | Marker recall and within-atlas AUROC of broad cell types |
| `cell_cycle/` | Cell-cycle content of the selected genes (identity and phase comparisons) |
| `validation/` | Choice of reference, sensitivity to the settings, perturbed annotations, profile–centroid alignment, encoders, and the biological analyses (program enrichment, developmental trajectory, CD8 T-cell scoring) |
| `program_scoring/` | Scoring single cells with curated gene programs (Fig. 6, Supplementary Fig. 8) |
| `scoring_benchmark/`, `ans_replication/` | RECAST against the gene-set scoring methods with the protocol of adjusted neighbourhood scoring (ANS), and the ANS inputs |
| `attribution_methods/` | The attribution code of the analyses (contrastive and population integrated gradients) |
| `atlases/`, `new_datasets/` | Attribution and label files of the atlases, broad-population expert panels, and the added datasets |
| `paper/` | Tables assembled from the analysis outputs for the figures and the Supplementary Data |
| `cpu_server/` | Scripts that ran on a separate CPU server (the R marker-selection methods, data export, timing) |
| `external/` | Three files of SIGnature used by the encoder comparisons |
| `real_counts.py` | Reads the raw integer counts of an atlas (see its docstring) |

## Paths

Machine-specific paths are replaced by placeholders:

- `/path/to/research/analysis/` is this folder on the machine where the analyses ran, together with the data and
  result files the scripts read and write (not included);
- `/path/to/cpu_server/` is the working folder on the CPU server (`cpu_server/` here);
- `/path/to/envs/<name>` is the conda environment `<name>` (`environments/<name>.yml`);
- `/path/to/scimilarity` is the SCimilarity v1.1 model (Zenodo record 10685499);
- `/path/to/SIGnature` is a checkout of github.com/Genentech/SIGnature (commit c1546cb);
- `/path/to/scratch/` is a download folder for source files.

## Names

Scripts, folders, variables and most output files have descriptive names. Some stored result files and identifiers
keep the names under which they were written, because other steps read them under those names:

| Name | Meaning |
|---|---|
| `fs_*`, `tf_*` | Lineages of the subtype benchmark (`fs_`: one dataset per lineage; `tf_`: lineages present in several atlases, used for transfer) |
| `fs37_<tissue>_<dataset>` | Lineages of the added lung, gut and brain datasets |
| `recast_M0`, `M0`, `m0` | RECAST in result tables |
| `SIGnature_native`, `native_direct`, `native_gated` | Reference-free attribution (SIGnature) in result tables |
| `fs_uniq_perstate_v3.csv` | Marker recall per subtype; its `fine` tier defines the 61 subtypes of the first benchmark cohort |
| `fs_cm_arm3_perstate_<lineage><tag>.csv`, `arm2`, `arm3` | Per-subtype one-versus-rest AUROC; `arm2` and `arm3` mark marker-recall and marker-free evaluations |
| `topN_<lineage>_recast_M0.csv`, `topN200_*` | RECAST's top 50 and top 200 genes per subtype |
| `mr4_*` | Tables of Fig. 2f, g (target aggregation, encoder and target sample size) |
| `ps_*` | Runs with perturbed subtype labels (Supplementary Fig. 5f–h) |

### Result tags

Scripts that read the environment variable `RECAST_RESULT_TAG` append it to the names of their output files. Results
were recomputed twice, and each recomputation was written beside the earlier files under a tag:

| Tag | Recomputation |
|---|---|
| `_v2` | From the raw integer counts (`real_counts.py`); the comparators' count matrices (`sce_export_v2`) and gene lists (`fs_genesets_v2`) are those of this run |
| `_v3` | RECAST again, with the reference and target profiles computed as means of per-cell log-normalized expression (`centroid="mean_lognorm"`) in every lineage; the comparator results stay those of `_v2`, which do not depend on this setting |
| `_v3x` | The `_v3` marker-benchmark tables with the five marker-panel tools added (`subtype_benchmark/dedicated_tools/`) |

The figures use the `_v3` results (`_v3x` for the marker benchmark) where a script writes tagged files.

## Supplementary Data

The values of each sheet are computed by the scripts named in `MAP.tsv`. The workbook itself (titles, notes, display
names and formatting) was assembled separately and is not built by this repository.
