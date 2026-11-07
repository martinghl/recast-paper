# Package check

Each script recomputes the RECAST results behind some of the manuscript's panels with the released
package and compares them with the tables the analysis scripts wrote. Attributions come only from
`import recast`; `common.assert_package_only()` stops a check if any other attribution code is loaded.
The results below were obtained with sc-recast 0.8.1. Version 0.8.2 changes only `recast.composite`,
the command-line interface and the single-file script, none of which these checks call.

## Running

Environment `SC` (`../environments/SC.yml`) with
`pip install git+https://github.com/martinghl/recast@v0.8.2`, and a GPU.

```
export RESEARCH_ROOT=/path/to/research    # its analysis/ folder: this analysis/ folder with the data and result files
export SCIMILARITY_MODEL=/path/to/scimilarity
export PACKAGE_CHECK_OUT=/path/to/output
python check_benchmark_panels.py              # usage of each script is in its first lines
```

## Results (sc-recast 0.8.1, October 2026)

| Script | Items | Package vs. stored tables |
|---|---|---|
| `check_benchmark_panels.py` | Fig. 3, Supplementary Fig. 3, Supplementary Data, sheets 12–15 | 120 subtype panels in 33 lineages: identical for 118; the other 2 have the same top-200 set, with two genes in swapped order after rank 189. |
| `check_deletion_insertion.py` | Fig. 2e, Supplementary Fig. 1a | Deletion and insertion AUCs equal to 4 decimals; curves within 5.2 × 10⁻⁵. |
| `check_target_groups.py` | Fig. 2c, 2d (RECAST: the rankings with n = N), 2f, 2g | 97 subtypes, 11,891 group rankings: 11,881 top-20 lists identical; 9 differ by two adjacent genes in swapped order and 1 in its 20th gene (CD99 instead of FOS, neither in the subtype's marker panel), so Recall@20 is unchanged for all. |
| `check_reference_choice.py` | Fig. 4b–f | 112/112 top-20 lists identical; recall, reference clusters and rank correlations equal for 112/112. |
| `check_program_scoring.py` | Fig. 6, Supplementary Fig. 8, Supplementary Data, sheets 7–11 | Gene weights within 1.1 × 10⁻⁴ (largest weight 0.044); reference profiles within 2 × 10⁻⁵, except at genes with duplicated symbols in four datasets (up to 0.24: the check sums duplicated symbols, as the attribution scripts did, while the original scoring code took the first); per-cell program call equal for 284,996/285,002 cells; balanced accuracy and macro-F1 within 1.3 × 10⁻⁴ (Fig. 6a) and 1.8 × 10⁻³ (Fig. 6d variants); Fig. 6d means equal to 4 decimals and no win or loss changes among 2,016 method comparisons. |
| `check_top50_panels.py` | Fig. 2b, Fig. 5, Supplementary Fig. 6 | 93/93 top-50 lists identical (26 lineages). |
| `check_cd8_separability.py` | Fig. 2a | kNN accuracy 0.891 and macro-F1 0.907, as in the figure. |
| `check_profile_centroid.py` | Supplementary Fig. 2a–d | 456 comparisons with five encoders: all quantities within 6 × 10⁻⁴; medians equal to 4 decimals. |
| `check_sensitivity.py` | Supplementary Fig. 5a–e | 319/320 rows equal; in one (zhao_Mono CD16+ Mo, 100 reference cells) one curated gene fewer enters the top 20 in one of three draws (recall@20 0.333 vs. 0.367). |
| `check_label_noise.py` | Supplementary Fig. 5f–h | 496/496 lists identical; all 120 perturbed label sets regenerated. |
| `check_cell_replacement.py` | Supplementary Fig. 1b, 1c | All counts equal; identity loss within 0.005, the other read-outs within 6 × 10⁻⁶. |
| `check_encoders.py` | Supplementary Fig. 2e | SCimilarity, two SSL encoders and scFoundation: recall equal 48/48, kNN accuracy 24/24, macro-F1 23/24 (difference 0.001). scVI: the original run kept no model and a retrained model differs (recall within 0.12); on the same retrained model the package and the original code give identical top-20 lists (24/24), recall and kNN values. |
| `check_broad_populations.py` | Supplementary Fig. 4a, 4b | 107/107 top-50 lists identical (8 atlases). |
| `check_cell_cycle.py` | Supplementary Fig. 4c, 4d | 35/35 top-200 sets identical; order identical for 33. |
| `check_development.py` | Supplementary Fig. 7, Supplementary Data, sheets 5–6 | Pseudotime, top-20 lists and profiles equal; 4,670/4,702 dopaminergic ranks equal, the first difference at rank 894. |
| `check_transfer.py` | Supplementary Fig. 9, Supplementary Data, sheet 11 | Fine subtypes: 50/50 top-20 lists identical; broad populations: transfer AUROC and retention equal for 54/54. |
