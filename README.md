# RECAST paper: figure code

This repository redraws the figures of the paper "RECAST: reference-conditioned attribution of single-cell
transcriptomes for subtype-specific gene signature prioritization" from the result tables of its analyses, and holds
the analysis scripts that computed those tables.

The RECAST software itself is a separate repository: <https://github.com/martinghl/recast>.

## Redraw the figures

You need Linux (x86-64) and conda, mamba or
[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Then:

    git clone https://github.com/martinghl/recast-paper
    cd recast-paper
    bash reproduce.sh

The first run creates two conda environments in `envs/` (Python and R, about 3 GB). Drawing then takes about
10 minutes. No GPU, raw data or model weights are needed.

The finished figures are written to `figures/`:

    Figure2.pdf ... Figure6.pdf
    SupplementaryFigure1.pdf ... SupplementaryFigure10.pdf

Fig. 1 is a hand-drawn diagram, so it is not here. To draw again, delete `build/` and `figures/` first. To draw some
figures only: `envs/progression/bin/python draw_figures.py --envs envs --only "Fig. 4" "Supplementary Fig. 9"`.

## What is in this repository

| Path | Contents |
|---|---|
| `reproduce.sh` | Sets up the two environments, then runs `draw_figures.py` |
| `draw_figures.py` | Copies `code/` to `build/`, runs the drawing scripts there in order and copies the finished figures to `figures/` |
| `code/` | The drawing scripts and the result tables they read, one folder per figure |
| `environment/` | The two conda environment files and the figure font (Liberation Sans) |
| `analysis/` | The scripts that computed the result tables (they need the raw data; see `analysis/README.md`) |

## How `code/` is organized

Each figure has its own folder: `figure02/` to `figure06/` and `supplementary_figure01/` to `supplementary_figure10/`.
In each folder:

- the numbered scripts are the steps that make the figure, run in the order of their numbers (`01_draw.py`; Fig. 2
  first computes the tables of panels c and d with `01_marker_recovery.py`, then draws with `02_draw.py`);
- the other scripts draw single panels or read the tables, and their names say which (for example
  `figure04/panels_b_to_f.py`); the docstring of each numbered script lists them;
- `data/` holds the result tables that only this figure reads.

Two folders are shared: `shared_data/` holds the tables that several figures read (for example the marker benchmark of
Fig. 3 and Supplementary Fig. 3), and `common/` holds the code they share (style, colours, page assembly).

The scripts write their output to an `out/` folder next to them. `draw_figures.py` runs them in a copy of `code/`
under `build/`, so that `code/` itself is never changed.

## Where the result tables come from

Most result tables in `code/` come from the analyses described in the paper's Methods. Their scripts are in
`analysis/`; they need the raw datasets, so they do not run from this repository alone. `analysis/MAP.tsv` traces each
figure panel and each sheet of the Supplementary Data to the script that draws it, the tables it reads and the scripts
that computed them; `analysis/README.md` explains the rest. The tables of Fig. 2c and 2d are computed from stored gene
lists by `code/figure02/01_marker_recovery.py`, one of the drawing steps.

## Small differences from the figures in the paper

- The small random offsets (jitter) of some points differ in Fig. 3 and Supplementary Figs. 3, 4b and 9.
- The PDF files differ byte for byte from the submitted ones, but look the same and contain the same text.

## License

MIT (`LICENSE`). The font in `environment/fonts/` is under the SIL Open Font License 1.1.
