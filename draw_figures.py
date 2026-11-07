"""Draw the paper's figures from the result tables in code/.

    python draw_figures.py [--envs envs] [--out build] [--figures figures] [--only "Fig. 4" ...]

reproduce.sh runs this after creating the two conda environments (envs/progression for Python, envs/rfig for R).
Every figure has its own folder in code/ (figure02/ ... figure06/, supplementary_figure01/ ... supplementary_figure10/);
its numbered scripts are its steps, listed below in the order they run.  code/ is first copied to build/, and every step
runs inside that copy, so nothing outside build/ and figures/ is written.  The finished figures are copied to figures/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (figure, what the step does, script under code/, arguments, the finished figure if this is the figure's last step)
STEPS = [
    ("Fig. 2", "compute marker recovery and its tests", "figure02/01_marker_recovery.py", [], None),
    ("Fig. 2", "draw", "figure02/02_draw.py", [], "figure02/out/figure2.pdf"),
    ("Fig. 3", "draw", "figure03/01_draw.py", [], "figure03/out/figure3.pdf"),
    ("Fig. 4", "draw", "figure04/01_draw.py", [], "figure04/out/figure4.pdf"),
    ("Fig. 5", "draw", "figure05/01_draw.py", [], "figure05/out/figure5.pdf"),
    ("Fig. 6", "draw", "figure06/01_draw.py", [], "figure06/out/figure6.pdf"),
    ("Supplementary Fig. 1", "draw", "supplementary_figure01/01_draw.py", [], "supplementary_figure01/out/S1_attribution_faithfulness.pdf"),
    ("Supplementary Fig. 2", "draw", "supplementary_figure02/01_draw.py", [], "supplementary_figure02/out/S2_encoder_consistency.pdf"),
    ("Supplementary Fig. 3", "draw", "supplementary_figure03/01_draw.py", [], "supplementary_figure03/out/S3_marker_benchmark.pdf"),
    ("Supplementary Fig. 4", "draw", "supplementary_figure04/01_draw.py", [], "supplementary_figure04/out/S4_broad_and_cell_cycle.pdf"),
    ("Supplementary Fig. 5", "draw", "supplementary_figure05/01_draw.py", [], "supplementary_figure05/out/S5_sensitivity_and_annotation.pdf"),
    ("Supplementary Fig. 6", "draw", "supplementary_figure06/01_draw.py", [], "supplementary_figure06/out/S6_program_validation.pdf"),
    ("Supplementary Fig. 7", "draw", "supplementary_figure07/01_draw.py", [], "supplementary_figure07/out/S7_developmental_patterns.pdf"),
    ("Supplementary Fig. 8", "draw", "supplementary_figure08/01_draw.py", [], "supplementary_figure08/out/S8_scoring_added_genes.pdf"),
    ("Supplementary Fig. 9", "draw", "supplementary_figure09/01_draw.py", [], "supplementary_figure09/out/S9_cross_atlas_transfer.pdf"),
    ("Supplementary Fig. 10", "draw", "supplementary_figure10/01_draw.py", [], "supplementary_figure10/out/S10_whole_atlas_time.pdf"),
]


def final_name(figure):
    return figure.replace("Supplementary Fig. ", "SupplementaryFigure").replace("Fig. ", "Figure") + ".pdf"


def run_env(root, envs):
    home = f"{root}/_home"
    none = f"{home}/none"                              # never created: R reads no user profile or library
    return dict(os.environ, MPLCONFIGDIR=f"{home}/mpl", XDG_CACHE_HOME=f"{home}/cache", PYTHONNOUSERSITE="1",
                R_PROFILE_USER=none, R_ENVIRON_USER=none, R_LIBS_USER=none,
                RSCRIPT=f"{envs}/rfig/bin/Rscript", FIGURE_FONTS=os.path.join(HERE, "environment", "fonts"))


def check_fonts(py, rs, env):
    """Warn if either environment does not find Liberation Sans, the figure font."""
    probes = [[py, "-c", "import matplotlib.font_manager as f; "
                         "print(f.findfont(f.FontProperties(family='Liberation Sans'), fallback_to_default=False))"],
              [rs, "-e", 'cat(systemfonts::match_fonts("Liberation Sans")$path)']]
    for cmd in probes:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        path = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        if not os.path.basename(path).startswith("LiberationSans"):
            print(f"warning: {os.path.basename(cmd[0])} does not find Liberation Sans, so the figures are drawn with "
                  f"another font; reproduce.sh adds environment/fonts/ to both environments", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--envs", default=os.path.join(HERE, "envs"),
                    help="folder holding the environments progression and rfig (default: envs)")
    ap.add_argument("--out", default="build", help="working folder (default: build)")
    ap.add_argument("--figures", default="figures", help="folder for the finished figures (default: figures)")
    ap.add_argument("--only", nargs="+", metavar="FIGURE", help='draw only these figures, e.g. "Fig. 4"')
    a = ap.parse_args()
    root, envs, figs = os.path.abspath(a.out), os.path.abspath(a.envs), os.path.abspath(a.figures)
    known = list(dict.fromkeys(s[0] for s in STEPS))
    chosen = a.only or known
    if set(chosen) - set(known):
        sys.exit(f"unknown figure(s) {sorted(set(chosen) - set(known))}; choose from {known}")
    if os.path.exists(root):
        sys.exit(f"{root} exists; remove it or pass --out")
    shutil.copytree(os.path.join(HERE, "code"), os.path.join(root, "code"))
    py, rs = f"{envs}/progression/bin/python", f"{envs}/rfig/bin/Rscript"
    env = run_env(root, envs)
    check_fonts(py, rs, env)
    failed = []
    for i, (figure, what, script, args, _) in enumerate(STEPS, 1):
        if figure not in chosen:
            continue
        path = os.path.join(root, "code", script)
        cmd = [py, "-B", path, *args] if script.endswith(".py") else [rs, path, *args]
        t = time.time()
        r = subprocess.run(cmd, cwd=os.path.dirname(path), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True)
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"{i:2d}  {figure:22s} {what:58s} {status}  {time.time() - t:4.0f} s", flush=True)
        if r.returncode:
            failed.append(f"{i} {figure}: {what}")
            print(r.stderr[-2000:], file=sys.stderr)
    os.makedirs(figs, exist_ok=True)
    print(f"\nfinished figures in {figs}:")
    for figure, _, _, _, out in STEPS:
        if out and figure in chosen:
            src = os.path.join(root, "code", out)
            ok = os.path.exists(src)
            if ok:
                shutil.copyfile(src, os.path.join(figs, final_name(figure)))
            print(f"  {final_name(figure)}{'' if ok else '  (missing)'}")
    sys.exit(f"failed steps: {failed}" if failed else 0)


if __name__ == "__main__":
    main()
