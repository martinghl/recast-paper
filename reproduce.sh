#!/usr/bin/env bash
# Redraw all figures in one step. The first run creates the two conda environments in envs/ (Python: progression,
# R: rfig) and adds the figure font to them; then draw_figures.py draws the figures into figures/.
# Needs conda, mamba or micromamba. Options are passed on to draw_figures.py (e.g. --out DIR).
set -euo pipefail
cd "$(dirname "$0")"
if command -v micromamba >/dev/null; then CREATE=(micromamba create -y)
elif command -v mamba >/dev/null; then CREATE=(mamba env create)
elif command -v conda >/dev/null; then CREATE=(conda env create)
else echo "reproduce.sh needs conda, mamba or micromamba" >&2; exit 1; fi

make_env() {  # environment name, environment file
  [ -f "envs/$1/.complete" ] && return
  rm -rf "envs/$1"
  "${CREATE[@]}" -p "$PWD/envs/$1" -f "environment/$2"     # absolute path: micromamba runs pip from elsewhere
  touch "envs/$1/.complete"
}
make_env progression python.yml
make_env rfig r.yml

for e in progression rfig; do mkdir -p "envs/$e/fonts" && cp environment/fonts/*.ttf "envs/$e/fonts/"; done
cp environment/fonts/*.ttf "$(envs/progression/bin/python -c 'import matplotlib, os; print(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))')/"
exec envs/progression/bin/python draw_figures.py --envs envs "$@"
