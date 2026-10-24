#!/usr/bin/env bash
# Build the course paper: tables + figures from the stored JSON/RESULTS files, then LaTeX.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
PY=${PYTHON:-python3}
"$PY" -I build_tables.py
"$PY" -I build_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -n "^!" build.log | head -20; exit 1; }
echo "main.pdf: $(pdfinfo main.pdf | awk '/^Pages:/{print $2}') pages"
