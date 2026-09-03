#!/usr/bin/env bash

# Extract stored range curves and compile one PGFPlots page per benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GENDIL_ROOT="${GENDIL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)}"
INPUT_DIR="${1:-${GENDIL_ROOT}/results/spacemit-k3}"
OUTPUT_DIR="${2:-${GENDIL_ROOT}/results/spacemit-k3}"
TEX_FILE="${OUTPUT_DIR}/range-coordinates.tex"

if ! command -v pdflatex >/dev/null 2>&1; then
  printf 'error: pdflatex is required\n' >&2
  exit 1
fi
if ! kpsewhich pgfplots.sty >/dev/null 2>&1; then
  printf 'error: pgfplots.sty is required; install TeX Live pgfplots\n' >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${SCRIPT_DIR}/extract-range-results.py" "${INPUT_DIR}" \
  --csv "${OUTPUT_DIR}/range-coordinates.csv" \
  --pgfplots "${OUTPUT_DIR}/range-coordinates.pgfplots.tex"

cat > "${TEX_FILE}" <<'EOF'
\documentclass{article}
\usepackage[margin=0.35in]{geometry}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\input{range-coordinates.pgfplots.tex}
\end{document}
EOF

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory "${OUTPUT_DIR}" "${TEX_FILE}"
printf 'PDF written to %s/range-coordinates.pdf\n' "${OUTPUT_DIR}"
