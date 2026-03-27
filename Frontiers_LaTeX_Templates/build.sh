#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_TEX="main.tex"
MAIN_BASENAME="${MAIN_TEX%.tex}"

cd "${ROOT_DIR}"

clean() {
  if command -v latexmk >/dev/null 2>&1; then
    latexmk -C "${MAIN_TEX}" >/dev/null 2>&1 || true
  fi

  rm -f \
    "${MAIN_BASENAME}.aux" \
    "${MAIN_BASENAME}.bbl" \
    "${MAIN_BASENAME}.bcf" \
    "${MAIN_BASENAME}.blg" \
    "${MAIN_BASENAME}.fdb_latexmk" \
    "${MAIN_BASENAME}.fls" \
    "${MAIN_BASENAME}.lof" \
    "${MAIN_BASENAME}.log" \
    "${MAIN_BASENAME}.lot" \
    "${MAIN_BASENAME}.out" \
    "${MAIN_BASENAME}.run.xml" \
    "${MAIN_BASENAME}.synctex.gz" \
    "${MAIN_BASENAME}.toc"
}

if [[ "${1:-}" == "clean" ]]; then
  clean
  echo "Cleaned LaTeX build artifacts."
  exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -file-line-error "${MAIN_TEX}"
else
  pdflatex -interaction=nonstopmode -file-line-error "${MAIN_TEX}"
  bibtex "${MAIN_BASENAME}"
  pdflatex -interaction=nonstopmode -file-line-error "${MAIN_TEX}"
  pdflatex -interaction=nonstopmode -file-line-error "${MAIN_TEX}"
fi

echo "Build complete: ${ROOT_DIR}/${MAIN_BASENAME}.pdf"
