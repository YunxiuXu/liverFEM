#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

TEX_FILE="Main.tex"
PDF_FILE="main.pdf"
BASE_NAME="${TEX_FILE%.tex}"
SOURCE_PDF="${BASE_NAME}.pdf"

if [[ ! -f "$TEX_FILE" ]]; then
  echo "未找到 LaTeX 主文件: $TEX_FILE (当前目录: $SCRIPT_DIR)"
  exit 1
fi

if command -v latexmk >/dev/null 2>&1; then
  # latexmk 负责多次编译、引用等（若有需要）。
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$TEX_FILE"
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "未找到可用的 pdflatex 或 latexmk。请先安装 TeX Live 并确保命令可用。"
    exit 1
  fi

  # 清理常见中间文件，避免旧内容干扰（不删除源码与图片）。
  rm -f "${BASE_NAME}".aux "${BASE_NAME}".log "${BASE_NAME}".out "${BASE_NAME}".toc "${BASE_NAME}".synctex.gz "${BASE_NAME}".bbl "${BASE_NAME}".blg

  # access.tex 已改为 BibTeX（\bibliography{references}），所以需要跑 bibtex 再多次 pdflatex。
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"

  if ! command -v bibtex >/dev/null 2>&1; then
    echo "未找到 bibtex。请安装 TeX Live，并确保命令可用。"
    exit 1
  fi

  bibtex "$BASE_NAME"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"
  pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"
fi

if [[ ! -f "$SOURCE_PDF" ]]; then
  echo "编译未生成 ${SOURCE_PDF}，请查看上方输出的错误信息。"
  exit 1
fi

# 无论是 latexmk 还是纯 pdflatex，这里都把生成的 jobname.pdf 复制成 main.pdf
# macOS 下某些情况下若内容完全相同，cp 可能返回非 0 导致脚本失败，所以先判断。
if [[ -f "$PDF_FILE" ]] && command -v cmp >/dev/null 2>&1 && cmp -s "$SOURCE_PDF" "$PDF_FILE"; then
  echo "编译完成: $PDF_FILE (源文件: ${SOURCE_PDF}，内容未变化)"
else
  cp -f "$SOURCE_PDF" "$PDF_FILE"
  echo "编译完成: $PDF_FILE (源文件: ${SOURCE_PDF})"
fi

