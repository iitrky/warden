#!/bin/bash
# Build manuscript PDF (XeLaTeX), Word (.docx) and LaTeX source from the Markdown parts.
set -e
cd "$(dirname "$0")"
cat part1.md part2.md part3.md part4.md part5.md > manuscript.md
COMMON="--resource-path=.:.. -H header.tex -V fontsize=11pt -V geometry:margin=2.5cm -V linestretch=1.15 -V colorlinks=true -V linkcolor=black -V urlcolor=blue"
pandoc manuscript.md -o manuscript.pdf --pdf-engine=xelatex $COMMON \
  -V mainfont="TeX Gyre Termes" -V monofont="DejaVu Sans Mono" -V sansfont="DejaVu Sans"
pandoc manuscript.md -s -o manuscript.tex $COMMON
python3 make_springer_docx.py manuscript.md manuscript.docx
