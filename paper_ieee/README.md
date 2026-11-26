# IEEE Internet of Things Journal version

`main.tex` is the IEEE IoT-J submission (double-column, target <= 8 pages to avoid
mandatory over-length charges of USD 175/page).

## Compile on Overleaf (recommended)
1. New Project -> Upload Project -> upload `WARDEN_IEEE_IoTJ_source.zip`.
2. Compiler: pdfLaTeX. Main document: `main.tex`.
3. Overleaf has the official `IEEEtran.cls`, so the paper is typeset with it automatically
   (`ieee_fallback.tex` is then ignored). Check that the PDF is still 8 pages.

## Compile offline
`pdflatex main.tex` twice. If `IEEEtran.cls` is not installed, an IEEEtran-equivalent
layout (`ieee_fallback.tex`: letter paper, 43 pc x 58 pc text block, 1 pc column gap,
10-pt Times) is used so the page count can still be checked.

## References
Edit `refs_src.tex`, then run `python3 order_refs.py`; it writes `refs.tex` with the
entries numbered in order of first citation, as IEEE requires.

## Figures
`python3 scripts/fig_ieee.py` (from the repository root) regenerates every plot in `figs/`
from the saved results; Fig. 2 is drawn in TikZ inside `main.tex`.

## Word version
`python3 tex2md.py && python3 make_ieee_docx.py ieee.md WARDEN_IEEE_IoTJ.docx`
(run `pdflatex main.tex` first: numbers for equations, figures, tables and sections are read
from `main.aux`). The Word file uses native, editable Word equations and the IEEE two-column
layout; wide figures sit in single-column sections. IEEE re-typesets accepted papers, so the
LaTeX PDF is the reference for the 8-page count.
