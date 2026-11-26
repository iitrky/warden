#!/usr/bin/env bash
# Reproduce every table and figure of the paper (about 2.5 h on 2 CPU cores, ~3 GB RAM).
# Seeds run sequentially on purpose: two parallel runs need ~6 GB RAM.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python -m pytest -q tests
for s in 1 2 3; do
  python scripts/run_experiments.py --seed "$s" | tee "results/log$s.txt"     # base WARDEN, baselines, OOD, ablations
  python scripts/run_v2.py          --seed "$s" | tee "results/v2_log$s.txt"  # WARDEN-PI, WARDEN-U, prior misspecification
  python scripts/run_lowdata.py     --seed "$s" | tee "results/low_log$s.txt" # commissioning with 2/4/8 days
done
python scripts/make_figures.py      # Figs. 3-5, 8-11 and results/tables.md
python scripts/make_figures_v2.py   # Figs. 6-7 and results/tables_lowdata.md
python scripts/show_v2.py > results/tables_v2.txt
python scripts/fig_system.py        # Figs. 1-2 (needs Graphviz)
echo "done: see results/ and figures/"
python3 scripts/fig_ieee.py          # IEEE IoT-J column-width figures
