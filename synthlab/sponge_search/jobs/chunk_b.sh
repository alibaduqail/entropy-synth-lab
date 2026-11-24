#!/bin/bash
# Chunk B (~13 min in the lock): HIL energy and stealth searches, plausible (layout) tier.
set -u
S=~/synthlab/sponge-search
cd $S/code
export PYTHONWARNINGS=ignore
for m in energy stealth; do
python3 -m synthlab.sponge_search.search --run $S/runs/hil_${m}_layout --mode $m --tier layout --budget-s 330 --k 16 --bank 2 --popsize 14 2>&1 | tail -2
done
