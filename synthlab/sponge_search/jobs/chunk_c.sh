#!/bin/bash
# Chunk C (~13 min in the lock): decouple (layout) and unconstrained-pixel (blocks) latency.
set -u
S=~/synthlab/sponge-search
cd $S/code
export PYTHONWARNINGS=ignore
python3 -m synthlab.sponge_search.search --run $S/runs/hil_decouple_layout --mode decouple --tier layout --budget-s 330 --k 16 --bank 2 --popsize 14 2>&1 | tail -2
python3 -m synthlab.sponge_search.search --run $S/runs/hil_latency_blocks --mode latency --tier blocks --budget-s 330 --k 8 --bank 1 --popsize 20 --diag 2>&1 | tail -2
