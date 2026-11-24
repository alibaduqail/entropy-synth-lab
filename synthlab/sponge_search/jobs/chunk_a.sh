#!/bin/bash
# Chunk A (<= 15 min in the lock): proxy searches, first hardware-in-the-loop latency search,
# and a first interleaved measurement of the white-box banks.
set -u
S=~/synthlab/sponge-search
cd $S/code
export PYTHONWARNINGS=ignore
python3 -m synthlab.sponge_search.search --run $S/runs/proxy_det_layout --mode det --tier layout --budget-s 200 --popsize 24 --bank 2 2>&1 | tail -3
python3 -m synthlab.sponge_search.search --run $S/runs/proxy_cand_panel --mode cand --tier panel --budget-s 120 --popsize 24 --bank 2 --diag 2>&1 | tail -3
python3 -m synthlab.sponge_search.make_bank --run $S/runs/proxy_det_layout --name proxy_det_layout
python3 -m synthlab.sponge_search.make_bank --run $S/runs/proxy_cand_panel --name proxy_cand_panel
python3 -m synthlab.sponge_search.measure --out $S/results/first_look.json --rounds 5 --k 30 \
  base=$S/real_bg/a0_bank.npy wb_pixels=$S/banks/wb_pixels/frames.npy wb_panel=$S/banks/wb_panel/frames.npy \
  proxy_layout=$S/banks/proxy_det_layout/frames.npy proxy_panel=$S/banks/proxy_cand_panel/frames.npy 2>&1 | tail -5
python3 -m synthlab.sponge_search.search --run $S/runs/hil_latency_layout --mode latency --tier layout --budget-s 300 --k 16 --bank 2 --popsize 14 2>&1 | tail -4
