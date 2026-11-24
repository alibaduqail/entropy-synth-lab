#!/bin/bash
# Chunk D (<= 16 min in the lock): build finalist banks, interleaved re-measurement vs baseline.
set -u
S=~/synthlab/sponge-search
cd $S/code
export PYTHONWARNINGS=ignore
mkdir -p $S/results/measure
for m in latency_layout energy_layout stealth_layout decouple_layout latency_blocks; do
  [ -f $S/runs/hil_$m/best_x.npy ] && python3 -m synthlab.sponge_search.make_bank --run $S/runs/hil_$m --name hil_$m | tail -1
done
B=$S/banks
PL="base=$S/real_bg/a0_bank.npy"
for m in latency_layout energy_layout stealth_layout decouple_layout; do PL="$PL hil_$m=$B/hil_$m/frames.npy"; done
python3 -m synthlab.sponge_search.measure --out $S/results/measure/finalists.json --rounds 16 --k 40 $PL 2>&1 | tail -1
python3 -m synthlab.sponge_search.measure --out $S/results/measure/upper.json --rounds 8 --k 8 base=$S/real_bg/a0_bank.npy hil_latency_blocks=$B/hil_latency_blocks/frames.npy wb_panel=$B/wb_panel/frames.npy wb_pixels=$B/wb_pixels/frames.npy 2>&1 | tail -1
python3 -m synthlab.sponge_search.measure --out $S/results/measure/paced.json --rounds 8 --k 40 --fps 8.7 $PL 2>&1 | tail -1
