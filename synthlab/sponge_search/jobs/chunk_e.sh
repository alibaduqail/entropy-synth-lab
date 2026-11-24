#!/bin/bash
# Chunk E (~10 min): which pipeline settings cap the sponge (max_det, conf) on the strongest banks.
set -u
S=~/synthlab/sponge-search
cd $S/code
export PYTHONWARNINGS=ignore
B=$S/banks
for v in "max_det=300" "max_det=50" "max_det=10" "conf=0.5"; do
python3 -m synthlab.sponge_search.measure --out $S/results/measure/variant_${v/=/_}.json --rounds 6 --k 8 --variant $v base=$S/real_bg/a0_bank.npy wb_panel=$B/wb_panel/frames.npy hil_latency_blocks=$B/hil_latency_blocks/frames.npy wb_pixels=$B/wb_pixels/frames.npy 2>&1 | tail -1
done
