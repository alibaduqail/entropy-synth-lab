#!/bin/bash
# Step-3 unified comparison: one chunk of one randomised block (run under the lock).
# The bank list is frozen in <out>/banks.txt on first use so every block has every bank.
# Each block is one seeded shuffle of the full list, cut into chunks of <= PER banks.
# usage: cmp_chunk.sh thermal|lidar list            -> freeze list, print n_banks
#        cmp_chunk.sh thermal|lidar <block> <chunk> -> run that chunk
set -u
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
H=~/synthlab/harness; kind="$1"; O=${CMP_OUT:-~/synthlab/jetson-harness/runs/cmp}/$kind
PER=${PER:-12}
mkdir -p "$O"
if [ ! -s "$O/banks.txt" ]; then
  if [ "$kind" = thermal ]; then ls -d ~/synthlab/*/banks/*/frames.npy | xargs -n1 dirname
  else ls -d ~/synthlab/*/banks/*/frame_00000.npz | xargs -n1 dirname; fi | sort > "$O/banks.txt"
fi
n=$(wc -l < "$O/banks.txt")
if [ "$2" = list ]; then echo "$n banks, $(( (n + PER - 1) / PER )) chunks of <= $PER"; exit 0; fi
block="$2"; chunk="$3"
banks=$(python3 - "$O/banks.txt" "$block" "$chunk" "$PER" <<'EOF'
import sys, numpy as np
lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
block, chunk, per = map(int, sys.argv[2:5])
order = np.random.default_rng(9000 + block).permutation(len(lines))
print(" ".join(lines[i] for i in order[chunk * per:(chunk + 1) * per]))
EOF
)
[ -z "$banks" ] && { echo "empty chunk"; exit 0; }
if [ "$kind" = thermal ]; then
  extra="--backend pytorch --unpaced --seconds 10 --warmup 3 --idle-seconds 2"
else
  extra="--workload both --unpaced --seconds 15 --warmup 2 --idle-seconds 2"
fi
python3 $H/run_matrix.py --kind "$kind" --round "$block" --seed $((9100 + chunk)) \
    --out "$O/c$(printf %02d "$chunk")" --banks $banks -- $extra 2>&1 | grep -v -i warn | tail -3
