#!/bin/bash
# Train in N locked chunks so other tracks' measured trials can interleave on the shared Jetson.
#   synthlab/thermal_generative/run_chunks.sh <n_chunks> <minutes_per_chunk> <train args...>
# Each chunk takes the shared flock via tools/jetson.sh heavy, trains, checkpoints, releases.
set -u
cd "$(dirname "$0")/../.."
n="$1"; minutes="$2"; shift 2
for i in $(seq 1 "$n"); do
  echo "== chunk $i/$n $(date)"
  tools/jetson.sh heavy "cd ~/synthlab/thermal-generative/code && python3 -m synthlab.thermal_generative.train --minutes $minutes $* 2>&1 | grep -v -i warn | tail -4" || echo "chunk $i failed rc=$?"
done
echo "== done $(date)"
