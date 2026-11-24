#!/bin/bash
# Run from the Mac: step-3 comparison, blocks x chunks, one short locked command per chunk.
# Thermal and LiDAR chunks alternate inside each block.
# usage: synthlab/harness/jobs/cmp_queue.sh [first_block] [last_block]
set -u
J="$(cd "$(dirname "$0")/../../.." && pwd)/tools/jetson.sh"
first="${1:-0}"; last="${2:-2}"
log() { echo "$(date +%H:%M:%S) $*"; }
nch() { msg=$("$J" run "bash ~/synthlab/harness/jobs/cmp_chunk.sh $1 list"); log "$1: $msg" >&2
        echo "$msg" | awk '{print $3}'; }
NT=$(nch thermal); NL=$(nch lidar)
for b in $(seq "$first" "$last"); do
  maxc=$(( NT > NL ? NT : NL ))
  for c in $(seq 0 $((maxc - 1))); do
    for kind in thermal lidar; do
      lim=$NT; [ "$kind" = lidar ] && lim=$NL
      [ "$c" -ge "$lim" ] && continue
      log "block $b $kind chunk $c"
      "$J" measured "bash ~/synthlab/harness/jobs/cmp_chunk.sh $kind $b $c"
    done
  done
done
log "cmp queue done"
