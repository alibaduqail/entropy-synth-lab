#!/bin/bash
# Run from the Mac: queue the step-2 (null-hypothesis) runs, one short locked command each.
# Rounds are interleaved across configurations so slow drifts hit every config equally.
# usage: synthlab/harness/jobs/null_queue.sh [first_round] [last_round]
set -u
J="$(cd "$(dirname "$0")/../../.." && pwd)/tools/jetson.sh"
first="${1:-0}"; last="${2:-2}"
log() { echo "$(date +%H:%M:%S) $*"; }
log "noise floor tensorrt"
"$J" measured 'bash ~/synthlab/harness/jobs/noise_floor.sh tensorrt' | tail -4
for r in $(seq "$first" "$last"); do
  for cfg in "pytorch unpaced" "tensorrt unpaced" "pytorch unpaced nobie" "pytorch paced" "tensorrt paced"; do
    log "round $r: $cfg"
    set -- $cfg
    "$J" measured "bash ~/synthlab/harness/jobs/ctl_round.sh $1 $2 $r ${3:-}" | tail -2
  done
done
log "queue done"
