#!/bin/bash
# Run from the Mac: queue the step-2 (null-hypothesis) control rounds. The shared lock is busy,
# so two configurations share one locked command (~10 min). Rounds interleave configurations.
# Noise floor (noise_floor.sh pytorch|tensorrt) is queued separately.
# usage: synthlab/harness/jobs/null_queue.sh [first_round] [last_round] [skip_first_pt]
set -u
J="$(cd "$(dirname "$0")/../../.." && pwd)/tools/jetson.sh"
first="${1:-0}"; last="${2:-2}"; skip="${3:-}"
C=~/synthlab/harness/jobs/ctl_round.sh
log() { echo "$(date +%H:%M:%S) $*"; }
for r in $(seq "$first" "$last"); do
  if [ "$r" = "$first" ] && [ -n "$skip" ]; then a="bash $C pytorch unpaced $r nobie"
  else a="bash $C pytorch unpaced $r; bash $C pytorch unpaced $r nobie"; fi
  log "round $r: pytorch unpaced (+nobie)"; "$J" measured "$a" | tail -2
  log "round $r: tensorrt unpaced + pytorch paced"
  "$J" measured "bash $C tensorrt unpaced $r; bash $C pytorch paced $r" | tail -2
done
log "queue done"
