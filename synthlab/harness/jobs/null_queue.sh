#!/bin/bash
# Run from the Mac: queue the step-2 (null-hypothesis) control rounds. The shared lock is busy,
# so one locked command runs all four configurations of one round (~15 min).
# A configuration whose round already exists fails fast and the rest continue.
# usage: synthlab/harness/jobs/null_queue.sh [first_round] [last_round]
set -u
J="$(cd "$(dirname "$0")/../../.." && pwd)/tools/jetson.sh"
first="${1:-0}"; last="${2:-2}"
log() { echo "$(date +%H:%M:%S) $*"; }
for r in $(seq "$first" "$last"); do
  log "round $r: pytorch unpaced, nobie, tensorrt unpaced, pytorch paced"
  "$J" measured "C=\$HOME/synthlab/harness/jobs/ctl_round.sh; bash \$C pytorch unpaced $r; bash \$C pytorch unpaced $r nobie; bash \$C tensorrt unpaced $r; bash \$C pytorch paced $r" | grep "round"
done
log "queue done"
