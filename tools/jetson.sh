#!/bin/bash
# Run a command on the Jetson Orin Nano over the USB-C link.
#   tools/jetson.sh run '<cmd>'       plain command (setup, file listing, generation)
#   tools/jetson.sh measured '<cmd>'  holds the exclusive measurement lock: use for ANY
#                                     timed / power-measured trial so trials never overlap
#   tools/jetson.sh heavy '<cmd>'     holds the same lock: use for long GPU/CPU jobs
#                                     (training, big renders) so they never pollute a measurement
#   tools/jetson.sh push <local> <remote>   rsync to Jetson
#   tools/jetson.sh pull <remote> <local>   rsync from Jetson
set -euo pipefail
HOST="${JETSON_HOST:-jetsonnano@192.168.55.1}"
KEY="$HOME/.ssh/id_ed25519_jetson"
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 -o BatchMode=yes)
LOCK=/home/jetsonnano/synthlab/.measure.lock
mode="$1"; shift
retry() { for i in 1 2 3 4; do "$@" && return 0; rc=$?; [ $rc -ne 255 ] && return $rc; sleep 4; done; return 255; }
case "$mode" in
  run)      retry "${SSH[@]}" "$HOST" "$1" ;;
  measured|heavy) retry "${SSH[@]}" "$HOST" "mkdir -p /home/jetsonnano/synthlab && flock $LOCK bash -lc $(printf '%q' "$1")" ;;
  push)     rsync -az -e "${SSH[*]}" "$1" "$HOST:$2" ;;
  pull)     rsync -az -e "${SSH[*]}" "$HOST:$1" "$2" ;;
  *) echo "usage: jetson.sh run|measured|heavy|push|pull ..." >&2; exit 2 ;;
esac
