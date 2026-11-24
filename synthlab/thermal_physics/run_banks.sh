#!/bin/bash
# Measure thermal banks with the shared harness (run INSIDE `tools/jetson.sh measured`).
#   run_banks.sh <mode: unpaced|paced> <bank_dir>...   -> ~/synthlab/thermal-physics/runs/hb_<mode>_<bank>
# 30 s measured window, 5 s warmup, 2 repeats, 5 s idle (short trials: shared device).
set -u
MODE=$1; shift
H=~/synthlab/harness/run_thermal_bank.py
for B in "$@"; do
  N=$(basename "$B"); O=~/synthlab/thermal-physics/runs/hb_${MODE}_${N}
  [ -e "$O" ] && { echo "skip $N"; continue; }
  if [ "$MODE" = unpaced ]; then X=--unpaced; else X="--fps 8.7"; fi
  python3 $H --bank "$B" --out "$O" --seconds 30 --warmup 5 --repeats 2 --idle-seconds 5 $X > "$O.log" 2>&1 \
    && echo "ok $N" || echo "FAIL $N $(tail -2 $O.log)"
done
