#!/bin/bash
# Noise floor: one real bank repeated 8 times per configuration (run under the lock).
# usage: noise_floor.sh <backend> [bank]
set -u
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
H=~/synthlab/harness; R=~/synthlab/jetson-harness/runs/noise
backend="$1"; bank="${2:-$HOME/synthlab/jetson-harness/banks/real_heatlamp_off}"
mkdir -p "$R"
for mode in unpaced paced; do
  flag=""; [ "$mode" = unpaced ] && flag="--unpaced"
  python3 $H/run_thermal_bank.py --bank "$bank" --out "$R/${backend}_${mode}" --backend "$backend" \
      --repeats 8 --seconds 15 --warmup 4 --idle-seconds 2 $flag 2>&1 | grep -v -i warn
done
