#!/bin/bash
# One randomised round over the control banks (run under the lock).
# usage: ctl_round.sh <backend> <paced|unpaced> <round> [nobie]
set -u
export PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
H=~/synthlab/harness; B=~/synthlab/jetson-harness/banks
backend="$1"; mode="$2"; round="$3"; nobie="${4:-}"
name="${backend}_${mode}"; extra=""
[ "$mode" = unpaced ] && extra="$extra --unpaced"
[ "$nobie" = nobie ] && { extra="$extra --no-bie"; name="${name}_nobie"; }
python3 $H/run_matrix.py --kind thermal --round "$round" --seed 2026 \
    --out ~/synthlab/jetson-harness/runs/ctl/$name \
    --banks $B/real_* $B/ctl_flat $B/ctl_noise_2K $B/ctl_noise_8K $B/mvp_objects_* $B/mvp_noise_16 \
    -- --backend "$backend" --seconds 10 --warmup 3 --idle-seconds 2 $extra 2>&1 | grep -v -i warn
