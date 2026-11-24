#!/bin/bash
# Stock COCO YOLOv8n vs FLIR fine-tuned YOLOv8n through the shared thermal harness.
# Run from the repo root on the Mac. Rounds alternate models; each round holds the lock.
set -euo pipefail
S=/home/jetsonnano/synthlab
STOCK=/home/jetsonnano/yolov8n.pt
THERMAL=$S/thermal-yolo/train/flir_v8n/weights/best.pt
OUT=$S/thermal-yolo/runs
REAL="$S/thermal-datadriven/banks/real_A0_0 $S/thermal-datadriven/banks/real_A0_1 $S/thermal-datadriven/banks/real_A0_2 $S/thermal-datadriven/banks/real_A1_0 $S/thermal-datadriven/banks/real_A1_1 $S/thermal-datadriven/banks/real_A1_2 $S/jetson-harness/banks/real_radiometric_A0 $S/jetson-harness/banks/real_radiometric_A1 $S/jetson-harness/banks/real_heatlamp_off $S/jetson-harness/banks/real_heatlamp_on"
SYN="$S/thermal-datadriven/banks/people_0 $S/thermal-datadriven/banks/people_1 $S/thermal-datadriven/banks/people_2 $S/thermal-datadriven/banks/people_4 $S/thermal-datadriven/banks/people_8 $S/thermal-datadriven/banks/syn_A1 $S/thermal-physics/banks/A1_one_person $S/thermal-physics/banks/A2_three_people $S/thermal-physics/banks/B3_fog3_person $S/thermal-generative/banks/tg_ddpm_A1_2people_s1 $S/sponge-search/banks/hil_latency_layout"
run() { # model_name weights mode round banks extra
  tools/jetson.sh measured "python3 $S/harness/run_matrix.py --kind thermal --banks $5 --round $4 --seed 2026 --out $OUT/$3/$1 --max-minutes 18 -- --seconds $6 --warmup 5 $7 --model $2" | tail -1
}
for r in 0 1; do
  if [ $r = 0 ]; then order="stock thermal"; else order="thermal stock"; fi
  for m in $order; do
    w=$STOCK; [ $m = thermal ] && w=$THERMAL
    run $m $w unpaced $r "$REAL $SYN" 12 --unpaced
  done
done
for m in stock thermal; do
  w=$STOCK; [ $m = thermal ] && w=$THERMAL
  run $m $w paced 0 "$REAL" 20 ""
done
