#!/bin/bash
# one locked job: sample eval sets, metrics, speed (all time-boxed). Run via tools/jetson.sh heavy.
cd ~/synthlab/thermal-generative/code; mkdir -p ../out/eval ../out/fig
M="python3 -W ignore -m synthlab.thermal_generative"
$M.evaluate gen --ckpt ddpm_main --minutes 9 2>&1 | tail -3
$M.evaluate gen --ckpt vae_main --minutes 2 2>&1 | tail -2
$M.evaluate metrics --ckpt ddpm_main 2>&1 | tail -3
$M.evaluate metrics --ckpt vae_main 2>&1 | tail -3
$M.evaluate speed --ckpt ddpm_main 2>&1 | tail -3
$M.evaluate speed --ckpt vae_main 2>&1 | tail -1
touch ../out/EVAL_DONE
