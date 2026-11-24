#!/bin/bash
# one locked job: write the lean bank set, add real anchors, measure 2 x 20 s each via the shared harness.
cd ~/synthlab/thermal-generative/code; H=~/synthlab/thermal-generative
M="python3 -W ignore -m synthlab.thermal_generative"
$M.banks --minutes 10 --only tg_real_A0_lampoff_s0,tg_real_A1_lampon_s0,tg_ddpm_A0_lampoff_s0,tg_ddpm_A1_lampon_s0,tg_ddpm_A0_s1,tg_ddpm_A1_2people_s1,tg_ddpm_UNSEEN,tg_vae_A0,tg_vae_A1_lampon 2>&1 | tail -3
for a in A0 A1; do mkdir -p $H/banks/tg_anchor_real_${a}_0; cp ~/synthlab/thermal-datadriven/banks/real_${a}_0/* $H/banks/tg_anchor_real_${a}_0/; done
for r in 0 1; do $M.measure run --rep $r --seconds 20 --warmup 5 --minutes 14 2>&1 | tail -3; done
$M.measure collect 2>&1 | tail -1
touch $H/out/BANKS_DONE
