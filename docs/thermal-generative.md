# thermal-generative: a learned generative model of Lepton 3.5 frames, trained on the Jetson

Status: DONE for tonight. Every number below comes from a file in `results/thermal-generative/` (metrics JSON, speed JSON, `logs/`, `harness_runs.csv`, `metric_table.md`). Figures are in `results/thermal-generative/figures/`. Checkpoints stay on the Jetson (`~/synthlab/thermal-generative/ckpt/`), not in git.

## What this track builds
- `synthlab/thermal_generative/mapping.py` - fixed, invertible TLinear-counts <-> model-space map
  (affine over the pipeline's 10-50 C range, log-compressed above 50 C for the heat lamp), and exact
  ports of the real pipeline's 8-bit image and block image entropy (HT). The HT port reproduces the
  HT values logged in the real `trial_data.csv` to 4e-9 (300 random frames, measured on the Jetson).
- `data.py` - index of the real radiometric recordings, trial-level labels and whole-trial hold-out.
- `models.py`, `diffusion.py` - patch-2 conditional U-Net, cosine-schedule v-prediction DDPM, DDIM
  sampler with classifier-free guidance on person / lamp / session; conditional VAE baseline.
- `train.py`, `run_chunks.sh` - time-boxed resumable training under the shared Jetson lock, with
  tegrastats power logging.
- `sample.py`, `banks.py` - banks in the hand-off format; `measure.py` - runs them through the shared harness.
- `evaluate.py`, `metrics.py`, `plots.py` - held-out evaluation.

## Data (measured on the Jetson pack, every 3rd frame of each recording)
| session | person | lamp | train trials / frames | held-out trials / frames | what is in view |
|---|---|---|---|---|---|
| heat_lamp (8 Oct) | 0 | 0 | 20 / 3447 | 5 / 845 | empty room |
| heat_lamp (8 Oct) | 1 | 1 | 20 / 3451 | 5 / 867 | one standing person holding the heat lamp |
| warmup120 | 0 | 0 | 4 / 1376 | 1 / 344 | empty, darker (cooler) scene |
| warmup120 | 1 | 0 | 4 / 1377 | 1 / 344 | TWO seated people, close to the camera |
| radiometric | 0 | 0 | 2 / 346 | 1 / 169 | empty room, different view |
| radiometric | 1 | 0 | 2 / 337 | 1 / 173 | one standing person |

13,076 frames, 71 trials, 3 camera set-ups. Hold-out is by whole trial (every 5th run of the 25-run
groups, the last run of the small groups). Labels are trial-level (A0/A1, lamp folder), not per-frame.

Design facts that limit any model trained on this:
- Lamp and person are perfectly confounded: the lamp is hand-held by the person and "lamp on, no
  person" was never recorded. "Person, lamp off" exists only in the two other camera set-ups.
- Only three scenes and two or three people. There is no fog, no crowd, no clutter sweep.
- "person=1" means one person in two sessions and two people in the third.

## Training (measured on the Jetson, `logs/*_log.jsonl`, `logs/*_power.csv`)
| model | params | steps | train wall time | mean VDD_IN while training | energy (mean W x s) | held-out loss (train / test) |
|---|---|---|---|---|---|---|
| DDPM (patch-2 U-Net, base 48, v-pred, cosine, batch 32, fp16 AMP) | 4.0 M | 8373 (3 chunks of 17-18 min) | 3180 s (53 min) | 18.3-18.8 W | about 59 kJ (16 Wh) | 0.045 / 0.057 |
| CVAE (batch 64) | 10.4 M | 2732 (1 chunk, 10 min) | 600 s | 18.7 W | about 11 kJ | 0.007 / 0.054 (recon) |

This is under the planned budget (90 min DDPM, 20 min CVAE): the shared lock queue delayed the training chunks by 80+ minutes
each, and a first attempt was OOM-killed (the full-pack float64 mapping used about 6 GB; fixed by converting in chunks).
DDPM loss was still falling slowly at the last chunk (0.0668, 0.0590, 0.0565 test), so it is undertrained, not converged.

## Held-out evaluation (whole trials held out, n = 256 real vs 128 generated per cell; `metric_table.md`)
Cell = (session, person, lamp). "NN" = median nearest-neighbour RMSE (C) to the training set.
- **Histogram / HT**: DDPM mean HT is within 2-30 percent of the real held-out cell mean (e.g. warmup120 two people 0.1038 vs 0.1040; heat_lamp person+lamp 0.0757 vs 0.0765; empty heat_lamp 0.0121 vs 0.0145, radiometric empty 0.0521 vs 0.0567). HT KS distance gen-vs-test is 0.11-0.84; the real train-vs-test KS is 0.15-0.84 in the same cells, so the HT *distribution* is about as close as a different real trial of the same scene, except the empty-room cells (0.71 vs 0.48). CVAE HT means are off in the empty cells (0.0095 vs 0.0145, 0.0250 vs 0.0155) and its KS is 0.79-1.00 in 5 of 6 cells.
- **Per-pixel noise (the weak point)**: median temporal-like noise proxy `noise_mk_median` is 52-77 mK for DDPM vs 79-116 mK real (about 60-75 percent of real); the CVAE gives 7-25 mK (smooth, 5-10x too clean). Median pixel s.d. is also too low for DDPM (0.16-0.40 C vs 0.50-1.02 C real, one cell reversed). Frames are too smooth; HT is nevertheless close because BIE is dominated by large-scale scene content.
- **Radial power spectrum**: see `figures/quality_metrics.png` and `spec_gen_vs_test` in the metrics JSON; the DDPM high-frequency tail is below real, consistent with the noise result. I did not tabulate it further.
- **Memorisation**: DDPM gen-to-train NN RMSE is about the same as held-out-to-train (empty cells 0.13/0.13/0.19 vs 0.14/0.12/0.19; person cells 1.04 vs 1.02, 0.75 vs 1.14, 0.82 vs 0.67) and `nn_gen_below_p5_of_test` was 0.0 in the cell inspected. **No evidence of memorisation** at this training length; with 8k steps over 13k frames this says little about longer training. The CVAE shows larger distances (0.4-0.9 C) because it blurs.
- **Real-vs-generated classifier** (small CNN): for the DDPM it collapsed to predicting "real" for everything (accuracy 0.64 = real share, gen recall 0.0) in all 3 seeds, so it is uninformative, not proof of indistinguishability; for the CVAE it separates easily in some seeds (acc 0.68-0.82).
- **Condition fidelity**: a person classifier trained on real data gets 96.6 percent on held-out real and 100 percent on DDPM samples (CVAE 100 percent); session 96.1 / 98.7 percent (CVAE 83 percent). Person-frame rate by YOLO on samples versus real: two-people scene 1.00 / 1.00; **one-person scenes 0.80 real vs 0.08 DDPM (heat_lamp), 1.00 vs 0.16 (radiometric)** and 0.00 for the CVAE: generated single people are often not recognised by YOLOv8n (distorted, partial bodies; see grid). The label is respected statistically (warm fraction rises), but the people are not realistic enough for the detector.
- **Unseen combinations** (lamp without person, person without lamp, never recorded): lamp-hot fraction was 0.0000 for both models in every cell: neither can show a lamp without a person. "Person without lamp" produced a person-frame rate of 0.23 in heat_lamp session for DDPM. That is a wrong or unverifiable extrapolation, not evidence of controllability, because the lamp and person are confounded in the training data.
- **Sampling speed** (`ddpm_main_speed.json`, batch 32): 22.97 frames/s at 10 DDIM steps, 11.5 at 20, 4.60 at 50, 2.30 at 100 (guidance 1.0; guidance 2.0 halves it); mean VDD_IN 18.7-19.5 W, so 0.8 J/frame at 10 steps and 4.2 J/frame at 50 steps. CVAE: 805 frames/s, 0.019 J/frame. A 128-frame DDPM bank at 50 steps takes about 28 s.

## Banks measured through the real pipeline (shared harness, PyTorch YOLOv8n, paced 8.7 fps, 20 s x 2 repeats, 174 frames each, `harness_runs.csv`)
Mean over the 2 repeats.
| bank | HT | total ms/frame | VDD_IN W | detections/frame |
|---|---|---|---|---|
| real A0 anchor (datadriven) | 0.0155 | 73.7 | 5.794 | 0.0 |
| real A1 anchor (datadriven) | 0.0837 | 76.5 | 5.845 | 1.0 |
| real held-out A0 lamp off | 0.0142 | 73.9 | 5.789 | 0.0 |
| real held-out A1 lamp on | 0.0748 | 76.0 | 5.875 | 0.98 |
| DDPM A0 lamp off | 0.0121 | 73.9 | 5.789 | 0.0 |
| DDPM A1 lamp on | 0.0752 | 74.1 | 5.836 | 0.11 |
| DDPM A0 (warmup session) | 0.0153 | 73.9 | 5.793 | 0.0 |
| DDPM A1 two people | 0.1028 | 76.9 | 5.888 | 2.01 |
| DDPM unseen A0 lamp on | 0.0185 | 73.4 | 5.804 | 0.0 |
| DDPM unseen A1 lamp off | 0.0616 | 74.0 | 5.798 | 0.18 |
| CVAE A0 | 0.0094 | 74.2 | 5.779 | 0.0 |
| CVAE A1 lamp on | 0.0801 | 73.7 | 5.802 | 0.0 |

YOLO inference time is flat (30.80-30.92 ms for all 24 runs). The per-frame time differences come from NMS / post-processing (about 2 ms with no detections, 5-7 ms with detections) and BIE (about 34 ms in all banks). Across the 12 banks, r(HT, total ms) = 0.73 / 0.64 and r(HT, VDD_IN) = 0.63 / 0.83 in the two repeats (`figures/cost_vs_ht.png`), but r(HT, YOLO inference) = 0.21 / 0.29. The power spread (5.75-5.90 W) is about 2.5 percent, comparable to the idle-power drift between repeats (4.77-5.14 W idle), so the power correlation is weak evidence. Paced at 8.7 fps these measure the bank, not the generator.

## Verdict (candid)
What this approach CAN give the study:
- A fast way to make many frames at chosen HT. The DDPM reproduces the scene-level statistics that drive HT (mean HT within a few percent for the occupied cells, banks with HT 0.012-0.103 spanning the real range), so frames from it cost the same through the pipeline as real frames of similar HT (A0-like 73.9 vs 73.7-73.9 ms; two-people 76.9 ms vs real A1 76.0-76.5 ms).
- A learned, data-driven generator is cheap to run after training (4.6 frames/s at 50 steps; 23 at 10 steps) and trained in under an hour on the Orin Nano at about 18.5 W.

What it CANNOT give:
- **Nothing beyond the recorded manifold.** Only 3 scenes, 71 trials, 2-3 people, no fog, no crowd, lamp confounded with person. It cannot produce lamp-only, more than two people, or fog (B conditions). The unseen-combination banks show this directly (lamp-hot fraction 0 in every cell).
- Not a test of hypothesis A beyond what a replayed real frame already tests: the generated banks inherit the real trials' HT range and its confounds. Hypothesis B (fog, HT flat) and C (knee over a sweep) cannot be tested because no controllable sweep exists in this model.
- Realism is insufficient for the detector where it matters: one-person samples get YOLO person-frame rates of 0.08-0.16 against 0.80-1.00 real, and noise is 25-40 percent too low; the CVAE is far worse (blurred, 5-10x too clean). For a sponge-style test, the detection count is what moves latency (NMS about 2 ms to 7 ms), so DDPM banks mis-state the cost in A1 conditions (74.1 ms vs 76.0 ms, 0.11 vs 0.98 detections per frame lamp-on).
- Honest null on the main question: in these 24 harness runs YOLO inference does not change with HT or content (30.8-30.9 ms); only post-processing and BIE-adjacent costs move, and by a few ms. A generative model cannot change that.

Unverified / limits: single seeds; DDPM undertrained (loss still falling, 53 min); held-out set has 6 trials in 3 sessions; the radial spectrum is shown in a figure, not tabulated; the real-vs-generated classifier is uninformative for the DDPM; each harness bank had only 174 frames per repeat with n = 2 repeats; power resolution is about 2 percent of the signal.

Reproduce: `synthlab/thermal_generative/run_chunks.sh` (training), `job_eval.sh` (evaluation), `job_banks.sh` (banks and harness runs), `plots.py` / `report.py` (figures); all run through `tools/jetson.sh heavy|measured`.
