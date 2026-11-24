# jetson-harness: shared measurement harness and cross-approach comparison

Branch `jetson-harness`. Code in `synthlab/harness/`, installed on the Jetson at
`~/synthlab/harness/`. Results in `results/jetson-harness/`. Every number below is taken from
a file in that folder (named next to the number).

## 1. What the harness does

| Script | Input | What runs on the Jetson | Output |
|---|---|---|---|
| `run_thermal_bank.py` | `frames.npy` uint16 (N,120,160) TLinear counts + `manifest.json` | TLinear→°C, fixed 10–50 °C → 8-bit, block image entropy (HT), YOLOv8n, NMS | `frames.csv`, `tegrastats.csv`, `summary.json` |
| `run_lidar_bank.py` | `frame_*.npz` 100 ms frames (`xyz`, `tag`) + `manifest.json` | 1 s windows → ROI → 5 cm voxels → radius kNN → eigenvalues → H_DS, H_eigen; optional Euclidean clustering | `windows.csv`, `tegrastats.csv`, `summary.json` |
| `run_matrix.py` | many banks | one randomised round over all banks, model loaded once | one trial folder per bank |
| `build_trt_engine.py` | ONNX | TensorRT FP16 engine for `--backend tensorrt` | `models/*.engine` |
| `make_control_banks.py` | recorded frames, earlier generator | control and real-anchor banks | `~/synthlab/jetson-harness/banks/` |

Usage (all timed runs hold the shared lock):

```
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_thermal_bank.py --bank <bank dir> --out <new dir> \
    [--seconds 60 --warmup 15 --fps 8.7 --unpaced --backend pytorch|tensorrt|none --no-bie --repeats N]'
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_lidar_bank.py --bank <bank dir> --out <new dir> \
    [--window 1.0 --workload entropy|cluster|both --seconds 60 --warmup 10 --unpaced --z-min -0.85 --roi ...]'
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_matrix.py --kind thermal|lidar --banks <dirs...> \
    --round R --seed S --out <dir> -- <options passed to the bank runner>'
```

The thermal functions are copied from the original project's `scripts/experimentA.py`
(`raw_to_celsius`, `radiometric_bie_image`, `block_image_entropy`, `yolo_frame_from_thermal`),
and YOLO is called through the same Ultralytics `predict()` call (`imgsz=640`, `conf=0.25`,
device 0). The LiDAR entropy stage re-implements the method in the LiDAR track's
`compute_hl.py` (same parameters: 5 cm leaf, 0.20 m radius, k_max 150, k_min 10, ROI
±6 m × ±6 m × −2…3 m, floor cut −0.85 m); the teammate's file is not in this repo.

### Per-frame columns (thermal)

`convert_ms` (TLinear→°C and 8-bit map), `bie_ms` (entropy stage), `prep_ms` (3-channel copy),
`yolo_pre_ms`, `yolo_inf_ms` (network forward, CUDA-synchronised), `yolo_post_ms` (NMS and
box rescale), `yolo_wall_ms` (whole `predict()` call), `yolo_extract_ms` (boxes to CPU),
`total_ms`, `total_no_bie_ms`, `HT`, `n_det`, `n_person`, `any_det`, `n_candidates`
(anchors above the confidence threshold before NMS; counted outside all timings), `max_conf`.

### What is measured and what is not

- Measured: processing of frames already in RAM, on the real Jetson, with the real code path.
- Not measured: camera or LiDAR acquisition, USB/network transport, rosbag decoding, disk I/O,
  preview windows, saving arrays.
- Power is `tegrastats` `VDD_IN`: module input power sampled every 200 ms. It is not wall
  power and no external meter was attached. Energy per frame is the trapezoid integral of
  `VDD_IN` over the measured window divided by frames processed; in paced mode this includes
  the idle time between frames.
- Each trial: 2–3 s idle (idle power), warmup (not recorded), then the measured window.
  `summary.json` has `ok`/`problems` from telemetry coverage checks (sample count ≥ 80 % of
  expected, no gap > 1 s, window fully covered, wall clock steady).
- Two pacing modes: paced at 8.7 fps (the Lepton rate; the GPU idles most of each period) and
  unpaced (back to back; the device is busy all the time).
- Two YOLO backends: PyTorch CUDA FP32 (what the original thermal pipeline uses) and a
  TensorRT 10.3 FP16 engine at the same 480×640 input (what the project Guide specifies).
- Device state recorded with every run (`run_config.json`): power mode MAXN_SUPER, CPU
  governor `schedutil`, GPU governor `nvhost_podgov`. Neither was changed (no sudo). `jtop`
  runs as a background service on the device and was left alone.

## 2. Null-hypothesis test (step 2)

### 2a. Noise floor (measured, `results/jetson-harness/step2_noise_floor.csv`, `fig_noise_floor.png`)

Same bank (`real_heatlamp_off`, real Lepton recording) repeated 8 times per configuration,
15 s measured window per trial, 4 s warmup, 2 s idle. n = 8 trials x 4 configurations, all 32
passed the telemetry checks. Values are mean / trial-to-trial SD.

| config | total ms/frame | YOLO inference ms | NMS/post ms | BIE ms | VDD_IN W | mJ/frame |
|---|---|---|---|---|---|---|
| PyTorch FP32, unpaced | 63.88 / 0.12 | 30.58 / 0.21 | 1.70 / 0.03 | 26.07 / 0.16 | 7.251 / 0.128 | 464.7 / 8.2 |
| PyTorch FP32, paced 8.7 fps | 73.78 / 0.39 | 30.82 / 0.03 | 2.07 / 0.06 | 34.17 / 0.34 | 5.812 / 0.059 | 668.1 / 6.8 |
| TensorRT FP16, unpaced | 42.45 / 0.12 | 9.64 / 0.01 | 1.71 / 0.03 | 25.85 / 0.06 | 6.502 / 0.063 | 277.1 / 3.3 |
| TensorRT FP16, paced 8.7 fps | 69.69 / 1.00 | 9.78 / 0.005 | 2.75 / 0.02 | 49.01 / 0.95 | 5.149 / 0.010 | 591.8 / 1.2 |

- Trial-to-trial latency CV is 0.2-1.4 %; power CV is 0.2-1.8 %. With 3 repeats per bank the
  minimum detectable difference (80 % power, alpha 0.05) is 0.28 ms and 0.29 W (PyTorch unpaced).
- VDD_IN is quantised at ~39.5 mW per step. The PyTorch-unpaced trial SD (128 mW) is ~3 steps,
  so trial means are not resolution-limited; TensorRT-paced SD (10 mW) is below one step, so
  differences under ~40 mW in that mode cannot be trusted from tegrastats alone.
- BIE (the entropy measurement itself) is 41 % (PyTorch unpaced) to 70 % (TensorRT paced) of
  per-frame time. In paced mode BIE is slower (34-49 ms vs 26 ms) because the CPU governor
  (`schedutil`) drops clocks in the idle gaps between frames.
- Chip temperature drifted < 1.4 C within a configuration; power-vs-temperature slope was
  0.002-0.35 W/C with R^2 <= 0.12 (not a dominant term at this drift).

### 2b-2e. Control banks, one randomised round per configuration

14 banks (4 real Lepton recordings, flat and noise controls, MVP composites with 0-16 objects),
10 s measured per bank, round 0 only for each of 4 configurations (rounds 1-2 were dropped
because the shared lock queue was ~1.5 h deep; the step-3 comparison repeats the PyTorch-unpaced
configuration over 3 blocks). n = 1 trial per bank per configuration; frame-level fits use
1218-1344 bank-frames. Files: `step2_spread_vs_noise.csv`, `step2_ht_regression.csv`,
`step2_stage_correlations.csv`, `fig_inference_vs_nms.png`, `fig_stage_breakdown.png`.

**(b) Inference vs NMS - confirmed with fresh data.**
- YOLO inference does not follow the input: TensorRT FP16 inference is 9.67 ms with 0.04 ms SD
  across all bank-frames (r with candidates 0.09). PyTorch inference sits on a few discrete levels
  (~18, 24, 27, 31 ms) that track GPU clock state (r with candidates -0.17 and 0.02 with BIE in the loop; 0.45 without BIE, where the clock level changed between banks of one round, so it is confounded with run order).
- NMS/postprocess follows the pre-NMS candidate count (r = 0.93 in every configuration):
  ~1.7 ms at 0 candidates, ~3.2 ms at 1, ~4.7 ms at 2-10, up to ~11 ms at ~40 candidates.
  Regression (PyTorch unpaced): +1.82 ms (SE 0.04) for any detection plus 0.16 ms (SE 0.003) per
  candidate. The audit's "~2 ms step from an early exit at zero candidates" is right about the
  zero-candidate shortcut, but the cost keeps rising with candidate count after that.
- HT alone predicts NMS time (31.8 ms per unit HT, R^2 0.22) only because HT and detections
  co-vary: with any-detection and candidate count in the model the HT slope is -0.78 ms
  (SE 0.46) per unit HT, i.e. zero. The same holds for total time without BIE under TensorRT
  (0.16 ms per unit HT, SE 0.58). Across these banks HT spans only 0.097 units.

**(c) PyTorch FP32 vs TensorRT FP16** (engine built without sudo from the extracted TensorRT 10.3
wheels; noise-floor bank, 8 repeats each): inference 30.58 -> 9.64 ms, total 63.88 -> 42.45 ms,
energy 465 -> 277 mJ/frame, VDD_IN 7.25 -> 6.50 W (unpaced).

**(d) Paced 8.7 fps vs unpaced** (PyTorch): 5.81 vs 7.25 W VDD_IN; 668 vs 465 mJ/frame (paced
energy includes idle time between frames); per-frame time is *longer* when paced (73.8 vs
63.9 ms) because CPU and GPU governors down-clock during the idle gaps. Across 14 banks paced
power spanned 69 mW (1.2 noise SD, under 2 VDD_IN steps).

**(e) With vs without the BIE stage in the measured loop** (PyTorch unpaced, 14-bank means):
total 64.4 -> 25.2 ms/frame, energy 462 -> 240 mJ/frame, but VDD_IN *rises* 7.16 -> 9.46 W and
YOLO inference *drops* 29.8 -> 18.1 ms. Removing the CPU-bound BIE stage keeps the GPU busy back
to back, so the GPU governor holds a higher clock. Inference latency therefore depends on duty
cycle, not on image content - a confound for any latency-vs-entropy claim made with BIE in
the loop.

**Bank-to-bank spread vs noise (PyTorch unpaced).** Across 14 banks, power spanned 0.127 W =
0.99 x trial noise SD (no detectable input effect); total latency spanned 1.15 ms (9 noise SD),
of which NMS spanned 2.98 ms (largest on `real_radiometric_A1`, which has people detected).

## 3. Unified comparison (step 3) - every bank from every track

**Design.** Bank list frozen at 01:07: 74 thermal banks (jetson-harness 18, thermal-datadriven 30,
thermal-physics 17, sponge-search 9; thermal-generative had none) and 29 LiDAR banks
(lidar-sim). Each block is a seeded shuffle of all banks; PyTorch FP32 unpaced, 8 s measured +
2 s warmup per thermal bank; LiDAR entropy + clustering, unpaced, 12 s per bank. Chip junction
temperature is a covariate. Model per metric: y = bank + block + b*temperature; the residual SD is
the in-situ noise floor; MDE = 2.8 x SD x sqrt(2/blocks). **Completed: block 0 for all banks,
block 1 for 40 of 74 thermal banks and all 29 LiDAR banks** (114 thermal + 58 LiDAR trials, all
passed telemetry checks). Block 2 and the rest of block 1 did not run: the shared lock queue was
20-35 min deep per command and the run stopped at 03:00 for the report deadline. So n_blocks is
1-2, not >= 3. Files: `cmp_thermal_banks.csv`, `cmp_lidar_banks.csv`, `cmp_verdicts.csv`,
`cmp_named_comparisons.csv`, `cmp_sweeps.csv`, `cmp_thermal_sponge_banks.csv`, `cmp_summary.json`,
figures `fig_cmp_thermal_cost_vs_HT.png`, `fig_cmp_thermal_cost_vs_HT_controlled.png`,
`fig_cmp_thermal_stages.png`, `fig_cmp_lidar_cost_vs_HL.png`.

In-situ noise (thermal, 65 non-adversarial banks): trial SD 1.10 ms for total time (larger than
the 0.12 ms same-bank floor, because PyTorch inference changes clock level between banks: 1.0 ms
SD in inference alone), 0.127 W VDD_IN (~3 VDD_IN steps of 39.5 mW), 10.4 mJ/frame. MDE between
two banks: 1.1 ms, 0.127 W, 10.4 mJ (n = 2 blocks). Junction temperature ranged 49.0-61.3 C (the
hot end follows the 942 ms/frame sponge bank).

### Thermal: cost vs HT (65 banks; sponge-search banks reported separately)

| metric | slope per unit HT [95 % CI] | effect over HT range 0.19 | MDE | slope after controlling any-det + candidates |
|---|---|---|---|---|
| total ms/frame | +8.3 [3.6, 14.8] | +1.6 ms (2.5 %) | 1.1 ms | +2.2 (SE 1.9) |
| NMS/post ms | +24.2 [11.7, 40.4] | +4.7 ms | 0.13 ms | +0.34 (SE 0.70) |
| YOLO inference ms | -17.0 [-30.5, -7.0] | -3.3 ms | 1.0 ms | +1.8 (SE 1.7) |
| VDD_IN W | +0.64 [0.31, 1.08] | +0.124 W | 0.127 W | +0.30 (SE 0.19) |
| mJ/frame | +98 [52, 158] | +19 mJ (4 %) | 10 mJ | +35 (SE 17) |

- Raw HT correlates with cost (r = 0.47 latency, 0.42 power), but the latency effect is NMS work
  on detections and candidates: once those are in the model, every HT slope is within ~1-2 SE of 0.
  Power over the whole HT range moves +0.12 W raw (about one MDE) and +0.06 W controlled.
- No knee: linear-vs-hinge BIC difference < 6 for every thermal metric (`cmp_verdicts.csv`).
- Per track: thermal-datadriven banks show the clearest raw slope (power +1.23 W/unit HT,
  CI 0.56-1.95; n = 30), thermal-physics none (+0.02 W/unit, CI -0.51-0.24; n = 17).

### Named comparisons asked for by the tracks (`cmp_named_comparisons.csv`, `cmp_sweeps.csv`)

- **Synthetic vs real A1** (thermal-datadriven `syn_A1` vs `real_A1_0..2`): cost is the same
  within noise - total -0.24 ms [-4.0, 3.5], VDD_IN -0.06 W [-0.15, 0.03], -5 mJ/frame [-35, 24] -
  but YOLO finds the person in 55 % of synthetic frames vs 96 % of real (diff -0.41 [-0.48,
  -0.34]). thermal-physics A1 vs real A1: 26 % vs 96 %, NMS -2.1 ms, VDD_IN -0.13 W [-0.22, -0.04].
  Synthetic A0 vs real A0: VDD_IN -0.15 W [-0.25, -0.04], latency equal.
- **Cost vs people count** (thermal-datadriven people_0..8, mean labelled 0-7.5): total
  +0.026 ms/person [-0.18, 0.30], VDD_IN +0.010 W/person [-0.009, 0.023] - flat; detections rise
  only +0.17/person [-0.23, 0.56].
- **Thermal fog** (thermal-physics B0..B3, one person): HT *falls* -0.0094/level [-0.016, -0.004];
  total +0.25 ms/level [0.08, 0.34] (0.74 ms over the sweep, below the 1.1 ms MDE); VDD_IN
  -0.009 W/level [-0.13, 0.10]; YOLO detected nobody in any fog bank.
- **Sponge-search banks** (adversarial, `cmp_thermal_sponge_banks.csv`): `wb_pixels` 942 ms/frame
  (3026 pre-NMS candidates, NMS 879 ms), 5.6 J/frame (12x baseline) while VDD_IN *drops* to
  5.95 W (CPU-bound NMS, GPU idle); `wb_panel` 137 ms, 0.89 J/frame. Physical-layout banks
  (`hil_*`, `proxy_*`) reach 63-73 ms, at most +9 ms.

### LiDAR: cost vs HL and fog (29 banks x 2 blocks)

- Fog sweep (`fog_a0p02` .. `fog_a3p00`, 13 banks): HL-stage time is an inverted U: 514 ms/window at
  alpha 0.02, 851 ms at alpha 0.45 (+66 %), back to 278 ms at alpha 3.0, following occupied voxels
  (15.7k -> 25.3k -> 7.5k). Knee/hinge fit preferred (BIC difference 20, knee at alpha ~0.36).
  Energy per window follows: 3.26 -> 5.36 -> 1.77 J. HL itself barely moves (0.755 -> 0.810 ->
  0.815); its slope against alpha has a CI including 0.
- VDD_IN is flat across all LiDAR banks (6.27-6.40 W, range 0.13 W, MDE 0.078 W): unpaced, the
  single-threaded entropy stage keeps one core saturated whatever the input, so extra work shows
  up as time and energy per window, not as watts. Paced 1 s windows (the real 10 Hz/1 s use) were
  not run in the comparison; there the busy fraction would rise with cost.
- Across all 29 banks, cost vs HL has no usable linear relation (r = 0.03); cost follows voxel
  count, which HL does not track.

### Verdicts (effect sizes from the files above)

- **Null (fixed compute graph => constant cost): supported for YOLO inference and for power.**
  TensorRT inference is constant to 0.04 ms; PyTorch inference varies only with GPU clock state.
  Thermal VDD_IN across 65 banks spans 0.29 W, and its HT slope is not distinguishable from zero
  after controlling for detections. Not supported for latency: NMS cost is input-dependent
  (+1.8 ms for any detection, +0.16 ms per candidate, up to 879 ms for an adversarial bank).
- **Hyp A (complexity raises HT/HL and power): weak/partial.** Raw thermal HT predicts +0.12 W and
  +1.6 ms over the full HT range (about one MDE), almost all through detections; people count
  0 -> 8 does not raise cost. LiDAR cost rises with scene points/voxels, not with HL.
- **Hyp B (fog raises HL, HT flat, power rises): partially supported, in time/energy not watts.**
  LiDAR fog up to alpha ~0.45 raises entropy-stage time and energy per window by ~65 %, while
  thermal HT falls slightly under fog and thermal cost is flat. HL rises only ~0.05. Instantaneous
  VDD_IN does not rise in unpaced runs.
- **Hyp C (knee): supported for LiDAR fog** (inverted U, knee near alpha 0.3-0.45), **not for
  thermal** (no hinge preferred).
- Caveats: n_blocks 1-2 (not 3); synthetic banks only for fog; VDD_IN is module input at 39.5 mW
  resolution, not wall power; BIE in the loop changes GPU clocks (section 2e), so latency numbers
  depend on pipeline composition.

