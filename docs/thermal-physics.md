# Thermal track: physically-based Lepton 3.5 simulator (`thermal-physics`)

Branch `thermal-physics`. Code `synthlab/thermal_physics/`, tests `tests/test_thermal_physics.py`,
numbers + figures `results/thermal-physics/`. Every number below comes from a file in that folder
(file named in brackets). "Real" = Ali's Lepton 3.5 recordings under the original project's `data/`
(read only); "sim" = this simulator.

## 1. What it is

Scene -> 8-14 um band radiance at the aperture -> Lepton 3.5 sensor model -> uint16 TLinear frames.

| Stage | Status | Detail |
|---|---|---|
| Planck band radiance 8-14 um, T<->L LUT | physics | flat spectral response ASSUMED (camera reports black-body-equivalent T, so only the shape of T<->L matters) |
| Geometry | room `scene/room_v1.json` from the room-3d track (photo-guided, +-0.3 m; camera pose ASSUMED) | ray caster: boxes, ellipsoids, capsules (numpy or torch/CUDA) |
| Surface emission / reflection | physics | `L = e(th) B(T) + (1-e(th)) [(1-s) B(T_amb) + s L_spec]`, Schlick grazing roll-off, single specular bounce |
| Material temps / emissivities | ASSUMED | walls ~air temp, skin 33-34 C, clothing 28-31 C, emissivity 0.90-0.98 |
| Person | ASSUMED shape | articulated capsules/ellipsoids, poses standing/walking/seated/arms-up |
| Heat lamp | ASSUMED | hot housing + steady-state warm patch on the surfaces it faces |
| Fog | physics + ASSUMED constants | Beer-Lambert `L = tau L_scene + (1-tau) B(T_mist)`, `tau = exp(-beta d)`; beta = 0.14 m^2/g x LWC (literature 0.13-0.15); optional mist cooler than air, optional turbulent plume |
| Lens PSF | MEASURED from real edges: sigma 0.50 px | literature/datasheet-derived value 0.74-0.79 px (research track) - our edge estimate is sharper; not resolved |
| Temporal noise (pixel/row/column/flicker, AR(1) rho 0.89) | MEASURED from 5 flat-wall runs | `real_calibration.json` |
| Fixed-pattern noise, low-order non-uniformity | MEASURED (flat wall) | may partly be the wall itself |
| Global drift, FFC every ~180 s with 1.8 s gap and post-FFC bump (~165 counts = 1.65 K) | MEASURED (FFC interval from datasheet) | |
| Output lattice 2.25 counts + dead-band hold (63 % pixels unchanged frame to frame) | MEASURED / TUNED | |
| Absolute radiometric accuracy | not modelled (datasheet +-5 C) | |

HT is computed with the exact BIE code path of the original `scripts/experimentA.py`
(fixed 10-50 C -> 8 bit, 15 px blocks) (`synthlab/thermal_physics/bie.py`, test-checked).

## 2. Realism check: real vs simulated  [`validation.json`, `contact_sheet_real_vs_sim.png`, `validation_hist_psd_ht.png`, `validation_temporal_psd.png`]

The real recordings were not taken from the room_v1 camera pose, so each real group was matched with a
hand-set view preset (air temperature, distance, pose set by eye once, not fitted to the metrics).
n = 1031 frames per side for the static A0 groups; other groups use their full recordings.

| Condition | mean C real / sim | frame spatial SD K real / sim | histogram W1 (K) | radial PSD rms log10 ratio | HT real / sim (ratio) |
|---|---|---|---|---|---|
| flat wall A0 | 21.79 / 21.88 | 0.28 / 0.23 | 0.14 (real-vs-real 0.10) | 0.33 | 0.0153 / 0.0120 (0.79) |
| empty room A0 | 24.89 / 24.97 | 0.20 / 0.32 | 0.10 (real-vs-real 0.08) | 0.41 | 0.0132 / 0.0198 (1.49) |
| two seated A1 | 24.48 / 25.51 | 3.13 / 4.14 | 1.24 | 0.25 | 0.0985 / 0.1165 (1.18) |
| standing person A1 | 27.59 / 26.52 | 1.43 / 1.41 | 1.08 | 0.21 | 0.0724 / 0.0631 (0.87) |
| person holding lamp A1 (lamp on) | 25.06 / 24.72 | 1.74 / 1.43 | 0.39 | 0.60 | 0.0703 / 0.0545 (0.78) |

Temporal noise (static scenes, NEdT from frame differences): real 24.0 / 23.6 mK vs sim 25.1 / 25.0 mK
(flat wall / empty room); pixels unchanged frame-to-frame real 63 % vs sim 64 %; AR(1) rho real 0.90 vs sim 0.86.

Where it does NOT match (plain):
- **HT is within a factor 0.78-1.49 of the real value, not equal.** The A0 < A1 ordering and the size of the
  people step (HT x5-7) are reproduced; absolute HT per condition is off by up to 49 % (empty room).
- **Empty room has too much large-scale structure in sim** (spatial SD 0.32 vs 0.20 K, a strong wall/door edge
  the real view lacks) because the geometry/pose is not the real camera view. Real empty room shows
  a small hot spot (equipment) the sim lacks.
- **Noise spatial correlation is too low in sim** for the empty-room recording (neighbour corr 0.26 vs 0.57-0.62);
  the flat-wall value (0.32-0.42) was the calibration target.
- **People look like mannequins**: uniform clothing, no hair/folds, faces brighter and more uniform than real
  (two-seated sim is 1 K warmer on average and has 33 % more spatial SD).
- **Hand-held lamp is missing a visible hot housing in sim** (real max 39.7 C vs sim 33.6 C) -> lamp-on HT is
  22 % low and the radial spectrum is the worst match (0.60 log10 rms).
- PSF: measured sigma 0.50 px is sharper than the 0.74-0.79 px literature value; not reconciled.

## 3. Fog: how flat is the thermal "control"?  [`fog_sweep.json`, `fog_levels_summary.json`, `ht_vs_fog.png`]

Physics: fog makes every pixel a mix of the scene and the mist temperature, so contrast can only shrink
(unless the mist itself is structured). Study fog levels B1-B3 = beta 0.07 / 0.21 / 0.56 per m
(LWC 0.5 / 1.5 / 4.0 g/m^3, ASSUMED k = 0.14 m^2/g). 24 frames per configuration, seed 7, room_v1.

| Level | tau (mean over image) | HT one person (change vs B0) | plain Shannon, fixed 10-50 C scale | Shannon, auto-stretched | HT empty room (change) |
|---|---|---|---|---|---|
| B0 | 1.00 | 0.0670 | 4.43 bits | 6.20 bits | 0.0162 |
| B1 | 0.90 | 0.0641 (-4.5 %) | -1.0 % | +0.5 % | -2.8 % |
| B2 | 0.74 | 0.0574 (-14.4 %) | -1.6 % | +2.3 % | -7.0 % |
| B3 | 0.46 | 0.0443 (-34.0 %) | -2.6 % | +7.0 % | -9.7 % |

(mist 1 K cooler than air; with mist = air: person HT -4.0 / -12.4 / -31.5 %, empty -3.3 / -9.0 / -13.8 %.)
A turbulent mist plume (mist -1 K) keeps empty-room HT nearly flat (0.0141-0.0152 for beta 0.035-2,
vs 0.0162 without fog; at beta 0.56 it is 4 % above uniform fog), i.e. structured fog partly cancels the
contrast loss in an empty scene; with a person the plume does not help (-38 % at beta 0.56).

Meaning for Hypothesis B: in this physics, **HT is not flat in fog; it falls**, by 4 % (B1), 14 % (B2) and 34 %
(B3) for one person. Plain global Shannon entropy on the fixed scale is much flatter (-1 to -3 %), and on an
auto-stretched image it even rises (+0.5 to +7 %) because the stretch amplifies noise. So "thermal is the
control" holds for light fog only and depends on which entropy is used; for B2-B3 the thermal channel
moves in the opposite direction to LiDAR (whose 905 nm attenuation is ~5x stronger, research track).
This is a simulated prediction; it has never been checked against a real fogged Lepton recording.
Caveat: the room is 2.6 m, so path lengths are short; a larger space would lose contrast faster.

## 4. Jetson: renderer speed, banks and cost

### Renderer speed on the Jetson  [`renderer_bench_jetson.json`, n = 30 frames per row, run under the measurement lock]

frames/s including the sensor model (render + sensor), supersample 2 (default) / 1:

| Condition | numpy CPU | torch CUDA |
|---|---|---|
| A0 empty | 44.7 / 147 | 87.1 / 116 |
| A1 one person | 1.8 / 7.9 | 14.6 / 24.0 |
| A2 three people | 0.7 / 2.7 | 9.5 / 11.6 |
| A3 people + warm objects | 0.9 / 3.8 | 10.0 / 12.5 |
| B2 fog + person | 1.9 / 8.2 | 19.1 / 22.9 |
| C16 16 warm objects (static) | 80.0 / 221 | 87.1 / 116 |

A repeat run (n = 24, `renderer_bench_jetson_repeat.json`) agrees within 5 % for every torch row.
With torch CUDA every condition renders faster than the Lepton's 8.7 fps, so the simulator could feed
the pipeline live; static scenes are cached (only the sensor model runs per frame). The banks below were
pre-rendered on the Mac (numpy, `bank_generation_summary_mac_numpy.json`), so rendering is NOT part of
the measured pipeline cost.

### Banks  [`bank_generation_summary_mac_numpy.json`, `refbank_manifests.json`]
On the Jetson under `~/synthlab/thermal-physics/banks/<name>/` (frames.npy uint16 (128,120,160) TLinear +
manifest.json with per-frame labels), scene room_v1, seed 20261123, supersample 2, sensor noise on.
Real reference banks (128 frames evenly subsampled from each real recording) under
`~/synthlab/thermal-physics/refbanks/real_*`.

| Bank | HT (mean of 32 frames) | mean / max C |
|---|---|---|
| A0 empty | 0.0166 | 22.7 / 24.2 |
| A1 one person (walking) | 0.0916 | 24.7 / 33.8 |
| A2 three people | 0.0871 | 26.9 / 34.9 |
| A3 three people + 5 warm objects + lamp | 0.1132 | 29.6 / 95.6 |
| B0-B3 standing person, fog 0-3 | 0.0732 / 0.0689 / 0.0607 / 0.0446 | 24.5-23.0 / 33.6-30.0 |
| L0 / L1 lamp off / on (empty room) | 0.0166 / 0.1939 | 22.7 / 53.9 max |
| C00 C01 C02 C04 C08 C12 C16 warm objects | 0.0166 0.0170 0.0463 0.0609 0.0911 0.0899 0.0888 | max up to 99 C |

Notes: A2 HT is slightly below A1 (people overlap and fill blocks uniformly; HT is not monotone in
people count). The complexity sweep saturates above 8 objects (blocks already contain a hot edge;
objects above 50 C clip in the fixed 10-50 C BIE scale). C01 is almost equal to C00 because the single
object drawn for this seed is a mug, which covers only a few pixels (or is occluded) from this pose.

### Cost on the Jetson  [`jetson_bank_runs.json` (per trial), `jetson_bank_summary.json` (per bank), `cost_vs_ht.png`]

Shared harness `~/synthlab/harness/run_thermal_bank.py` (jetson-harness track): TLinear -> 8 bit -> BIE ->
YOLOv8n PyTorch CUDA FP32 (the original pipeline), unpaced, 5 s warmup + 30 s measured, 2 repeats per bank,
under `tools/jetson.sh measured`, MAXN_SUPER. 17 synthetic + 5 real banks = 44 trials, all `ok`.
Power = tegrastats VDD_IN mean over the measured window.

| | synthetic banks (n = 17) | real recordings (n = 5) |
|---|---|---|
| HT range | 0.017 - 0.194 (11x) | 0.013 - 0.099 |
| total per-frame time | 63.94 - 64.50 ms | 64.11 - 64.59 ms |
| YOLO inference | 27.6 - 30.6 ms | 26.6 - 30.6 ms |
| YOLO post-processing (NMS etc.) | 1.68 - 4.57 ms | 1.65 - 5.86 ms |
| BIE (HT) stage itself | 26.2 - 26.7 ms (constant, 41 % of the loop) | same |
| VDD_IN power | 7.11 - 7.26 W (idle 4.7-4.9 W) | 7.12 - 7.24 W |
| trial-to-trial SD (same bank) | <= 0.17 ms, <= 0.03 W | |

Across the 17 synthetic banks: Pearson r(HT, total ms) = +0.23, r(HT, power) = +0.03, r(HT, YOLO inference)
= -0.05, but r(detections, YOLO post ms) = +0.99 and r(detections, total ms) = +0.83. The fog banks B0-B3
(HT -39 % from B0 to B3) cost 63.98 / 64.04 / 64.09 / 64.18 ms and 7.13 / 7.13 / 7.12 / 7.12 W.
Lamp on vs off: HT 0.194 vs 0.017, 64.20 vs 64.17 ms, 7.163 vs 7.158 W.

What this says: **in this pipeline the cost follows the number of YOLO detections, not HT**; the
whole spread is ~0.6 ms (1 %) and ~0.15 W (2 %). Inference time drops a little when detections appear
while post-processing rises (likely CUDA-sync time moving between the two timers), so their sum is flat.
The earlier finding r(HT, yolo_ms) = 0.70 on the 279-trial sweep is consistent with this if HT and
detections co-varied there.

Big caveat: **YOLO rarely sees the simulated people**. Person detected in 0 % of frames for the standing
person (B0-B3), 9 % (A1 walking), 29 % (A2), 0 % (A3), versus 57-100 % in the real recordings. So these
physics banks under-drive detection-dependent cost; they are good for testing HT and the BIE stage, not
for driving YOLO's per-detection load. (C04 triggers one false detection per frame on a warm object.)

## 5. What it means for the hypotheses
- **Hyp A (complexity raises HT and power):** HT rises with people, warm objects and the lamp (0.017 -> up to
  0.19, but not monotone: A2 < A1, C sweep saturates above 8 objects). Power and latency do not follow
  HT (spread <= 2 %); they follow detections. Thermal side of A: HT yes, power no (for this pipeline).
- **Hyp B (fog: HL up, HT flat, power up):** simulated HT is not flat - it falls 4 / 14 / 34 % at B1 / B2 / B3.
  Plain fixed-scale Shannon is flatter (-1 to -3 %). Thermal cost under fog is flat (64.0-64.2 ms,
  7.12-7.13 W), so any power rise in Experiment B would have to come from the LiDAR side.
- **Hyp C (knee):** nothing to see on the thermal side; there is no slope to bend.
- **Null:** supported for the thermal pipeline: a fixed compute graph gives constant cost except for the
  detection-count-dependent NMS/post-processing (~1.7 -> 5.9 ms).
- Design flag: the BIE stage itself costs 26.5 ms per frame (41 % of the loop) and is HT-independent;
  measuring entropy is a larger cost than anything entropy changes.

## 6. Limits
- Appearance validated against 5 real recording groups only, none taken from the room_v1 camera pose.
- People, lamp, materials and temperatures are assumptions; HT absolute values differ from real by up to 49 %.
- Fog constants are literature values, no real fog recording exists to validate.
- No reflection beyond one bounce; no convection plumes from people; no AGC (radiometric TLinear only).
