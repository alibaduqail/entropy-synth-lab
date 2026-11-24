# thermal-datadriven: a real-data-driven thermal generator

Branch `thermal-datadriven`, code in `synthlab/thermal_datadriven/`, results in
`results/thermal-datadriven/`. Cost numbers come from the shared Jetson harness
(`origin/jetson-harness`, `docs/jetson-harness.md` section 3 and `results/jetson-harness/cmp_*`).
Nothing was rerun to write this note; every number below is read from a file named next to it.

## 1. What the approach is

Instead of simulating physics, take the real Lepton 3.5 recordings (empty room A0, one person
with a heat lamp A1, two seated people), split them into parts, and recombine the parts into new
160x120 TLinear frames. The sensor's own noise, drift, fixed pattern and the real shape and
temperature of real people are reused, so the output inherits the measured statistics instead of
an idealised model. The generator exposes controls the real experiment cannot vary cleanly:
number of people, lamp strength, warm objects, ambient shift, noise multiplier, drift multiplier.
Labels (person masks) come for free.

The point is to feed the Jetson thermal pipeline (BIE entropy HT + YOLOv8n) with frames that look
like real ones and see whether cost responds to HT, to people count, or to noise.

## 2. How it is built

Frame model (`compose.py` docstring):

`frame_t = round(B + ambient_shift + lamp * heated_surface (+ warm objects, + people pasted far to near) + drift_t * drift_mult + w * noise_t)`

1. **Background and drift** (`sensor_model.py`). B is the temporal median of the session's build
   trials (static scene plus fixed pattern). Drift is a 3-mode PCA of frame minus trial median
   (`N_MODES = 3`); real coefficient series, including the shutter (FFC) steps, are stored and
   replayed because FFC steps are not stationary.
2. **Measured noise model** (`sensor_model.py`). Per-pixel temporal SD map sigma
   (median 7.5 counts = 0.075 K for the heat-lamp session, `models/build_summary.json`),
   times a unit-variance field with the measured 2-D spatial power spectrum. Each spatial
   frequency is a slow AR(1) part plus a white part, solved from measured lag-1 and lag-4
   correlations (median rho 0.92, slow fraction 0.995). Row/column-correlated noise sits on the
   ky=0 and kx=0 lines of the spectrum (row-noise variance fraction 0.31, column 0.045).
3. **Real person layers** (`layers.py`, `library.py`). Frame minus empty background gives a
   temperature residual; a person is the connected warm region, kept as a soft matte plus
   "un-mixed" foreground temperature, so the Lepton's own blur softens the edge. Layers are
   pasted far to near.
4. **Lamp signature** (`layers.py`). Pixels hotter than 38 C (`LAMP_T_C`) are the hand-held
   heat lamp and its halo; the heated surface is stored separately so lamp strength is a
   control (`lamp_surface_hl.npz`).
5. **Recombine with controls** (`compose.py`).

Assumptions, none measured (all in code):

- Horizon at image row 53 (`HORIZON_Y`, camera pitched about 3 deg down, taken from
  `scene/room_v0.json`). Scaling a layer by s about the horizon moves the person from depth Z to
  Z/s (pin-hole, level floor).
- Shoulder width 0.42 m (`BODY_WIDTH_M`) for depth ordering.
- Skin 34 C (`SKIN_C`) does not follow ambient; clothing temperature follows ambient linearly.
- PSF sigma 0.7 px (`PSF_SIGMA`) restored after down-scaling a layer.
- Bottom-truncated layers are only enlarged (s >= 1); person scale drawn from 0.7-1.2.

## 3. Validation on held-out sessions

Design (`split.py`, `valsets.py`): the generator is built only from BUILD trials. Real TEST
trials are never touched. For each condition the validation sets are 64-frame clips from
held-out real trials (real_test), from build trials (real_build, the real-vs-real floor), and
from synthetic seeds. Seven conditions: hl_A0 (12 held-out empty trials), hl_A1lamp (13), rj_A0,
rj_A1, w120_A0, w120_A2 (two seated people), and rj_A1_transplant (people cut from the w120
session pasted into the rj room). Source: `results/thermal-datadriven/validation/metrics.json`
and `classifier.json`.

### 3.1 Entropy HT, spectrum, temporal noise

HT ratio is synthetic mean HT over real-test mean HT. PSD columns are log10 ratios of the radial
power spectrum (rms over all frequencies / high-frequency band); the floor is real_build against
real_test.

| condition | HT real | HT synth | HT ratio synth | ratio floor | PSD rms synth / floor | PSD high-freq synth / floor |
|---|---|---|---|---|---|---|
| hl_A0 | 0.0132 | 0.0129 | 0.975 | 0.999 | 0.082 / 0.023 | -0.103 / 0.000 |
| hl_A1lamp | 0.078 | 0.0824 | 1.057 | 0.953 | 0.126 / 0.066 | -0.123 / 0.049 |
| rj_A0 | 0.0565 | 0.053 | 0.939 | 0.932 | 0.033 / 0.027 | 0.027 / 0.013 |
| rj_A1 | 0.0708 | 0.0768 | 1.086 | 1.025 | 0.070 / 0.052 | -0.015 / 0.026 |
| w120_A0 | 0.0154 | 0.0168 | 1.086 | 1.000 | 0.048 / 0.027 | -0.040 / -0.013 |
| w120_A2 | 0.1032 | 0.1105 | 1.071 | 0.960 | 0.040 / 0.037 | 0.038 / -0.053 |
| rj_A1_transplant | 0.0708 | 0.0909 | 1.284 | 1.025 | 0.121 / 0.052 | 0.121 / 0.026 |

- HT is reproduced within about 2 to 9 percent for the six same-session conditions (ratio
  0.94 to 1.09). The cross-room transplant overshoots by 28 percent.
- The radial PSD is close (rms log10 ratio 0.03 to 0.13) but synthetic is about 2 to 4 times
  further from real than build-vs-test is, in hl_A0, hl_A1lamp and rj_A1.
- Temporal noise (per-pixel temporal SD, K; real_test vs synth): hl_A0 0.060 vs 0.063,
  hl_A1lamp 0.080 vs 0.108, rj_A0 0.061 vs 0.059, rj_A1 0.073 vs 0.101, w120_A0 0.062 vs 0.063,
  w120_A2 0.112 vs 0.221. Empty-room noise matches; with people in frame the synthetic noise is
  too high, by a factor of two in w120_A2. Lag-1 autocorrelation is also slightly high for
  synthetic (hl_A0 0.816 vs 0.781; w120_A2 0.956 vs 0.903).
- Column-mean SD (a fixed-pattern proxy) is about twice real with people (hl_A1lamp 0.540 vs
  0.265 K; w120_A2 0.610 vs 0.279 K).
- Temperature level: median-removed pixel histogram W1 is small for empty rooms (0.005 to 0.044 K)
  but 0.22 K (hl_A1lamp), 0.34 K (w120_A2) and 0.27 K (transplant) with people, against floors
  of 0.045, 0.078 and 0.036 K.

### 3.2 Classifier test: this is NOT indistinguishable

A gradient-boosting classifier (`classifier.py`; features: log radial spectrum, median-removed
histogram, frame-difference SD, row/column mean SD; clip-grouped cross-validation) separates
held-out real from synthetic frames. 0.5 would mean indistinguishable. The floor is the same
classifier on real_build vs real_test.

| condition | synth vs real | real-vs-real floor | gap | noise-features only, synth / floor |
|---|---|---|---|---|
| hl_A0 | 0.963 | 0.806 | 0.16 | 0.979 / 0.716 |
| hl_A1lamp | 0.980 | 0.804 | 0.18 | 0.737 / 0.592 |
| rj_A0 | 0.950 | 0.940 | 0.01 | 0.971 / 0.844 |
| rj_A1 | 0.946 | 0.946 | 0.00 | 0.881 / 0.922 |
| w120_A0 | 0.912 | 0.724 | 0.19 | 0.959 / 0.688 |
| w120_A2 | 0.952 | 0.832 | 0.12 | 0.780 / 0.772 |
| rj_A1_transplant | 0.992 | 0.946 | 0.05 | 0.975 / 0.922 |

Plainly: a classifier tells synthetic from real in 91 to 99 percent of frames in every
condition. The floor is itself high (0.72 to 0.95) because different recording days and
trials are separable even among real data, so the honest reading is the gap to the floor, not
the distance to 0.5.

- The gap is largest for empty rooms in the main sessions: w120_A0 (0.19), hl_A1lamp (0.18),
  hl_A0 (0.16). With noise features alone, hl_A0 reaches 0.979 against a floor of 0.716 and
  w120_A0 0.959 against 0.688, so the residual difference lives in the noise fine structure
  (high-frequency spectrum, row/column and frame-difference statistics), not in the scene.
- It is close to the floor only for the rj session (gap 0.00 to 0.01), where the floor is
  already 0.94 because the rj trials are few (1 to 2 build trials, 378 to 756 test frames).
- The transplant is the most separable condition (0.992), consistent with the PSD and HT misses.

Figures: `results/thermal-datadriven/classifier_accuracy.png`,
`validation_distributions.png`, `contact_sheet_real_vs_synth.png`; curves in
`validation/curves.npz`.

## 4. The banks (30)

Written by `make_banks.py` (128 frames each). Synthetic banks use only BUILD layers; `real_*`
banks are held-out real trials. HT below is the bank mean from the harness file
`cmp_thermal_banks.csv` (the `HT` column, first block).

| group | names | HT |
|---|---|---|
| condition ladder | syn_A0, syn_A1 (one person holding lamp), syn_A2 (2 people + lamp + 1 warm object), syn_A3 (4 people + lamp + 3 warm objects, intensity 0.8) | 0.012, 0.074, 0.100, 0.090 |
| people sweep, no lamp | people_0 ... people_8 | 0.012, 0.089, 0.107, 0.109, 0.078, 0.091, 0.092, 0.093, 0.081 |
| noise sweep on the A1 scene | noise_0x, 0p5x, 1x, 2x, 4x, 8x, 16x (same scene, seed 500, only noise differs) | 0.092, 0.092, 0.090, 0.088, 0.083, 0.076, 0.065 |
| lamp sweep, one person | lamp_0, lamp_0p5, lamp_1, lamp_1p5 | 0.113, 0.085, 0.091, 0.103 |
| held-out real | real_A0_0..2, real_A1_0..2 | 0.015, 0.013, 0.013 / 0.083, 0.069, 0.078 |

HT rises from 0.012 (empty) to about 0.09 with one person and then saturates: people_1 to
people_8 sit in 0.078 to 0.109 with no trend (people-count slope +0.0045 HT/person, CI -0.0047
to +0.0135, `cmp_sweeps.csv`). The people_k labels are the mean number of people seen (the
harness records 0 to 7.5), because overlap and cropping hide some. The noise sweep lowers HT
at high noise (0.092 at 0x, 0.065 at 16x): added noise blurs the block-entropy structure of the
person rather than raising it. lamp_0 (one person, no lamp) has the highest HT in the lamp sweep
(0.113), so the lamp does not behave as a clean HT lever in these banks.

## 5. Cost on the Jetson (from the shared harness)

Source: `origin/jetson-harness:results/jetson-harness/cmp_thermal_banks.csv`,
`cmp_named_comparisons.csv`, `cmp_sweeps.csv`, `cmp_verdicts.csv`, and `docs/jetson-harness.md`
section 3. Design: every bank run in seeded-shuffled blocks, PyTorch FP32 YOLOv8n unpaced,
8 s measured + 2 s warmup per bank, chip temperature as covariate. **Blocks completed: 2 for
most thermal-datadriven banks, 1 for the rest** (the harness doc states block 0 for all banks,
block 1 for 40 of 74; the planned third block never ran). All trials passed telemetry checks.
Noise floor (65 non-adversarial banks): 1.10 ms total time, 0.127 W VDD_IN, 10.4 mJ/frame; the
minimum detectable difference between two banks at n=2 blocks is 1.1 ms, 0.127 W, 10.4 mJ.
VDD_IN is Jetson module input power at 39.5 mW resolution, not wall power.

**Synthetic A1 vs real A1** (`syn_A1` vs `real_A1_0..2`, `cmp_named_comparisons.csv`):

| metric | syn_A1 | real_A1 | diff [95% CI] |
|---|---|---|---|
| total ms/frame | 63.97 | 64.37 | -0.40 [-1.08, 0.29] |
| VDD_IN W | 7.12 | 7.19 | -0.066 [-0.153, 0.021] |
| mJ/frame | 457 | 464 | -7.0 [-14.7, 0.7] |
| YOLO post/NMS ms | 3.27 | 4.44 | -1.17 [-1.42, -0.92] |
| frames with a detection | 0.552 | 0.963 | -0.41 [-0.48, -0.34] |
| detections per frame | 0.648 | 0.965 | -0.32 [-0.39, -0.25] |
| HT | 0.0739 | 0.0771 | -0.0031 [-0.0111, 0.0048] |

Total latency, power and energy match real within noise, but **YOLO detects the person in 55
percent of synthetic frames versus 96 percent of real A1 frames**, which confirms the harness
statement. The post-NMS time is 1.2 ms lower because there are fewer detections and candidates.
(The prose in `docs/jetson-harness.md` quotes slightly wider intervals for the total, for
example -0.24 ms [-4.0, 3.5]; the CSV rows above are what I quote. The conclusion is the same.)
Synthetic A0 vs real A0: latency equal (+0.63 ms [-0.81, 2.06]), no detections either side,
VDD_IN -0.135 W [-0.221, -0.049], so a small but non-zero power offset on empty frames.

**Cost vs people count** (`people_0..8`, `cmp_sweeps.csv`): total +0.015 ms/person
[-0.137, +0.207]; VDD_IN +0.0087 W/person [-0.0098, +0.0213], over the range 0 to 7.5 people:
+0.11 ms and +0.066 W, both below the minimum detectable difference. Detections +0.17 per person
[-0.23, +0.56]. Cost is flat in people count. The fraction of frames with a detection is 1.0 for people_1 to people_7 and 0.91 for
people_8, so once one person is found in every frame, extra people add little work.

**Noise sweep** (`noise_0x` ... `noise_16x`): total time 63.6 to 64.4 ms with no trend (slope
vs HT -7.8 ms per unit HT, CI -50 to +6), VDD_IN 7.13 to 7.30 W. The only visible effect is on
detection: frames with a detection fall from 0.59 at 0x to 0.05 at 16x, and the NMS time falls
from 3.4 to 1.8 ms. High noise hides the person from YOLO and removes work rather than adding it.

**Cost vs HT over the 30 banks** (`cmp_verdicts.csv`): total ms +10.6 per unit HT [3.9, 21.2],
VDD_IN +1.23 W per unit HT [0.56, 1.95], r = 0.55 and 0.64. Over the realised HT range 0.10 that
is +1.07 ms and +0.12 W, about one minimum detectable difference. After controlling for
detection frequency and candidate count the slopes are 1.23 ms (SE 5.3) and 0.97 W (SE 0.57),
no longer distinguishable from zero. No knee: linear-vs-hinge BIC difference is -6.7 (latency)
and -5.3 (power), so the hinge is not preferred. The harness doc calls this the clearest raw
slope of any track.

## 6. What it means for the hypotheses

- **Null (fixed compute graph, constant cost):** supported for power and inference time in
  these banks. Cost on the thermal-datadriven banks is within about one minimum detectable
  difference across people count, noise and lamp strength. The one input-dependent term is
  NMS time, which depends on detections.
- **Hyp A (complexity raises HT and power):** weak. HT rises with a person present (0.012 to
  about 0.09) and power rises raw with HT, but through detections; people count 0 to 7.5 raises
  neither HT beyond saturation nor cost. Not a strong confirmation, not a refutation.
- **Hyp B (fog: HL up, HT flat, power up):** not tested here. This generator has no fog.
  The only thermal fog data are thermal-physics banks (see the harness doc).
- **Hyp C (knee):** no hinge preferred for thermal-datadriven; the 30 banks do not sample a
  wide enough or high enough HT range to see one (maximum bank HT 0.113).
- **Caveat that matters most for A/B/C:** the generator under-detects people (55 vs 96 percent),
  and detections are the only path through which HT reached cost. So cost slopes derived from
  synthetic banks may understate what real people would do.

## 7. Limits

- No new rooms: backgrounds are the three recorded sessions (hl, rj, w120).
- No fog or haze; no new viewpoints (the camera geometry of each session is fixed).
- Cross-room transplant is worse: HT ratio 1.28, PSD high-frequency ratio 0.121 vs floor 0.026,
  classifier 0.992. Do not use cross-session people.
- Not statistically indistinguishable (section 3.2); the noise fine structure is the tell, and
  with people in frame synthetic temporal noise is too high (up to 2x in w120_A2).
- Synthetic people move more than seated real people (observation from the generator's
  designer; the pasted layers come from standing people and the w120 clips are seated), so
  synthetic motion and noise with people are likely too high.
- The 0.42 m shoulder width, row 53 horizon, 34 C skin and 0.7 px PSF are assumptions, never
  fitted to ground truth.
- Rows in `cmp_thermal_banks.csv` come from 1 to 2 blocks (not 3), so per-bank cost differences
  below about 1.1 ms or 0.13 W are not resolved.

## 8. Unverified items

- No figure was inspected for this write-up; the figure names are listed from the directory.
- Cause of the 55 vs 96 percent detection gap is not established (candidates: edge sharpness,
  person temperature contrast, over-high noise with people). No ablation was run.
- That the same-session HT agreement transfers to new rooms is not tested.
- Whether the people-count flatness persists with detections gated to a real-like 96 percent
  rate has not been measured.
- The synthetic-versus-seated motion claim has no number in the files read.
- The 2-block power differences (for example syn_A0 vs real_A0, -0.135 W) are near the 0.127 W
  floor and should be repeated with at least 3 blocks before being relied on.
