# Measuring edge power and latency with synthetic or replayed inputs

Research track, night of 2026-11-23. Sources fetched tonight unless marked. The Jetson probe
numbers come from `results/research/jetson_ina3221_sysfs_probe.txt` (read-only sysfs reads, 3 s,
no measurement lock needed, taken at 21:51 EST while other tracks may have been running).

## Precedent for replayed and synthetic inputs

- **MLPerf Inference / MLPerf Power.** The standard edge benchmark feeds stored datasets from a
  load generator instead of a live sensor, and measures full-system power aligned with the run,
  with a minimum 60 s window and an external analyser. Tschand et al., "MLPerf Power:
  Benchmarking the Energy Efficiency of Machine Learning Systems from Microwatts to Megawatts
  for Sustainable AI", HPCA 2025, https://arxiv.org/abs/2410.12032 (search-summary level).
  This is the strongest precedent: replay is the norm for power benchmarking.
- **Liu, Wang, Shi 2022** (arXiv 2209.05487, https://arxiv.org/pdf/2209.05487) replayed KITTI
  through detectors on a Jetson AGX Xavier, including images with rain *rendered in*, and
  measured latency. A direct precedent for synthetic-weather-in, timing-out on a Jetson.
- **Hahner et al. 2021** (https://ar5iv.labs.arxiv.org/html/2108.05249) is the precedent for
  simulated fog standing in for real fog, validated on real fog afterwards.
- **Rasshofer et al. 2011** describe a hardware-in-the-loop target simulator for automotive
  laser radar (abstract level).

What replay cannot show: the capture path. USB/UVC transfer, the Lepton's FFC pauses, and the
8.7 Hz frame pacing are part of the real duty cycle. A replay that feeds frames faster than
8.7 Hz changes the idle fraction and therefore mean power. Replay at the sensor's rate, or
report energy per frame over idle rather than mean watts.

No paper was found that replays synthetic LWIR or Livox data through a Jetson while logging
power. A dedicated search for "sensor replay power" literature was not exhaustive.

## The Jetson power sensor: what was measured tonight

| Item | Value | Source |
|---|---|---|
| Device | INA3221 at hwmon1, channel 1 = VDD_IN, 2 = VDD_CPU_GPU_CV, 3 = VDD_SOC | probe |
| Shunt | 5000 micro-ohm on each channel | probe |
| Current step | 8 mA (22 distinct values, all 8 mA apart) | probe |
| Bus voltage | 4936 mV, one value in 3 s (8 mV step per datasheet) | probe |
| One VDD_IN step | 8 mA x 4.936 V = 39.5 mW | derived (`derived_numbers.json`) |
| Driver averaging | `samples` = 512, `update_interval` = 1 | probe |
| Observed refresh | 2,248 reads in 3 s gave 80 value changes: about 27 Hz or slower | probe |

Reading: every tegrastats VDD_IN number is a 512-sample hardware average quantised to about
40 mW. That is good for mean power and useless for per-frame power.

**The effects under study are at or below one step.** The audit track reports 36 mW (sweep) and
73 mW (replay) as the largest input effects, and 45 mW of temperature drift. All three are
within two quantisation steps. They can still be estimated as differences of means over many
samples, because rail noise dithers the quantiser, but they need long windows, interleaving and
confidence intervals. A single trial pair proves nothing.

## Known pitfalls

- **Accuracy.** Shalavi, Khoshsirat, Stellini, Zanella, Rossi, "Accurate Calibration of Power
  Measurements from Internal Power Sensors on NVIDIA Jetson Devices", IEEE EDGE 2023,
  https://arxiv.org/abs/2306.13107 : internal sensors underestimate true power by up to 50 % in
  most cases; a regression calibration against external hardware brings error within 3 %.
  Use differences between conditions, not absolute watts, until calibrated.
- **tegrastats interval.** `--interval` sets only the reporting rate, not the sensor's
  sampling. How tegrastats forms its "average" field is undocumented; an NVIDIA forum thread
  asking this has no NVIDIA answer
  (https://forums.developer.nvidia.com/t/units-and-sampling-intervals-for-jetson-orin-power-measurements/378378).
  Read the hwmon files directly at a fixed rate and timestamp them yourself.
- **VDD_IN scope.** VDD_IN is module input: it includes regulator losses and everything on the
  module, but not carrier-board peripherals (USB devices such as the PureThermal, the LiDAR's
  own supply). The Mid-360 draws 6.5 W on its own supply and will not appear.
- **Temperature.** Leakage rises with die temperature. The audit track measured about
  12 mW/degC here, so a 3-4 degC drift equals the whole effect. Log all thermal zones, start
  blocks at the same temperature, and include temperature as a covariate.
- **DVFS and state jumps.** Governors change CPU/GPU frequency with load. The audit track found
  a mid-condition jump (inference 30.8 -> 37.0 ms on a fixed graph) that explained 83 % of a
  reported power gap. The brief forbids `jetson_clocks` tonight, so log CPU and GPU frequency
  every sample and discard blocks with a frequency or latency regime change.
- **Throttling.** Forum reports put the software throttle near 99 degC, and over-current events
  on Orin Nano cut clocks without an OS notification (earlier project note). Log the hwmon
  over-current alarm counters.
- **Warm-up.** First CUDA calls, cuDNN autotuning and allocator growth inflate the first frames.
  Discard a fixed warm-up (the Dimitrovski et al. protocol in the earlier note used 50 runs).
- **Order effects.** Running A0 then A1 then A2 confounds condition with time and temperature.
- **The harness itself.** Python logging, disk writes and the metric code (BIE at 34.5 ms per
  frame per the audit track; HL at about 350 ms per window) cost more than the effect being
  measured. Compute metrics offline, not in the measured loop.
- **Shared device.** Tonight several tracks share the Jetson. Only locked `measured` runs are
  comparable.

Dimitrovski et al., "Towards Practical Precision Agriculture: Real-Time Fruit Detection and
Video Analytics on Embedded Edge Hardware", arXiv 2609.13551 (Sep 2026),
https://arxiv.org/abs/2609.13551 , is on the same board (Orin Nano Super). The abstract confirms
TensorRT FP16 cuts gross energy by 64-66 % against PyTorch FP32; the protocol details quoted in
the earlier project note were not re-verified tonight.

## Statistics for small effects

These are standard methods; the citations are from memory except where a URL is given.

- **Design.** Interleave conditions in randomised blocks (for example ABBA or a Latin square)
  so drift cancels. Kalibera and Jones, "Rigorous Benchmarking in Reasonable Time", ISMM 2013
  (https://kar.kent.ac.uk/id/document/3127886, confirmed by search) show how to choose the
  number of repetitions at each level (frames within runs within sessions) and argue for effect
  sizes with confidence intervals.
- **Unit of analysis.** Frames within a run are autocorrelated, and the sensor already averages
  512 samples. The independent unit is the block or run, not the frame. Use a mixed model with
  run as a random effect, or a block bootstrap.
- **Paired differences.** Compare each condition block with its neighbouring baseline block and
  analyse the differences.
- **Covariates.** Add die temperature and detection count to the model. The question "does
  entropy matter beyond detection count" is a partial-effect question.
- **Claiming a null.** A non-significant test is not evidence of no effect. Use equivalence
  testing (two one-sided tests; Lakens 2017, not re-fetched) with a bound stated in advance, for
  example "any effect is smaller than 1 % of VDD_IN (about 58 mW)".
- **Power analysis.** With a 40 mW step and block-to-block SD measured from pilot data, compute
  the number of block pairs needed to detect 20 mW. Do this before the fog experiment.
- **Measurement bias.** Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously
  Wrong!", ASPLOS 2009 (not re-fetched): environment size and link order alone shift timings.
  Keep the process environment identical between conditions.
- **Multiple comparisons.** A0-A3 x B0-B3 x several metrics is many tests. Pre-register the
  primary contrast.

## What to do with this tonight

- Every measured run: sample hwmon current and voltage at a fixed rate with timestamps, plus
  thermal zones and CPU/GPU frequency; pace frames at 8.7 Hz or report idle-subtracted energy
  per frame.
- Interleave banks (baseline, condition, baseline, condition) inside one locked run; never
  compare runs taken an hour apart.
- Print "one VDD_IN step = 39.5 mW" beside every power difference on the slides.
- Report differences with block-bootstrap confidence intervals and n = number of blocks.

## What to do with this next month

- Attach the AVHzY CT-3 and calibrate VDD_IN against it (Shalavi et al. method).
- Pin clocks (`jetson_clocks`) once the owner allows it, and repeat the key contrast.
- Run a pilot to get block-to-block SD, then size the real fog experiment with a power
  analysis and a pre-stated equivalence bound.
- Add a capture-path control: live camera on a static scene versus replay of the same frames.
