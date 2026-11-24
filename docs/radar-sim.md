# radar-sim: simulated TI IWR6843ISK 60 GHz radar

Branch `radar-sim`. Code: `synthlab/radar_sim/`. Tests: `tests/test_radar_sim.py` (11 tests, run as
`python3 tests/test_radar_sim.py`, all pass; pytest is not installed on this Mac). All numbers below come from
files in `results/radar-sim/` (file named at each section). Everything here is SIMULATED unless it says
"measured"; the only measured quantities are Jetson timings and power (`jetson_bench/`).

## 1. Model and assumptions
- Signal-level FMCW: raw ADC cube (96 loops x 3 TX x 4 RX x 96 samples) generated from point scatterers, then
  range FFT, static-clutter removal, Doppler FFT, detection matrix, CFAR, angle estimation, clustering, tracking
  (`process.py`, stage names in section 5).
- Radar preset `people_counting_6m` (`fog_summary.json` -> `radar`): 60.75 GHz start, 1.78 GHz bandwidth, centre
  63.01 GHz, range resolution 8.4 cm, max range 8.08 m, velocity resolution 0.093 m/s, max 4.45 m/s, 18.2 frames/s
  (55 ms), 12 dBm TX, 7 dBi antennas, 12 dB noise figure. A second preset `oob_3d` is defined (not used in the studies).
- Scene: `room_v1.json` (room-3d track), radar pose and a person mark taken from it. Static scatterers: 264;
  a person adds 15 moving scatterers (`jetson_bench/summary.json`).
- Person: scatterer cluster with breathing (+-2 mm chest, 0.25 Hz), heartbeat (0.15 mm, 1.15 Hz), optional postural
  sway (`person.py`). These amplitudes are ASSUMED, not measured on a real person.
- Link budget (`fog_summary.json`): a 0.5 m^2 torso at 1.33 m gives 86.2 dB SNR after processing; at 4 m, 67.1 dB.
  RCS 0.5 m^2 is assumed.
- Fog physics: liquid-water attenuation from the ITU-R P.840 double-Debye model, Rayleigh backscatter from drops,
  refractivity change with humidity, water film on the radome. Drop diameter 4 um for B1-B3 (assumed); LWC levels
  B1/B2/B3 = 0.1/0.5/2.0 g/m^3 with RH 0.80/0.95/1.00 (assumed to match the team's fog conditions, not measured).
- Not modelled: multipath beyond the listed static scatterers, real antenna patterns beyond a cos^n model, ADC
  saturation, real firmware/TI demo-code details, interference between radars.

## 2. 60 GHz fog loss over the room: is the radar a fog witness?
Source: `fog_summary.json` (analytic), figure `fog_attenuation.png`.
- Liquid-water specific attenuation at 63.0 GHz, 22.5 C: 1.83 dB/km per g/m^3.
- Two-way loss through fog, 4 m path (`liquid_two_way_db`): B1 0.0015 dB, B2 0.0073 dB, B3 0.029 dB; the 2 m values
  are half that (B3 0.0146 dB). A dense natural fog (0.5 g/m^3) is the same as B2. Even a 20 g/m^3 plume next to the
  mist maker gives 0.29 dB over 4 m.
- For comparison, oxygen absorption (always present, not fog) is 0.12 dB two-way over 4 m.
- Simulated person-bin power change vs B0 (`fog_summary.json` -> `simulated`, 36 frames, seed 0): -0.025 dB (B1),
  -0.037 dB (B2), -0.049 dB (B3). Detections per frame stay at 3.14 in all four levels; normalised range-Doppler
  entropy h_rd_total_norm 0.2504 (B0) vs 0.2505 (B1-B3).
- Refractivity: humidity rise shifts apparent range by 0.08 mm (B1) to 0.13 mm (B3) at 2 m.
- Fog backscatter per range cell is -124 to -97 dBsm for 4 um drops at B2-B3 (1 m cell), versus a noise floor of
  about -94 dBsm in that cell; 20 um drops reach -87 dBsm only in the mist-maker plume. So drops of the assumed size
  are essentially invisible to the radar.
- Verdict: at 60 GHz the radar is effectively blind to fog of this density. It is NOT a fog-sensitive sensor but a
  fog-blind reference: if the radar output stays flat while LiDAR HL rises and thermal HT stays flat, that supports
  "the room did not change, only the air did" (witness of the scene, not of the fog). Caveat from the same file: a
  water film on the radome is the one fog effect the radar sees: 50 um film costs 9.0 dB two-way in the
  analytic table, and the simulated B3 + 50 um radome case drops detections from 3.14 to 2.0 per frame and the
  person-bin power by 9.0 dB. Keep the radome dry or note it.

## 3. Smallest person movement the stillness metric detects
Source: `stillness_summary.json`, `stillness_trials.csv` (530 trials, 12 seeds per point, 16 s per trial, event at
8 s lasting 1 s, thresholds = max over the no-event baseline trials for the same sway). Metric names as in the file.
- Baseline (breathing + heartbeat only, no sway), `excursion_mm` radial: median 0.20 mm, max 0.33 mm over 32
  trials. A person who is merely breathing therefore sits at about 0.2-0.3 mm on this metric; threshold 0.33 mm.
- Toward the radar, no sway: a 0.5 mm step is already detected (p_detect 1.0, `min_detectable_mm` 0.5); 2 cm
  gives p = 1.0 (median 20.0 mm); 20 cm gives p = 1.0 (median 192.6 mm at the 200 mm point).
- With 2 mm RMS postural sway the baseline threshold rises to 4.5 mm (`excursion_mm`), and the smallest reliably
  detected radial step is 5 mm (p = 1.0; 2 mm: p = 0.58). 2 cm: p = 1.0 (median 22.1 mm); 20 cm: p = 1.0 (196.1 mm).
- Sideways (lateral) motion is much harder: `excursion_mm` p_detect for a 2 cm lateral move is 0.67 with no sway
  and 0.08 with sway; reliable at 5 cm with no sway, 20 cm with sway. The dedicated `lateral_shift_mm` metric detects
  lateral steps down to 0.5 mm with no sway (p = 1.0) and 5-10 cm with 2 mm sway (2 cm: p = 0.83, median 18.1 mm).
- Summary in words: breathing is below threshold (it is the baseline); a 2 cm shift toward or away from the radar
  and a 20 cm step are both detected every time; the limit for radial motion is about 0.5 mm with a perfectly
  still subject and about 5 mm with realistic sway.
- Medium changes with a perfectly still scene (`medium_changes`): a gradually building fog gave excursion median
  2.1 mm (max 3.2 mm) in the non-ideal case and 0.16 mm in the ideal case, so fog onset alone can come close to the
  sway-free threshold in the non-ideal case. (`fog_onset` vs `fog_onset_ideal` keys in `medium_changes`; the ideal variant presumably removes the rig/noise terms, see
  `stillness_study.py`.) Treat a flagged "movement" during fog onset with caution.

## 4. Radar entropy candidate across conditions
Source: `jetson_bench/cost_table.json` (3 repeats per condition, 48 frames each; derived from
`jetson_bench/summary.json`). h_rd_total_norm = normalised Shannon entropy of the range-Doppler magnitude map;
h_rd_dyn_norm = same after clutter removal (dynamic part); h_points_bits = entropy of the detected point cloud.

| cond | description | h_rd_total_norm | h_rd_dyn_norm | h_points_bits |
|---|---|---|---|---|
| A0 | empty room | 0.259 | 0.995 | 0.000 |
| A1 | one still person | 0.246 | 0.617 | 1.020 |
| A2 | three still people | 0.276 | 0.426 | 1.186 |
| A3 | three people + warm objects | 0.282 | 0.426 | 1.186 |
| B0..B3 | one still person, fog 0/0.1/0.5/2 g/m^3 | 0.246 for all | 0.617 for all | 1.020-1.023 |
| M1 | 1 person walking (extra) | 0.336 | 0.274 | 2.241 |
| M3 | 3 people walking (extra) | 0.362 | 0.354 | 3.179 |

- h_rd_total_norm rises with people and movement (0.246 -> 0.276 -> 0.362) and is flat across fog B0-B3: the radar
  entropy follows scene complexity (hypothesis A) and does not respond to fog.
- A3 equals A2 in h_points and h_rd_dyn: warm objects are invisible to the radar (they are not radar scatterers
  in the model), consistent with the thermal-only role of the warm objects. h_rd_total changes slightly (0.276 to
  0.282), a small effect from the objects' static return, not shown to be significant (n = 3 repeats).
- h_rd_dyn_norm is NOT a usable entropy: it is highest (0.995) for the empty room because it normalises noise-only
  maps; it falls as targets appear. Use h_rd_total_norm and h_points_bits.
- Consequence for hypothesis B: radar entropy is flat in fog, so it is a clean control channel; it cannot itself
  show a fog effect.

## 5. Cost vs condition (measured on the Jetson Orin Nano, MAXN_SUPER)
Source: `jetson_bench/cost_table.json` (run through `tools/jetson.sh measured`; raw run on the Jetson at
`~/synthlab/radar-sim/out/bench_v1/` incl. `frames.csv`, `power_trace.csv`; code at `~/synthlab/radar-sim/code/`).
Numpy 1.26.4, Python 3.10.12, CPU only. 3 repeats x 10 conditions, ~2300 processed frames per condition flat out.

Per-frame processing time (ms, mean over conditions' frames), chain run flat out ("free"):

| cond | detections/frame | fixed ms | data-dependent ms | total ms | gen ms/frame | power idle / flat-out / paced (mW) |
|---|---|---|---|---|---|---|
| A0 | 0.0 | 6.84 | 0.05 | 6.89 | 3.6 | 4775 / 6310 / 4893 |
| A1 | 4.2 | 6.87 | 0.68 | 7.55 | 33.2 | 4753 / 6280 / 4910 |
| A2 | 6.1 | 6.88 | 0.75 | 7.64 | 91.6 | 4819 / 6304 / 4942 |
| A3 | 6.1 | 6.81 | 0.75 | 7.56 | 91.7 | 4814 / 6241 / 4967 |
| B0 | 4.2 | 6.78 | 0.67 | 7.45 | 34.8 | 4739 / 6262 / 5004 |
| B1 | 4.2 | 6.80 | 0.68 | 7.48 | 81.3 | 4845 / 6309 / 4993 |
| B2 | 4.2 | 6.80 | 0.68 | 7.47 | 81.2 | 4766 / 6333 / 4921 |
| B3 | 4.2 | 6.85 | 0.68 | 7.53 | 81.4 | 4965 / 6265 / 4908 |
| M1 | 20.6 | 6.86 | 0.89 | 7.75 | 33.4 | 4724 / 6220 / 4966 |
| M3 | 32.8 | 6.79 | 1.06 | 7.84 | 92.3 | 4781 / 6293 / 4923 |

- Fixed-cost stages (do the same work every frame): range FFT 1.4 ms, clutter removal 0.6 ms, Doppler FFT 2.7 ms,
  detection matrix 0.6 ms, CFAR 1.4-1.5 ms (A0 values; the CFAR cost is flat because it scans the whole map regardless
  of how many cells fire). Together 6.8 ms.
- Data-dependent stages: angle estimation, clustering and tracking, whose cost scales with detection count. Pooled
  over 23,843 flat-out frames (0 to 46 detections): data-dependent time = 0.52 ms + 0.019 ms per detection
  (r = 0.71 with detection count); clustering alone r = 0.87 (0.011 ms/detection), angle r = 0.46 (0.0064 ms/detection),
  track r = 0.56 (0.0015 ms/detection). CFAR time has r = -0.04 with detection count and fixed time r = -0.03.
- Total time per frame spans 6.9 ms (empty) to 7.8 ms (3 people walking), a 14% rise; the radar frame period is
  55 ms, so the chain uses 12.5-14% of real time. Rise in time is entirely the data-dependent tail.
- Fog (B0 to B3): total 7.45, 7.48, 7.47, 7.53 ms with identical detection count (4.2). This is within run-to-run
  spread (the three repeats of B0 alone gave 7.29, 7.61, 7.45 ms; B3 gave 7.47, 7.58, 7.55 ms), so the null (no fog effect) holds for the radar
  chain on the Jetson CPU. Power at fog levels varies 6262-6333 mW flat out and 4908-5004 mW paced with no trend
  (idle baselines themselves moved 4739-4965 mW between runs).
- Power paced at the radar's real 18.2 fps is 4.89-5.00 W against an idle ~4.8 W, i.e. the radar chain adds
  roughly 0.1-0.2 W, indistinguishable from idle-baseline noise. Flat out it adds about 1.5 W (idle 4.8 to 6.3 W).
- Signal generation (not part of the real radar pipeline) costs 3.6-92 ms/frame and scales with scatterer
  count (empty 3.6, one person 33, three people 92 ms); the fog conditions with a mist model are ~81 ms in this
  bench because fog is modelled per-cell. This is simulator cost, to be excluded when attributing Jetson load.
- The `--stress` option (lowered CFAR thresholds to span a wide detection range) was requested for this run, but
  no stress rows were written to the output files, so no stress curve exists. The pooled per-frame fit above covers
  0-46 detections and is the cost-vs-detections evidence.

## 6. What a simulated radar can and cannot tell the team
Can:
- Predict the shape of the cost curve: fixed FFT stages dominate and are constant; only angle/cluster/track scale
  with detections; a few percent of frame time at most (n = 23,843 frames).
- Show that 60 GHz fog loss at B1-B3 densities is below 0.03 dB over 4 m and so a real radar should not see fog.
- Give a resolution and noise floor for the stillness metric (0.5 mm no sway, 5 mm with sway) as a design target.
- Provide a radar entropy (h_rd_total_norm, h_points_bits) that rises with people/motion and stays flat in fog, to be
  used as the control channel in hypothesis B.
Cannot:
- Say anything about a real IWR6843ISK: there is no real radar data in the project yet, so scatterer RCS, breathing
  amplitude, clutter, multipath, firmware CFAR and the demo's own cost are all assumed.
- Say how drop-size or condensation behaves in the real room (drop diameter 4 um assumed; the mist maker's real LWC is
  unknown, upper bound 3.2 g/m^3 after 10 min with no evaporation per `mist_maker_budget`).
- Give the real radar's power or latency: the timings are numpy on the Jetson CPU, not the TI on-chip DSP/the host
  parsing pipeline.
- Show a fog effect on the radar except through the radome water film.
- Confirm that stillness thresholds hold with real sway and clutter; sway of 2 mm RMS is assumed.

## Reproduce
`python3 -m synthlab.radar_sim.stillness_study --out results/radar-sim`;
`python3 -m synthlab.radar_sim.fog_study`; `python3 -m synthlab.radar_sim.figures`;
Jetson: `python3 -m synthlab.radar_sim.run_conditions --out <dir> --stress --seed 0` under `tools/jetson.sh measured`;
`python3 -m synthlab.radar_sim.bench_table summary.json frames.csv cost_table.json`.

Note: the Jetson benchmark used `room_v1.json` as of 22:07-22:18; the room-3d track has since reported a final `room_v1.json` (sensor poses assumed); the bench was not rerun against it.
