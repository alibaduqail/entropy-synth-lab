# lidar-sim: Livox Mid-360 + fog simulator in the test room (first look at Hypothesis B)

Status: FINAL for the overnight run (2026-11-24 02:45). Cost: 1 pass per workload (pass 2 cancelled, lock queue).
Everything about fog here is a SIMULATION-ONLY prediction. No real fog data exists in the project.

## What was built

| Piece | File | What it does |
|---|---|---|
| Room + ray caster | `synthlab/lidar_sim/scene.py` | Loads any `room_v*.json` (v0 and the room-3d track's v1), boxes, wall apertures (open door, glazing), people (capsule + head sphere), clutter. Vectorised numpy. |
| Sensor | `synthlab/lidar_sim/sensor.py` | Mid-360: 200,000 pts/s, 10 Hz frames, 360 x [-7, +52] deg, non-repetitive pattern (analytic approximation), range/angle noise, 1 mm quantisation, blind zone, dropout, glue points, Livox tag byte. |
| Fog | `synthlab/lidar_sim/fog.py` | Rasshofer 2011 / Hahner 2021 hard + soft return model, uniform or plume extinction, numpy or torch-CUDA convolution. |
| HL entropy | `synthlab/lidar_sim/entropy.py` | Re-implementation of the LiDAR teammate's method (window -> ROI -> 5 cm voxels -> radius kNN -> eigenvalues -> H_DS, H_eigen, mean_S, ...), stage-timed. |
| Banks | `bank.py`, `make_banks.py` | Conditions A0-A3, B0-B3, plume P1-P3, fog sweep, thinned controls; hand-off format of AGENTS.md. |
| Experiments | `experiments.py`, `figures.py` | Entropy-side sweeps and figures. |
| Cost | `cost_campaign.py`, `cost_analysis.py` | Drives the shared Jetson harness over all banks in shuffled order, then correlations / partial correlations. |

Tests: `tests/test_lidar_sim.py` (12 tests: geometry, scan pattern, Beer-Lambert attenuation, single-return
bound, tags, plume locality, entropy limits for line / plane / volume, bank round trip and determinism).

## Model and assumptions

MEASURED/SPEC = from the Livox spec page; ASSUMED = my choice, no data.

Sensor
- SPEC: 905 nm, 200,000 pts/s first return, 10 Hz, FOV 360 x [-7, +52] deg, blind zone 0.1 m, range
  precision 1 sigma <= 2 cm at 10 m and <= 3 cm at 0.2 m, angular precision < 0.15 deg, 40 m at 10 %
  reflectivity. Detection threshold is set from the last number: P_th = 0.10 / 40^2 in units where a
  Lambertian target returns rho cos(i) / R^2.
- ASSUMED scan pattern. The recorded Livox pattern file (`livox_laser_simulation/scan_mode/mid360.csv`,
  16.8 MB) was NOT downloaded (no owner approval for downloads overnight). The simulator uses 4 emitters,
  each sweeping elevation sinusoidally at 131.7 Hz while the head turns at 9.83 Hz; the frequencies are
  incommensurate so the pattern never repeats. `SensorParams.scan_csv` accepts the real file later
  without code changes. Consequence: per-frame line spacing and the elevation density profile are
  plausible, not verified.
- ASSUMED: random dropout 2 %, lognormal power noise sigma 0.3, glue points when a second ray 0.2 deg
  away sees a depth differing by > 5 cm (probability 0.5), clear-air particle-tag rate 0.46 % (copied
  from the real lab M0 value).
- Single return (Mid-360 reports the FIRST return only; research-track note): each laser shot yields at most
  one point, so fog can never add points beyond the 20,000 shots per frame. A fog echo either REPLACES the
  surface point or, on a ray that had no surface return (open door, glass, out of range), fills an empty
  shot. That is why points per 1 s window rise from 170.7k to at most 195.8k (= 200k minus 2 % dropout)
  and not beyond.
- Return rule. The literal Mid-360 behaviour is "first": the earliest echo above threshold wins, so any
  detectable fog echo in front of the surface replaces it. "strongest" (Hahner 2021, built for a Velodyne in
  strongest-return mode) keeps the surface unless the fog peak is larger. The banks used for the Jetson
  cost runs were generated with "strongest" (the earlier default); the entropy predictions below are given
  for BOTH, and the pre-registration headline uses "first". For cost this choice matters little: both
  rules span the same range of point counts and voxel counts (7k-30k voxels).

Fog (per ray, extinction alpha [1/m], MOR = ln(20)/alpha = 3.0/alpha)
- Hard return: P_hard = rho cos(i) / R0^2 * exp(-2 * optical depth to the target).
- Soft return: fog impulse response S(r) = pi * beta * xi(r) * T2(r) / r^2 for r < R0, with
  beta = 0.046/MOR (Rasshofer), convolved with a sin^2 pulse of half-width tau_H.
- Choice: "strongest" (Hahner): the fog echo replaces the surface echo when its peak is larger.
  Alternative "first" rule is run as a variant.
- Detection: the surviving echo must pass P_th; fog echoes must also pass a near-field gate.
- ASSUMED and influential (varied in `fog_variants.csv`):
  - receiver overlap xi(r): linear ramp 0 at 0.1 m (the blind zone) to 1 at 0.5 m. Hahner used
    0.9-1.0 m for a Velodyne HDL-64. The Mid-360 value is unknown.
  - pulse half-width tau_H = 5 ns (Hahner: 20 ns for the Velodyne).
  - near-field gate rho_gate = 0.0012 on apparent reflectivity P*R^2 of a fog echo. Without any gate,
    target-less rays (open door, glass) fill with fog points at any trace of haze. The value is set so
    that this clutter starts around MOR 45 m, the clearest fog in which Bijelic et al. 2018 saw
    near-sensor clutter on Velodyne units. It is a stand-in for the real receiver's unknown near-range
    sensitivity control.
  - fog range is drawn from S(r) (not Hahner's argmax + 2^U(-1,1) noise); gamma speckle (shape 4) on the
    fog echo power; firmware flags 60 % of fog points in tag bits 2-3 (the real classifier is
    proprietary; this number is a guess).
- Plume option: extinction = background + core * Gaussian(sigma_xy 0.45 m) * exp(-height/0.6 m) above
  `dynamic_slots.fog_source`.

Fog levels (ASSUMED mapping; nobody has measured the mist maker). Density is stated as the extinction
coefficient alpha at 905 nm, assumed spatially uniform, wavelength-independent between visible and 905 nm
(true for droplets much larger than 1 um), with visibility = meteorological optical range MOR = ln(20)/alpha
= 3.0/alpha (Koschmieder, 5 % contrast). Two-way transmission to the 1.5-2 m room walls at B1/B2/B3:
0.74-0.67 / 0.41-0.30 / 0.05-0.02. Cross-check with the research track: two-way loss at 4 m is 0.62 at
50 m visibility and 0.008 at 5 m; this model gives exp(-2*0.06*4) = 0.62 and exp(-2*0.6*4) = 0.008.

| Level | alpha [1/m] | MOR [m] | Meaning |
|---|---|---|---|
| B0 | 0 | inf | clear, person at mark |
| B1 | 0.10 | 30 | light haze |
| B2 | 0.30 | 10 | clearly visible fog |
| B3 | 1.00 | 3 | dense, room-filling |
| P1/P2/P3 | plume core 1 / 3 / 10 | - | mist only above the source, clear elsewhere |

Liquid water needed for a uniform alpha (droplet radius 2.5 um assumed, alpha = 1.5 LWC / (rho_w r)):
B2 needs 0.5 g/m^3, B3 1.7 g/m^3 suspended, i.e. about 13 g and 43 g of airborne droplets in the
17 m^3 room, on top of roughly 9.7 g/m^3 (about 170 g) of vapour to saturate 22 C air from 50 % RH so
droplets stop evaporating. A hobby ultrasonic mist maker (order 300 mL/h) therefore needs tens of
minutes with the door shut to reach B2-B3 uniformly; a local plume is the more likely real condition.
These are back-of-envelope figures, not measurements.


Scene. Banks and sweeps use the room_v1 draft that room-3d published at 21:57 (copy in `data_local/scene/`).
The final room_v1 (room-3d branch) moves the door and furniture by 3-6 cm; not worth regenerating 40
banks for. `scene/room_v0.json` (closed room) is run as a model variant.

## Results (all fog numbers are SIMULATION-ONLY predictions)

All tables are generated by `python -m synthlab.lidar_sim.summarize` from the CSVs in
`results/lidar-sim/` (`summary_tables.md`, `summary.json`). Every window is exactly 10 simulated
frames (1 s, 200k shots), HL parameters as the teammate's script (leaf 5 cm, radius 0.20 m, k 10-150).

### 1. Simulator and fog model checks
- `tests/test_lidar_sim.py`: 12/12 pass (Beer-Lambert two-way loss, single-return bound: points per frame
  never exceed shots, tags, plume locality, entropy limits line / plane / volume, bank determinism).
- First-return fact holds in every output: points per 1 s window never exceed 195.8k of 200k shots
  (`fog_sweep*.csv`, `n_raw`). Fog does not add points; it converts empty shots into fog points and
  replaces surface points.

### 2. Entropy vs condition (`conditions.csv`, 5 windows each; mean +/- SD across windows)

| cond | H_eigen | H_DS | mean_S | frac_high_S | particle-tag frac | n_voxel |
|---|---|---|---|---|---|---|
| A0 empty | 0.834 +/- 0.001 | 0.726 +/- 0.001 | 0.105 | 0.017 | 0.005 | 11743 +/- 55 |
| A1 = B0 person | 0.836 +/- 0.001 | 0.731 +/- 0.003 | 0.108 | 0.018 | 0.005 | 11773 +/- 48 |
| A2 people + warm objects | 0.848 +/- 0.001 | 0.766 +/- 0.003 | 0.128 | 0.056 | 0.005 | 12103 +/- 74 |
| A3 + clutter | 0.852 +/- 0.002 | 0.802 +/- 0.002 | 0.140 | 0.078 | 0.005 | 11375 +/- 68 |
| B1 alpha 0.1 (MOR 30 m) | 0.888 +/- 0.001 | 0.772 +/- 0.003 | 0.204 | 0.209 | 0.077 | 15352 +/- 60 |
| B2 alpha 0.3 (MOR 10 m) | 0.932 +/- 0.001 | 0.802 +/- 0.001 | 0.285 | 0.362 | 0.174 | 19405 +/- 48 |
| B3 alpha 1.0 (MOR 3 m) | 1.044 +/- 0.001 | 0.780 +/- 0.002 | 0.566 | 0.821 | 0.571 | 17057 +/- 33 |
| P1/P2/P3 plume | 0.892 / 0.894 / 0.910 | 0.761 / 0.758 / 0.751 | 0.22-0.26 | 0.24-0.29 | 0.09-0.18 | ~15.6k |

(B rows use the strongest-echo banks that were costed on the Jetson.) Scene complexity (A0 -> A3) moves
H_eigen by +0.018 and H_DS by +0.076; fog moves H_eigen by +0.05 to +0.21. Fog is a much larger
"entropy" stimulus than people and clutter.

### 3. Fine fog sweep: which metric responds, and the sign (`fog_sweep.csv`, `fog_sweep_first.csv`)
15 extinction levels (0 to 3 /m), 3 windows each, both return rules. Figure `fig_hl_vs_fog.png`.
- H_eigen, mean_S, frac_high_S and the particle-tag fraction RISE with fog (sign +), by 100-2500 clear-air
  window SDs. Fog droplets are isolated scatterers, so their neighbourhoods are volumetric (lambda1 ~
  lambda2 ~ lambda3): H_eigen climbs toward ln 3 = 1.10 and mean_S toward 0.6.
- H_DS is NOT a reliable fog indicator: +0.04 to +0.07 and non-monotonic (first rule: peak +0.07 at
  alpha 0.05, then falls; strongest rule: peak at 0.45). As Sam's script itself notes, H_DS (entropy of the
  dimensionality labels L/P/S) is not an isotropy measure; it is maximal when linear, planar and scatter
  labels are equally mixed, so it rises while fog is a minority and falls once scatter dominates.
  Pre-register H_eigen (or mean_S / frac_high_S) as the fog metric, not H_DS.
- Knee (Hyp C on the entropy side): yes, a saturating curve. Under the literal first-return rule the onset is
  between alpha 0.02 and 0.05 /m (MOR 150-60 m, i.e. faint haze) and H_eigen saturates (~1.05) by alpha
  0.2. Under the strongest rule onset ~0.05, half-rise ~0.45, saturation ~1.0. Above alpha ~1 the fog wall
  hides the room and the metrics drift back down slightly.
- Point and voxel count are NON-monotonic: n_voxel x2.55 at alpha 0.1 (first rule) / x1.9 at 0.45
  (strongest), then falls below clear (x0.63 at alpha 3) when only near-sensor fog remains.
- The onset alpha is the most model-dependent number: across 9 variants (`fog_variants.csv`,
  `fig_fog_model_variants.png`) the +0.02 onset ranges 0.01-0.45 /m, driven by the assumed near-field
  gate and receiver overlap. The saturated magnitude (dH_eigen +0.13 to +0.24 at alpha 1) is robust.

### 4. Pre-registration table for Experiment B (predicted change vs B0; headline = first-return rule, 3 windows; [range over all 9 model variants and both rules])

| level | d H_eigen | d H_DS | d mean_S | d frac_high_S | d particle-tag frac | n_voxel ratio |
|---|---|---|---|---|---|---|
| B1 (0.1 /m, MOR 30 m) | +0.182 [-0.000, +0.184] | +0.057 [-0.001, +0.058] | +0.388 [-0.000, +0.390] | +0.678 [-0.001, +0.681] | +0.448 [-0.000, +0.449] | x2.55 [1.00, 2.56] |
| B2 (0.3 /m, MOR 10 m) | +0.214 [+0.007, +0.215] | +0.038 [+0.011, +0.085] | +0.481 [+0.010, +0.482] | +0.834 [+0.022, +0.835] | +0.581 [+0.004, +0.581] | x2.18 [1.05, 2.20] |
| B3 (1.0 /m, MOR 3 m) | +0.219 [+0.067, +0.241] | +0.034 [+0.033, +0.081] | +0.495 [+0.134, +0.498] | +0.872 [+0.261, +0.874] | +0.594 [+0.097, +0.594] | x1.35 [0.77, 1.97] |
| clear-air SD of one 1 s window | 0.0012 | 0.0019 | 0.0014 | 0.0019 | 0.0002 | 58 voxels |

Reading: sign is + for every entropy metric at every level; for B2-B3 even the weakest variant gives a
shift of > 5 window SDs in H_eigen. B1 may produce no effect at all if the real receiver's near-field
suppression is strong (lower bound 0). The particle-tag fraction saturates at 0.60 only because the
firmware flag rate is ASSUMED 60 %; its real value is unknown, but the real clear-air value (0.0046) is
matched. Recommended pre-registered test: one-sided, H_eigen(Bk) > H_eigen(B0), with a measured MOR
(or transmission) per trial so the result can be placed on this curve.

### 5. Method sensitivity (`hl_sensitivity.csv`, `fig_hl_sensitivity.png`)
The fog effect on H_eigen keeps its sign for every window (0.1-2 s), leaf (2-10 cm) and radius
(0.1-0.4 m) tried (+0.09 to +0.40). H_DS changes sign with the leaf (-0.19 at 2 cm, +0.17 at 8 cm) and
radius: another reason not to use it for fog.

### 6. Clear-air sanity check against the teammate's real Mid-360 summaries (`real_reference.json`; different room, AGX)

| metric | real lab M0 empty | sim A0 empty | real M1 people/clutter | sim A3 |
|---|---|---|---|---|
| H_DS | 0.770 | 0.726 | 0.776 | 0.802 |
| H_eigen | 0.823 | 0.834 | 0.826 | 0.852 |
| mean_S | 0.130 | 0.106 | 0.135 | 0.140 |
| frac_high_S | 0.129 | 0.017 | 0.137 | 0.078 |
| n_voxel | 12146 | 11743 | 12142 | 11375 |
| particle-tag frac | 0.0046 | 0.0046 (copied) | 0.0047 | 0.0046 |

Magnitudes agree within ~0.01-0.04 on H_eigen / H_DS / n_voxel. frac_high_S is 7x lower in simulation:
the simulator has too few isolated stray points in clear air (real sensors produce edge mixed pixels and
multipath that are not modelled beyond "glue" points). The real lab M1-M0 difference (+0.004 H_eigen) is
even smaller than the simulated A3-A0 (+0.018): scene clutter barely moves HL in reality either.

### 7. Cost on the Jetson (MEASURED on simulated input; `cost_runs_*.csv`, `cost_combined.{json,md}`)
Setup: shared harness `~/synthlab/harness/run_lidar_bank.py`, every run inside `tools/jetson.sh measured`,
Orin Nano MAXN_SUPER, 29 banks (A0-A3, B1-B3, P1-P3, 13 fog-sweep levels, 6 thinned controls; B0 = A1),
each bank looped from RAM for 12 s after 2 s warm-up, UNPACED (windows back to back, so VDD_IN is the
loaded power and energy per window = time-weighted VDD_IN x duration / windows), 3 s idle baseline before
each run, bank order shuffled (seed 11). Windows per run n = 17-54. Same ROI as the lab (z cut -0.3 to
1.3 m in the sensor frame), 10 frames per window. Excluded: driver, transport, bag decoding, disk.
Entropy and clustering were run as SEPARATE workloads so each has its own energy.

(a) HL entropy computation (`ent_unp`, 1 pass, 29 runs):

| condition | n_voxel | H_eigen | entropy ms / window | mJ / window (VDD_IN) | loaded VDD_IN W |
|---|---|---|---|---|---|
| B0 (= A1, clear, person) | 11820 | 0.837 | 374 | 2390 | 6.35 |
| A3 clutter | 11375 | 0.853 | 368 | 2460 | 6.63 |
| B1 (alpha 0.1) | 15338 | 0.889 | 480 | 3060 | 6.34 |
| B2 (alpha 0.3) | 19348 | 0.931 | 592 | 3770 | 6.37 |
| alpha 0.45 (peak voxels) | 22797 | 0.977 | 741 | 4710 | 6.33 |
| B3 (alpha 1.0) | 17123 | 1.044 | 572 | 3650 | 6.36 |
| alpha 3.0 (fog wall) | 7477 | 1.022 | 264 | 1700 | 6.43 |
| P3 plume | 15534 | 0.910 | 507 | 3260 | 6.38 |

- Clear-air 374 ms per 1 s window matches the teammate's live figure (~350 ms per window on the Nano).
- Time is set by the voxel count: r(ms, n_voxel) = 0.992, R^2 = 0.985, slope 33 ms per 1000 voxels
  (all 29 banks); r(ms, H_eigen) = 0.31. Energy per window: r = 0.991 with n_voxel, 0.30 with H_eigen;
  partial r(energy, H_eigen | n_voxel) = 0.41 for total energy and 0.02 for energy above idle.
- Matched point count (thinned controls): dense fog (H_eigen 1.03-1.04) vs clear (0.80-0.84) at equal voxel
  count costs 28.0 vs 26.9, 28.2 vs 26.9 and 31.6 vs 31.6 ms per 1000 voxels: fog adds at most ~5 % per
  voxel. A +0.2 jump in entropy by itself is nearly free.
- Loaded VDD_IN does not move (6.29-6.63 W over all banks, r with n_voxel -0.19): the workload is a
  single-threaded CPU loop, so fog changes HOW LONG the board is busy, not how hard it draws. In the live
  paced system (one window per second) this appears as extra average power = extra energy per second:
  +0.67 J/s at B1, +1.38 at B2, +1.26 at B3 vs clear (total energy differences from the table).
- Sign: cost goes UP for light-to-moderate fog (peak x1.98 time at alpha 0.45, strongest rule) and DOWN
  for very dense fog (x0.71 at alpha 3.0), because it follows n_voxel, which is non-monotonic in fog.
  Under the literal first-return rule the voxel peak is earlier and higher (x2.55 at alpha 0.1, i.e.
  B1); the fitted line (33.5 ms per 1000 voxels - 23 ms) predicts ~990 ms, +610 ms (x2.6) at that point. Predicted, not measured: those banks
  were not costed.

(b) Genuine LiDAR consumer: voxelise + Euclidean clustering + person-candidate filter (`clu_unp`, 1 pass,
29 runs, n = 137-367 windows per run). ms / mJ are the whole workload per 1 s window.

| condition | n_voxel | cluster ms / window | mJ / window | loaded VDD_IN W | person candidates / window |
|---|---|---|---|---|---|
| B0 (= A1) | 11820 | 65.9 | 403 | 6.04 | 1.00 |
| A3 clutter | 11375 | 67.3 | 415 | 6.03 | 1.40 |
| B1 | 15338 | 81.8 | 497 | 5.98 | 0.00 |
| B2 | 19348 | 95.9 | 587 | 6.03 | 0.00 |
| alpha 0.45 | 22797 | 111.5 | 681 | 6.03 | 0.00 |
| B3 | 17123 | 104.8 | 642 | 6.06 | 0.40 |
| alpha 3.0 | 7477 | 79.7 | 496 | 6.12 | 0.19 |

- The consumer costs ~6x less than measuring entropy (66 vs 374 ms, 0.40 vs 2.39 J per window in clear air).
  Fog raises it by +24 % (B1), +46 % (B2), +59 % (B3); r(ms, n_voxel) = 0.87, slope 5.2 ms per 1000 voxels,
  partial r(ms, H_eigen | n_voxel) = 0.40 (fog merges into large clusters, so the pair search does more
  work per voxel; matched-count pairs: 15.0 vs 19.4 and 27.3 vs 32.6 ms, but 65.9 vs 59.9 at ~12k voxels).
- Fog also BREAKS the consumer: the one person is no longer a candidate at B1-B2 (0.00 per window),
  because fog points glue onto the body cluster. A functional loss, not only a cost change.

### 8. What this means for the hypotheses (simulation + measured cost on simulated clouds)
- Hyp B (fog raises HL, power rises): HL side supported in simulation for H_eigen / mean_S / frac_high_S
  (sign +, robust across 9 model variants); NOT for H_DS. Power side: loaded VDD_IN is flat (+/- 0.3 W);
  what rises is busy time and therefore energy per window, by up to x2 for HL and x1.6 for clustering.
- The audit worry is confirmed in size: in clear air, measuring entropy costs 2.39 J per window, the only
  genuine consumer 0.40 J. Of the fog-induced extra energy at B2 (HL +1.38 J, clustering +0.18 J), 88 % is
  the cost of measuring fog. Experiment B should report both workloads separately, or a power rise will
  mostly measure the entropy node.
- Entropy itself does not cost: at matched voxel count, H_eigen +0.2 changes HL time by 0-5 %. Cost is
  a function of occupied voxels (R^2 0.985), which fog raises then lowers. Analyse power vs n_voxel with
  entropy as a covariate (partial correlation), and log n_voxel per window in the real runs.
- Hyp C (knee): entropy saturates (knee in H_eigen); cost is non-monotonic (rise then fall), not a knee.
- Null: a flat loaded power is expected; a flat energy per window is NOT expected under moderate fog.

### 9. Not verified / limits
- No real fog data. Onset density depends on the unknown near-field sensitivity (overlap, gate): onset
  0.01-0.45 /m across variants. Scan pattern analytic (real Livox CSV not used). Firmware particle-tag
  rate assumed 60 %. frac_high_S in clear air is 7x lower than real.
- Cost: one pass per workload (n = 1 run of 17-367 windows per bank); between-pass repeatability not
  measured tonight. Energy above idle is noisy (3 s idle baseline; idle 4.67-5.44 W). Unpaced runs; live
  paced behaviour inferred, not measured (one paced probe, `probe_paced_B3_both_summary.json`: B3 both workloads 1226 ms per window, overruns
  6/15 windows at 1 Hz).
- Jetson banks use the strongest-echo rule and the 21:57 room_v1 draft.

## Figures (`results/lidar-sim/`)
- `fig_clouds_clear_vs_fog.png`: simulated clouds, clear vs fog.
- `fig_hl_vs_fog.png`: HL metrics vs extinction, both return rules (knee, sign, H_DS non-monotonic).
- `fig_cost_summary.png`: mJ per window vs n_voxel / H_eigen / alpha for HL and for clustering.
- `fig_cost_vs_entropy_and_voxels_ent_unp.png`, `fig_cost_vs_fog_ent_unp.png`: HL ms per window.
- `fig_fog_model_variants.png`, `fig_hl_sensitivity.png`, `fig_conditions.png`, `fig_scan_pattern.png`.

