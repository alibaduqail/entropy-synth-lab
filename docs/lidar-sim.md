# lidar-sim: Livox Mid-360 + fog simulator in the test room (first look at Hypothesis B)

Status: IN PROGRESS (overnight run 2026-11-23). Sections marked TODO are filled as results land.
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
- Single return: a fog echo REPLACES the surface point. Points per frame can never exceed 20,000.

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

Fog levels (ASSUMED mapping; nobody has measured the mist maker)

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

## Results so far

TODO (filled below as files land in `results/lidar-sim/`).
