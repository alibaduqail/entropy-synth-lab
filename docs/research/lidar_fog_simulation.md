# LiDAR fog simulation and the Livox Mid-360

Research track, night of 2026-11-23. Sources fetched tonight unless marked. "My arithmetic"
marks numbers I derived; they are not from a paper.

## The fog model to implement (Hahner et al. 2021)

Hahner, Sakaridis, Dai, Van Gool, "Fog Simulation on Real LiDAR Point Clouds for 3D Object
Detection in Adverse Weather", ICCV 2021. https://ar5iv.labs.arxiv.org/html/2108.05249 ; code
https://github.com/martinhahner/lidar_fog_sim (`theory.py` plots a single beam).

- Transmit pulse: `P_T(t) = P0 sin^2(pi t / (2 tau_H))` for `0 <= t <= 2 tau_H`; `tau_H = 20 ns`
  assumed for the Velodyne HDL-64E.
- Hard target in fog: `i_hard = i_clear * exp(-2 alpha R0)`.
- Soft (fog) return:
  `P_soft(R) = C_A P0 beta * Int_0^{2 tau_H} sin^2(pi t / (2 tau_H)) * exp(-2 alpha r) / r^2 * xi(r) * U(R0 - r) dt`,
  with `r = R - c t / 2`, `xi` the receiver overlap function and `U` the step function. No closed
  form; the paper integrates with Simpson's rule.
- Decision: `i_soft = max_R P_soft(R)`. If `i_soft > i_hard` the point moves to the range of that
  maximum (scale xyz by `R_tmp / R0`) and is further scaled by a noise factor `2^p`,
  `p ~ U(-1, 1)`. Otherwise it keeps its position with intensity `i_hard`.
- Parameters: `beta = 0.046 / MOR` (from Rasshofer et al.); `beta_0 = 1e-6 / pi` (empirical);
  alpha in {0, 0.005, 0.01, 0.02, 0.03, 0.06} per metre for MOR = inf, 600, 300, 150, 100, 50 m,
  which is `alpha = ln(20) / MOR = 2.996 / MOR`.
- Validation in the paper: training with the simulated fog raised car AP (moderate, IoU 0.5) on
  real dense fog from the Seeing Through Fog set by +1.0 to +4.6 points across five detectors
  (for example PointPillars 35.23 -> 39.14). That is the standard of sim-to-real evidence in this
  field: a downstream gain on real fog, not a point-level match.

Origin of the physics: Rasshofer, Spies, Spies, "Influences of weather phenomena on automotive
laser radar systems", Advances in Radio Science 9:49-60, 2011.
https://ars.copernicus.org/articles/9/49/2011/ars-9-49-2011.pdf (Mie-theory model, verified by
the authors against fog, rain and snow measurements; equations not re-read tonight).

## Does fog matter over 1-4 m indoors?

My arithmetic, two-way loss `exp(-2 alpha R)`:

| Visibility (MOR) | alpha (1/m) | Loss at 2 m | Loss at 4 m |
|---|---|---|---|
| 50 m (outdoor "dense fog") | 0.06 | 0.79 | 0.62 |
| 20 m | 0.15 | 0.55 | 0.30 |
| 10 m | 0.30 | 0.30 | 0.09 |
| 5 m | 0.60 | 0.09 | 0.008 |
| 3 m | 1.0 | 0.018 | 0.0003 |

Outdoor-calibrated fog levels barely touch a 4 m room. A fog machine in a closed room easily
reaches 3-10 m visibility, where walls drop out and soft returns take over. So the simulator
should sweep alpha over about 0.02-1.0 per metre, well beyond Hahner's range.

Wavelength: Duthon et al., "Light Transmission in Fog: The Influence of Wavelength on the
Extinction Coefficient", Applied Sciences 9(14):2843, 2019,
https://www.mdpi.com/2076-3417/9/14/2843 : extinction at 905 nm and 1550 nm differ by under 10 %
(search-snippet level; the MDPI page returned HTTP 403). Visible-band visibility is therefore a
fair proxy for alpha at 905 nm.

**Not found:** any measured extinction coefficient for an ultrasonic humidifier plume or a
hobby fog machine at 905 nm. Treat alpha as an unknown to be swept, and measure it in the real
room with a known-reflectivity target at two distances (`alpha = ln(I1 R1^2 / (I2 R2^2)) / (2 (R2 - R1))`).
A physical bridge to the thermal track (my derivation, geometric-optics limit): for droplets
much larger than the wavelength, `alpha_905 ~ 1.5 * LWC / (rho_w * r_eff)`. With liquid water
content 1 g/m^3 and effective radius 2 um that gives 0.75 per metre (visibility about 4-5 m).

## What real fog looks like in LiDAR data

Bijelic, Gruber, Ritter, "A Benchmark for Lidar Sensors in Fog: Is Detection Breaking Down?",
arXiv 1912.03251 (IEEE IV 2018; venue from memory, unverified).
https://ar5iv.labs.arxiv.org/html/1912.03251 . Velodyne HDL-64 S2 and S3D, Ibeo LUX and LUX HD in
the CEREMA chamber at about 13, 24, 36 and 45 m visibility.

- Droplets near the sensor produce a cloud of high-intensity clutter points around it.
- Close-range point count rises above the clear-weather reference while far points vanish.
- "Below 40 m meteorological visibility, perception is limited to less than 25 m."
- Last return matches the clear scene better than strongest return.

Li, Duthon, Colomb, Ibanez-Guzman, "What happens to a ToF LiDAR in fog?", IEEE T-ITS (arXiv
2003.06660, 2020), https://arxiv.org/abs/2003.06660 : CEREMA chamber study with a model that
predicts the minimum visibility for successful ranging (abstract only read).

Livox in fog: a climatic-chamber comparison of Velodyne, Ouster, Livox, Cepton and AEye sensors
reports that the Livox unit held up well in fog (search-snippet level; ResearchGate returned 403,
paper title and Livox model not confirmed - unverified). No fog study of the Mid-360 was found.

## Livox Mid-360 facts

Spec page https://www.livoxtech.com/mid-360/specs :

| Item | Value |
|---|---|
| Wavelength | 905 nm, Class 1 |
| FOV | 360 deg x (-7 to +52 deg) |
| Point rate | 200,000 points/s, first return only |
| Frame rate | 10 Hz typical (20,000 points per 100 ms frame) |
| Range precision (1 sigma) | <= 2 cm at 10 m; <= 3 cm at 0.2 m |
| Angular precision (1 sigma) | < 0.15 deg |
| Blind zone | 0.1 m; 0.1-0.2 m "reference only"; dark or thin objects unreliable under 1 m |
| False alarm rate | < 0.01 % at 100 klx |
| Range | 40 m at 10 % reflectivity, 70 m at 80 % |
| IMU | ICM40609 |

The spec page itself warns that performance may degrade in heavy fog.

Tag byte, from the Mid-360 protocol page
https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/livox_eth_protocol_mid360.html :
bits 0-1 glue points between adjacent objects; bits 2-3 rain, fog, dust and other small
particles; bits 4-5 other; bits 6-7 reserved. Each field: 0 high confidence (normal), 1 medium,
2 low, 3 reserved. Points are int32 millimetres (type 1), 96 per packet, with uint8 reflectivity.
This matches the bank format in `AGENTS.md` (bits 2-3 = particle flag). How often real firmware
sets the flag in fog is not documented. Older Livox models used a different tag encoding, so do
not reuse Mid-40/Mid-70 code.

Two consequences of first-return-only output: fog can move or delete a point but never add one,
so points per frame are capped at 20,000; and a fog return replaces the wall behind it.

Scan pattern: Livox publishes the recorded pattern in its Gazebo plugin,
https://github.com/Livox-SDK/livox_laser_simulation (`scan_mode/mid360.csv`, 16.8 MB, columns
`Time/s, Azimuth/deg, Zenith/deg`, one row per shot; I read only the first rows). A community
fork has a re-measured `mid360-real-centr.csv`: https://github.com/fratopa/Mid360_simulation_plugin .

## Other weather simulators

- LISA: Kilic, Hegde, Sindagi, Cooper, Foster, Patel, "Lidar Light Scattering Augmentation
  (LISA): Physics-based Simulation of Adverse Weather Conditions for 3D Object Detection", arXiv
  2107.07004 (2021). https://arxiv.org/abs/2107.07004 . Hybrid Monte Carlo scatterer model for
  rain, snow and several fog types. Official code link not confirmed tonight.
- Snow: Hahner et al., "LiDAR Snowfall Simulation for Robust 3D Object Detection", CVPR 2022
  (pp. 16364-16374). https://arxiv.org/abs/2203.15118 ; code
  https://github.com/SysCV/LiDAR_snow_sim . Models individual flakes and wet ground; up to +2.1
  AP in heavy snowfall.
- Bijelic et al., "Seeing Through Fog Without Seeing Fog", CVPR 2020: the real fog dataset used
  to validate the above (covered in the earlier project note).
- Scene-level simulators with a Mid-360 pattern: `livox_gz_sim` for Gazebo Harmonic (uses the
  recorded patterns, output compatible with `livox_ros_driver2`; forum announcement, not tested),
  the official Gazebo 11 plugin above, and community configs for Isaac Sim (no built-in Mid-360
  model as of the forum thread found). CARLA and Blensor were not checked tonight.
- On a Jetson: none of Gazebo/Isaac/CARLA is needed. Ray-casting 20,000 rays per frame against a
  box-and-plane room in numpy, then applying the Hahner model per point, is cheap; the heavy
  simulators add nothing for a single static room.

## What this means for the hypotheses

- Hyp B needs fog to raise HL. The literature supports a mechanism: near-sensor clutter is
  volumetric, so local neighbourhoods there have high eigen-feature entropy, while lost wall
  points remove low-entropy planar neighbourhoods.
- Point count cannot rise, so any compute cost that scales with point count can only fall in
  fog. Cost that scales with neighbourhood search density could rise near the sensor.
- The unknown overlap function `xi(r)` of the Mid-360 controls where clutter appears. Every
  simulated fog result depends on that assumption.

## What to do with this tonight

- Use the Hahner model with alpha swept 0.02-1.0 per metre, `beta = 0.046 * alpha / 2.996`
  (same relation re-expressed), 2 cm range noise, 1 mm quantisation, 20,000 shots per frame.
- State the assumed `xi(r)` (suggest a ramp from 0 at 0.1 m to 1 at 0.5-1 m) and run a
  sensitivity check on it.
- Set tag bits 2-3 to 1 or 2 on soft returns, and report HL with and without tag filtering.
- Report HL together with points per frame and the fraction of points inside 1 m.

## What to do with this next month

- Measure alpha in the real room (two-distance target method above) for each mist level, and
  log visibility with a simple transmissometer (laser pointer and photodiode).
- Record a real fog bag and compare three numbers against the simulator: points lost versus
  range, radial histogram of near-field clutter, and fraction of points tagged.
- Read Li et al. (T-ITS) and Rasshofer in full for the pulse-shape and overlap details.
- Confirm the Livox climatic-chamber paper and whether Mid-360 `tau_H` is published anywhere.
