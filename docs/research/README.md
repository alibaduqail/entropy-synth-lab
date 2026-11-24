# Research synthesis: synthetic sensor data for the entropy-cost study

Night of 2026-11-23, branch `research`. One page; details and full citations are in the topic
files. Nothing here was measured on a pipeline except the INA3221 probe and the small
calculations in `results/research/`.

| File | Topic |
|---|---|
| `thermal_synthesis.md` | Lepton 3.5 sensor model, LWIR fog, generation approaches, datasets, validation |
| `lidar_fog_simulation.md` | Hahner fog model, indoor ranges, Mid-360 facts and scan pattern |
| `radar_simulation.md` | IWR6843ISK chirp, 60 GHz fog attenuation, tools |
| `sponge_and_energy_attacks.md` | Where input-dependent cost comes from; novelty check |
| `entropy_metrics.md` | What dos Santos and Ayunts define; why BIE falls with noise; alternatives |
| `sim_to_real_measurement.md` | Replay precedent, INA3221 limits, statistics |
| `room_reconstruction.md` | Metric room geometry (short) |

## Ranking of synthetic-data approaches for this project

The goal is to load the real Jetson pipeline the way real conditions would, so the ranking
criterion is "smallest unmodelled gap between the synthetic input and what the sensor would
deliver", not image quality.

1. **Physical perturbation of real recordings** (fog, heat or extra noise applied to recorded
   Lepton frames and recorded or ray-cast Mid-360 frames). Sensor noise, optics and fixed-pattern
   structure stay real; only the perturbation is modelled. This is the one approach whose
   precedent was validated against the real condition (Hahner et al., ICCV 2021,
   https://ar5iv.labs.arxiv.org/html/2108.05249).
2. **Compositing people and warm objects onto real backgrounds** (ThermalSynth, WACV-W 2023;
   Avi et al., ICCV-W 2021). Best way to vary scene content (conditions A0-A3) without a person
   in the room. Weak points are seams and doubled noise.
3. **Physics rendering of the room plus a Lepton sensor model.** Full control and the only way
   to get fog and geometry consistent across thermal, LiDAR and radar. Its sensor model rests on
   datasheet values (NETD 35-50 mK, MTF 51 % at Nyquist/2, FFC every 180 s) plus assumed
   fixed-pattern noise and assumed surface temperatures.
4. **Generative models** (ThermalGen, NeurIPS 2025; PID, Pattern Recognition 2025; few-shot
   LoRA). Published work targets 8-bit AGC imagery at higher resolution, conditioned on RGB.
   None models radiometric values, sensor noise or fog. Useful for appearance diversity; weakest
   for a measurement study.
5. **Procedural blobs and noise.** Keep as controls (flat, noise-only, zero-detection frames),
   not as stand-ins for scenes.

For LiDAR, ray-cast the room with the published Mid-360 scan pattern and apply the Hahner model
with alpha swept to indoor values (0.02-1.0 per metre). For radar, fog is a 0 dB effect at 1-4 m
(ITU-R P.840-9: 1.75 dB/km per g/m^3 at 60 GHz and 20 C), so the radar is a negative control.

## Top threats to novelty

1. **Natural scene and weather dependence of detector latency on a Jetson is already
   published.** Liu, Wang, Shi, arXiv 2209.05487 (https://arxiv.org/pdf/2209.05487):
   post-processing time correlates with detected-object count (r = 0.43 for YOLOv3 up to 0.98
   for Faster R-CNN), and rendered rain *lowered* inference time. The team's
   r(detections, yolo_ms) = 0.87 replicates this.
2. **"Input entropy changes power on a fixed workload" is already published for GPUs.**
   Bhalachandra et al., arXiv 2212.08805 (up to 67 % on an A100); Gregersen et al., arXiv
   2409.18324 (almost 40 %). These also mean the null "fixed graph => constant power" is not
   guaranteed by physics.
3. **The field's own definition excludes natural inputs.** The September 2026 survey (Gu et
   al., arXiv 2609.36732) sets "naturally hard instances" apart from latency attacks and counts
   only projector attacks as physical. The sponge framing holds only if the condition is
   attacker-inducible.
4. **NMS attacks are weak on YOLOv8 even when optimised.** Overload moves YOLOv8s on a Jetson
   Orin NX from 30.1 to 33.2 ms (arXiv 2412.02171); a NeurIPS 2025 re-evaluation finds the
   needed proposal counts unreachable with GPU NMS (via the survey). A natural effect must be
   smaller.

Still open, and the claim to make: no published work logs board power on a Jetson while a real
obscurant changes the input to a thermal-plus-LiDAR pipeline.

## Top threats to validity

1. **The effect is below the sensor's resolution.** One VDD_IN step is 39.5 mW (5 mOhm shunt,
   8 mA steps, measured tonight); reported effects are 36-73 mW and temperature drift is about
   45 mW (audit track). Internal sensors can also be off by up to 50 % in absolute terms
   (Shalavi et al., IEEE EDGE 2023, arXiv 2306.13107).
2. **HT is not a disorder measure.** BIE is a contrast-quality score that falls when noise is
   added (0.082 to 0.025 in a synthetic check while Shannon entropy rose from 2.45 to 7.08
   bits). See `entropy_metrics.md`.
3. **HL's cited source defines something else.** dos Santos et al. 2019 define the entropy of
   linearity, planarity and scattering for building detection, not eigenentropy, and say
   nothing about disorder or cost. The metric also depends on neighbourhood size and density.
4. **Detection count, not entropy, is the causal variable** in the literature and in the
   team's data; entropy must be tested after controlling for it.
5. **Fog can lower cost.** The Mid-360 is first-return only, so fog cannot add points; fog
   also removes thermal detections. The direction of Hyp B's power change is not given.
6. **Unvalidated simulators.** No extinction coefficient for humidifier mist at 905 nm or LWIR,
   no Mid-360 overlap function, and no Lepton fixed-pattern noise figure were found. Every
   simulated fog result inherits those assumptions.
7. **The metrics are themselves the heaviest input-dependent code** (BIE 34.5 ms per frame; HL
   about 350 ms per window) and must stay out of the measured loop.

## Not verified tonight

Items read only as search summaries or through a secondary source are marked in each file. The
main ones: the EVADE and DetStorm papers (read via the survey), the LWIR extinction coefficient
(0.13-0.15 m^2/g assumed), the Livox fog-chamber study, the exact Ultralytics zero-candidate
branch, Ayunts Eq. 5 from the publisher PDF, and all room-reconstruction accuracy claims.
