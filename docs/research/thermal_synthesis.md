# Ways to make synthetic LWIR data, and the Lepton 3.5 sensor model

Research track, night of 2026-11-23. The original project already holds detailed notes on
ThermalSynth (`docs/THERMALSYNTH_REPRODUCTION_PLAN.md`,
`docs/research_2026-11-16/thermalsynth_paper_review.md`) and on Lepton radiometry
(`docs/research_2026-11-10/radiometry_review.md`); this file does not repeat them. Verification
level is stated per item: "read" = page or PDF text read tonight; "snippet" = search summary
only; "derived" = my arithmetic, stored in `results/research/derived_numbers.json`.

## 1. Lepton 3.5 sensor model, with datasheet numbers

Source: FLIR Lepton Engineering Datasheet, Rev 200 (read; 83 pages, text extracted),
https://cdn.sparkfun.com/assets/f/6/3/4/c/Lepton_Engineering_Datasheet_Rev200.pdf . This is not
the latest revision; FLIR's current one was not fetched.

| Property | Value | Datasheet location |
|---|---|---|
| Array, pitch, band | 160x120, 12 um, 8-14 um, uncooled VOx | p. 8 |
| Lens | f/1.1 silicon doublet, 57 deg HFOV | p. 1 |
| NETD (random temporal noise) | < 50 mK, 35 mK typical, radiometry mode | Table 13, p. 68 |
| Lens MTF | >= 51 % on axis at Nyquist/2 | p. 68 |
| Frame rate | stream about 26.5 Hz, unique frames 8.7 Hz | pp. 8, 57 |
| TLinear | Kelvin x 100 at 0.01 resolution; 30000 = 26.85 C; all 16 bits used | p. 29 |
| Radiometric accuracy, high gain | greater of +/- 5 C or 5 % (typical) | p. 9 |
| Accuracy at 30 C ambient | +/- 7 C at 10 C scene, +/- 5 C at 50 C, +/- 5 C at 100 C; after per-unit calibration +/- 5 / 3 / 4 C | Tables 4-5, pp. 32-33 |
| Intra-scene range | high gain -10 to 140 C; low gain to 400 C | pp. 31-32 |
| FFC triggers | every 3 minutes (180,000 ms), or 1.5 C camera temperature change, and at start-up | p. 19 |
| FFC duration | nominally 23 frames at 26 Hz (about 0.88 s), integrating 8 frames; "FFC imminent" flag 52 frames (about 2 s) before; radiometry invalid during FFC | p. 19 |
| Pixel operability | > 99.0 %, single-pixel defects only, replaced from neighbours | Table 13 |
| Pipeline | NUC, defect replacement, spatial/temporal filtering with scene-based NUC, AGC | p. 13 |
| Dissipation | up to 160 mW | p. 68 |

Derived model parameters:

- Temporal noise: Gaussian, sigma 3.5 counts typical, 5 counts worst case, added before
  rounding to uint16.
- PSF: a Gaussian with MTF 0.51 at 0.25 cycles/pixel has sigma 0.74 px; including the
  pixel-aperture sinc (0.90) gives 0.79 px. Use 0.75-0.85 px. The datasheet figure is a lens
  minimum on axis, so real blur may be less.
- FFC: every 180 s, freeze about 8 unique frames, then reset the fixed-pattern component.
- Radiometric error: a slowly varying global offset of a few degrees, not per-pixel noise.

**Not in the datasheet:** residual fixed-pattern noise amplitude, its growth rate between FFCs,
column-noise amplitude, and the 1/f corner. The text says only that drift makes images
"grainier" or "blotchy" and that "columns ... may drift as a group" (p. 17). Any FPN number in a
simulator is an assumption unless measured from the team's own static recordings (per-pixel
temporal mean minus a spatial low-pass, tracked across an FFC). Microbolometer noise literature
(Johnson, 1/f and thermal-fluctuation noise; row/column noise attributed to read-out 1/f noise)
was found only at snippet level and gives no Lepton-specific values.

SBNUC relies on scene motion, so in a fixed-mount static scene it is ineffective and the
shutter does the work (p. 17). The team's set-up is exactly that case.

## 2. Fog and mist at LWIR

- Radiance model: `L = tau * L_scene + (1 - tau) * B(T_air)`, `tau = exp(-alpha d)`. Fog both
  attenuates and emits, so every pixel moves toward air temperature.
- Rivera Velazquez et al., "Analysis of Thermal Imaging Performance under Extreme Foggy
  Conditions", J. Imaging 8(11):306, 2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC9699133/
  (detail in the earlier project note): contrast loss under 10 % above 20 m visibility over a
  30 m chamber; all targets converge to fog radiance at 10 m.
- Extinction near 11 um is linear in liquid water content and nearly independent of droplet
  size: Chylek, "Extinction and Liquid Water Content of Fogs and Clouds", J. Atmos. Sci.
  35:296-300, 1978; verified by Pinnick et al. (1979) on 341 size distributions (snippet). The
  coefficient was not confirmed in a primary source: a patent on the method quotes about
  0.15 m^2/g at 11.5 um, and I recall about 0.13 m^2/g. Use `alpha = k * LWC`, k = 0.13-0.15
  m^2/g, labelled as assumed.
- A KTH thesis reports higher LWIR transmittance than visible/SWIR/MWIR in radiation fog (small
  droplets) but strong scattering even at IR in advection fog (large droplets) (snippet).
- Bridge to the LiDAR (derived, geometric-optics limit): at LWC 1 g/m^3 the 905 nm extinction is
  0.75 / 0.30 / 0.15 per metre for effective droplet radius 2 / 5 / 10 um, against 0.13-0.15
  per metre at LWIR. Fine mist hides walls from the LiDAR long before it dims the thermal image.
  This is the physical content of Hyp B, and it predicts a small HT change rather than none.

## 3. Emissivities (8-14 um)

Table compilations, about +/- 0.02 (snippet; ThermoWorks table
https://blog.thermoworks.com/emissivity/ and a materials table in arXiv 2608.02964): skin 0.98;
generic cloth 0.95; flat paint 0.94-0.97; wood 0.87-0.92; glass 0.91-0.94; plastic laminate
0.91; polished aluminium 0.04-0.15. Glass is opaque in LWIR and metal mirrors the room, so only
those two need a reflected-radiance term. No value specific to cotton or painted plasterboard
was found.

## 4. Approaches to generating data

| Approach | Examples | What it gets right | What it gets wrong for this project |
|---|---|---|---|
| Real-frame replay | Team's own recordings | Everything about the sensor | No new conditions |
| Perturbation of real frames | Hahner-style: apply a physical fog/heat model to recorded frames | Sensor noise, FPN and optics are real; only the perturbation is modelled | Needs per-pixel depth; perturbation model unvalidated |
| Compositing on real backgrounds | ThermalSynth (Madan et al., WACV-W 2023); Avi et al., "Infrared Dataset Generation for People Detection Through Superimposition of Different Camera Sensors", ICCV-W 2021 (Lepton; snippet); UAV thermal augmentation, arXiv 2507.06797 (snippet) | Backgrounds and noise are real | Seams, wrong blur and noise on the pasted object, no occlusion or shadow physics |
| Physics rendering | Blender LWIR with full Fresnel and Planck emission (J. Imaging 12(1):33, 2026, https://www.mdpi.com/2313-433X/12/1/33; snippet); VisionSIM (Jungerman et al., arXiv 2611.08368, Nov 2026; abstract read: models "history-dependent emission for thermal cameras", platform and indoor dataset to be released); `isaac-thermal-camera-simulation` (source-available; repo returned 404 tonight); a thermal camera pull request on a Project AirSim fork (snippet) | Full control of scene, temperature, fog | Needs surface temperatures and emissivities that nobody measured; sensor model must be added |
| RGB-to-thermal translation | ThermalGen (Xiao et al., NeurIPS 2025, arXiv 2509.24878, https://github.com/arplaboratory/ThermalGen; snippet); PID (Mao et al., Pattern Recognition 2025, arXiv 2407.09299, https://github.com/fangyuanmao/PID; snippet); ThermalDiff (J. Vis. Commun. Image Represent. 2025; snippet); Kieu et al. (earlier note) | Plausible appearance from abundant RGB | 8-bit AGC imagery at 256-640 px; no radiometric values, no sensor noise, no fog control |
| Thermal diffusion / few-shot fine-tuning | LoRA of FLUX.1 Kontext on 100 RGB-IR pairs, arXiv 2601.04381 (snippet); ThermVision-DB synthetic LWIR faces (snippet) | Works from little data | Same as above, plus memorisation risk on small sets |
| Procedural (warm blobs + noise) | The team's 279-trial sweep | Fast, fully controlled | Not shaped like people; detector response is unrealistic |

CARLA has no thermal sensor that this search found; classic AirSim's "infrared" camera is a
segmentation-ID trick, not radiometry (from memory, not re-checked).

## 5. Public low-resolution thermal person datasets

| Dataset | Sensor and size | Labels | Access |
|---|---|---|---|
| ChaLearn IPHD, Clapes et al., FG 2020, https://chalearnlap.cvc.uab.cat/dataset/34/description/ (read) | FLIR Lepton v3, 160x120 native (distributed 213x120), > 100K frames (84,818 / 12,974 / 15,115), paired with RealSense depth | Person boxes, YOLO format; train/val labels are weak cross-modal | Download scripts on a deprecated page; no licence stated; not downloaded |
| Drahansky et al., Sensors 2023, https://doi.org/10.3390/s23187822 (snippet; earlier note has a deep dive) | Lepton 3.5 indoor plus FLIR ADAS: 13,416 images, 53,628 people | Boxes | See earlier note |
| PAIR-LRT-Human, arXiv 2303.09279 (snippet) | Lepton 3.5, about 33K thermal-RGB pairs | Unclear | Availability unconfirmed |
| Zhu et al., SenSys 2021, https://dl.acm.org/doi/10.1145/3485730.3493692 (snippet) | 32x24, 1,770 images, people and warm objects indoors | Yes | Public per abstract |

IPHD is the closest public match to the team's sensor. It is non-radiometric (Lepton 3, AGC
8-bit), so it can validate detector behaviour and image statistics but not temperatures.

## 6. How papers validate sim-to-real for thermal

- Downstream task on real test data is the norm: ThermalSynth trains fall detectors, Kieu et
  al. and ThermalDiff report pedestrian detection on real thermal images, Hahner reports AP on
  real fog.
- Image-level scores (FID, PSNR, SSIM, LPIPS) against paired real frames for translation models.
- Mixed real + synthetic training usually beats synthetic alone; a UAV thermal study warns that
  synthetic-only models degrade when simulation settings differ from deployment (snippet,
  https://www.mdpi.com/2313-433X/11/12/436).
- Nobody found validates sensor noise, radiometric values or fog against real captures.

For this project the question is not "does it train a better detector" but "does it load the
pipeline like real data". The matching validation is distributional: on held-out real
recordings versus synthetic frames, compare (a) YOLOv8n detection count and confidence,
(b) BIE and its components, (c) temporal noise sigma on static pixels, (d) spatial power
spectrum, and (e) the measured post-processing time per frame.

## What to do with this tonight

- Physics track: use the sensor parameters in section 1 and the fog model in section 2; label
  FPN and k as assumptions.
- Data-driven track: prefer perturbing and compositing real Lepton frames; it is the approach
  with the strongest precedent and the smallest unmodelled gap.
- Generative track: report nearest-neighbour distance to training frames and the five checks
  above; a blurry generator will inflate BIE.
- All tracks: generate at 8.7 Hz semantics (no more unique frames per second than the sensor
  gives) and include FFC freezes in at least one bank.

## What to do with this next month

- Measure FPN and its drift from a 10-minute static recording spanning three FFCs; replace the
  assumed values.
- Record a real mist sequence with a blackbody-like reference (a matte warm plate) at known
  distance to fit k directly.
- Download IPHD (after checking terms) and run YOLOv8n on it as an external check on detector
  behaviour at 160x120.
- Fetch FLIR's current datasheet revision and confirm nothing in section 1 changed.
- Read the Chylek and Pinnick papers for the exact coefficient.
