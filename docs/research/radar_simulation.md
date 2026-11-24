# FMCW radar simulation, the IWR6843ISK, and fog at 60 GHz

Research track, night of 2026-11-23. Sources fetched tonight unless marked.

## IWR6843ISK parameters

TI EVM user guide SWRU546E, antenna table
(https://www.ti.com/document-viewer/lit/html/SWRU546E/GUID-868BBA8D-49E8-4A86-B97C-66CF39ECC363 ,
column "xWR6843ISK"):

| Item | Value |
|---|---|
| Tuning range | 60-64 GHz |
| Channels | 3 TX, 4 RX, antenna on PCB |
| Azimuth FOV | +/- 60 deg (6 dB beamwidth of the combined TX and RX pattern) |
| Elevation FOV | +/- 15 deg |
| Azimuth angular resolution | 15 deg |
| Elevation angular resolution | 58 deg |
| Antenna gain | 7 dBi |

Transmit power of about 10-12 dBm appears in TI forum and reference-design text; no official
EIRP figure was found (unverified).

TI indoor people-counting chirp, reference design TIDUE71D Table 2
(https://ti.com/document-viewer/lit/html/TIDUE71D/use-case-geometry-and-sensor-considerations-id-77756c08-381f-4f2b-b175-f6615b7047e0.html):

| Parameter | Value |
|---|---|
| Start frequency | 60.75 GHz |
| Ramp slope | 57 MHz/us |
| ADC samples per chirp | 96 at 2.95 Msps |
| Valid sweep bandwidth | 1780 MHz |
| Ramp end time / chirp repetition | 62 us / 184 us |
| Chirps per frame | 288 |
| Active frame time / frame period | 25.5 ms / 55 ms |
| Range resolution / max range | 8.4 cm / 8.1 m |
| Velocity resolution / max velocity | 0.324 km/h / 16.2 km/h |

Check (my arithmetic): `c / (2 * 1.78 GHz) = 8.4 cm`. A full 4 GHz sweep gives 3.75 cm. The
exact `profileCfg` text line could not be extracted (TI's PDF was not machine-readable), so any
cfg line written from these numbers is a reconstruction.

## Fog attenuation at 60 GHz

ITU-R P.840-9 (2023),
https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.840-9-202308-I%21%21PDF-E.pdf : specific
attenuation `gamma = K_l * M` in dB/km, with `M` the liquid water density in g/m^3 and

```
K_l = 0.819 f / (eps'' (1 + eta^2)),   eta = (2 + eps') / eps''
eps'  = (e0 - e1)/(1 + (f/fp)^2) + (e1 - e2)/(1 + (f/fs)^2) + e2
eps'' = f (e0 - e1)/(fp (1 + (f/fp)^2)) + f (e1 - e2)/(fs (1 + (f/fs)^2))
e0 = 77.66 + 103.3 (theta - 1),  e1 = 0.0671 e0,  e2 = 3.52
fp = 20.20 - 146 (theta - 1) + 316 (theta - 1)^2,  fs = 39.8 fp,  theta = 300 / T
```

I evaluated the formula tonight (a short Python calculation, not saved as a result file):

| f (GHz) | K_l at 20 C | K_l at 0 C |
|---|---|---|
| 60 | 1.75 | 2.49 |
| 64 | 1.96 | 2.73 |
| 77 | 2.71 | 3.54 |

Units: (dB/km)/(g/m^3). Typical fog water content from earlier editions of P.840 (search
snippet): about 0.05 g/m^3 for medium fog (visibility about 300 m) and 0.5 g/m^3 for thick fog
(about 50 m).

So thick fog costs about 0.9 dB/km at 60 GHz and 20 C. Over a 4 m target (8 m round trip) that
is 0.007 dB. A mist ten times denser gives 0.07 dB. For scale, oxygen absorption near 60 GHz is
roughly 15 dB/km, or 0.12 dB over the same path (standard ITU-R P.676 figure, not re-fetched).

**Conclusion: at 1-4 m, fog is invisible to the radar.** The radar is a negative control: its
point count and SNR should not change between B0 and B3. Plausible second-order effects are a
water film on the radome and air movement of the mist; neither was found quantified.

No fog-chamber measurement at 60-77 GHz was found. Golovachev et al., "Millimeter Wave High
Resolution Radar Accuracy in Fog Conditions - Theory and Experimental Verification", Sensors
18(7):2148, 2018, https://www.mdpi.com/1424-8220/18/7/2148 , measured attenuation and group
delay in artificial fog, but at 320-330 GHz.

## Simulation tools

| Tool | What it does | Fit for tonight |
|---|---|---|
| RadarSimPy, https://github.com/radarsimx/radarsimpy | GPL-3.0. FMCW and TDM-MIMO baseband from point targets and meshes, phase noise, CFAR, DoA. Prebuilt for Linux ARM64 | Needs Python 3.11+ and NumPy 2.0+, so it does not fit the Jetson's Python 3.10 / numpy 1.26 environment without a separate venv |
| witwin-radar, https://github.com/witwin-ai/witwin-radar | GPU, differentiable, torch DSP to point cloud | Python 3.10+; untested |
| radar-basics-sim, https://github.com/Kymopsis/radar-basics-sim | Teaching code: chirps, MIMO, range-Doppler, angle FFT | Idealised; a reference for structure |
| Own numpy model | Sum of complex beat tones per TX-RX pair plus thermal noise, then range FFT, Doppler FFT, CA-CFAR, angle FFT | The practical route |

RF Genesis, Shenron and MATLAB Radar Toolbox were on the list to check and were not reached.

## What this means for the hypotheses

Radar gives the cleanest test of the null in Hyp B: fog changes nothing in the input, so any
power change in the radar chain during fog trials is a system artefact (temperature, shared
load), not an input effect. That makes it a useful covariate for the whole experiment.

## What to do with this tonight

- Build the numpy point-target model with the TIDUE71D chirp, 3 TX x 4 RX, +/- 60 deg by
  +/- 15 deg antenna weighting, and thermal noise.
- Apply fog as a 0 dB change and say so; vary only the scene (people, clutter).
- Label the cfg line as reconstructed.

## What to do with this next month

- Pull the actual `.cfg` from the team's radar and replace the reconstructed chirp.
- Log radar point count and SNR during real fog trials as the negative control.
- Check radome wetting: a before/after static-reflector SNR measurement is enough.
