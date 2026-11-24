# Sponge search: how much cost can a thermal scene add on the real pipeline?

Track `sponge-search`, overnight run 2026-11-23/24. Pipeline under test (on the Orin Nano, via
`tools/jetson.sh measured`): TLinear -> Celsius -> fixed 10-50 C 8-bit mapping -> BIE (HT) ->
YOLOv8n PyTorch CUDA FP32 (imgsz 640, conf 0.25, iou 0.7, max_det 300), i.e. the deployed
`experimentA.py` functions, kept loaded. Power = tegrastats VDD_IN, 50 ms. All numbers below come
from files in `results/sponge-search/` (`paired_*.json`, `cloud_stats.json`, `runs/*/evals.jsonl`).

## Headline
- A physically plausible scene (people, warm objects, 21 C cold background, blur, mist) found by
  hardware-in-the-loop CMA-ES adds **+4.4 ms/frame [3.7, 5.0] (+6.8 %)** and **+30 mJ/frame
  [22, 38] (+6.5 %)** unpaced (n=16 interleaved rounds x 40 frames). At the 8.7 fps pace: **+5.8 ms
  [5.3, 6.2] (+7.8 %)**, energy per 115 ms period only **+7.5 mJ [1.2, 13.7] (+1.1 %)**, because the
  board idles between frames. Power (mW) does not rise (-23 mW [-116, +71] unpaced).
- Where it comes from: NOT inference (flat ~30.5 ms, even a bit lower) but post-processing (NMS
  and result handling): 1.7 -> 8.6 ms (+6.9 ms [6.85, 6.98]), driven by 4 detections / 37 pre-NMS
  candidates. The search's own score (+9.1 ms) overstated it: winner's curse on a 2-frame bank
  measured once per generation. The re-measurement on a 96-frame bank is the number to quote.
- Upper bound, non-physical pixel patterns (white-box gradient search, `wb_*` banks): wb_panel
  (coarse panel texture) +76.7 ms [76.1, 77.3] (+120 %), +472 mJ (+107 %); wb_pixels (per-pixel
  optimised) +909 ms (+1424 %), +5408 mJ (+1224 %). Power DROPS (-6 %, -13 %: the GPU idles while the
  CPU runs NMS), so the cost is time and energy, not watts. wb_pixels saturates max_det=300 with 3026
  pre-NMS candidates.
- Null hypothesis: holds for the network (inference does not depend on the input), fails for
  post-processing, which scales with candidate boxes surviving the confidence threshold.

## What was searched (all hardware-in-the-loop, ~5 min each unless noted, CMA-ES, k=16 frames/eval)
| search | tier | evals | result when re-measured (96-frame bank, 16 interleaved rounds vs baseline) |
|---|---|---|---|
| (a) max latency | plausible layout | 210 (+resume, no gain) | **+4.4 ms (+6.8 %)** |
| (b) max energy | plausible layout | 224 | +2.1 mJ [-4.7, 8.8], 0 detections: **no effect** (search optimum was a power-sampling outlier) |
| (c) stealthy (HT in empty-room band 0.011-0.020) | plausible layout | 224 | HT 0.022; +0.2 ms, +2.2 mJ: **no effect found** (with HT pinned to the empty band, the search found no scene with detections) |
| (d) decoupling (max HT at no cost) | plausible layout | see runs | HT 0.201 (15x the empty-room 0.013), +0.14 ms [-0.43, 0.71], -1 mJ: **HT up 15x, cost unchanged** |
| unconstrained 15x20 blocks, black-box | pixels | 1 run | no detections found in 270 s, +0.3 ms: black box in 300 dims failed |
| white-box gradient (earlier, from candidate count) | panel / pixels | n/a | see "Upper bound" above |
Paced re-measurement (8.7 fps) of (b)(c)(d): all within +-0.3 ms and +-9 mJ of baseline (CIs include 0).

## Noise floor
Same bank listed twice (A/A): 63.85 vs 63.98 ms/frame; sd about 0.5 ms across 8 rounds of 40 frames;
VDD_IN sd 90-150 mW; energy sd 8-12 mJ/frame. Controls (zeros, white, flat, noise, synthetic empty
room) all within 62-64 ms. One person detected: +1.2 ms post-NMS. Energy differences below about
+-10 mJ are not resolvable per frame.
Search-time objectives were noisier than that (k=16 frames, baseline re-measured once per
generation; zero-detection evals had d_ms sd 9.5 ms, likely contention/thermal). This is why finalists
were re-measured interleaved with many repeats.

## Cost vs entropy cloud (1478 timed evaluations from the 5 hardware searches)
- Cost (d_ms) vs HT: Pearson r = 0.015 (p=0.56); vs Shannon: r = 0.026; vs detections: r = 0.10 (p<1e-4,
  Spearman 0.45); vs pre-NMS candidates: r = 0.10 (Spearman 0.44). Regression on standardized
  predictors: candidates 0.79 ms/SD, Shannon 0.36, detections 0.28, HT -0.06 (R2 = 0.012 overall; noisy).
  Partial correlation of HT given detections and candidates: r = -0.01 (p=0.67).
- Among evaluations with detections (n=410): about +1.35 ms per detection [0.75, 2.14] and +11.5 mJ per
  detection [6.6, 17.3] (and in the controlled finalists about 1.7 ms per detection). Zero-detection
  evaluations (n=1068): slope in HT -0.02 ms per 0.01 HT [-0.13, 0.06].
- The decoupling finalist is a clean test: HT 15x higher at zero cost change.

Verdicts:
- **Hyp A (complexity raises HT and power):** HT rises with scene complexity (and with detections,
  r=0.27), but cost does not follow HT. Cost follows candidates/detections. Supported only through
  the detection channel, and power in mW does not rise. Not supported for HT as a cost driver.
- **Hyp B (HT flat while cost rises):** the reverse is demonstrated: HT can be made high with no cost
  (decoupling finalist); cost can be raised with HT moderately raised (0.096 vs 0.013, still modest).
  A stealth scene (cost up, HT in the empty band) was NOT found within plausible scenes; the
  white-box pixel scenes do have low HT (0.007) with +909 ms but are not physical.
- **Hyp C knee:** the cloud's best piecewise fit puts a breakpoint at HT 0.081 (and at 2 detections)
  but the BIC gain is tiny and R2 stays below 0.002 vs HT; treat as no evidence of a knee in HT. The
  only real nonlinearity is in candidates -> post-NMS time, not HT.
- **Null:** inference is constant; the null is falsified only by NMS post-processing, which is a
  few ms for a plausible scene.

## Which pipeline settings cap the sponge (`measure/variant_*.json`, 6 rounds x 8 frames)
| setting | wb_panel ms (post ms) | wb_pixels ms (post ms) |
|---|---|---|
| default (conf 0.25, max_det 300) | 142 (79) | 994 (927) |
| max_det 50 | 140 (77) | 967 (904) |
| max_det 10 | 138 (75) | 946 (883) |
| conf 0.5 | 111 (48) | 576 (510) |
max_det caps almost nothing: Ultralytics NMS sorts and filters all candidates first and only then
truncates. The confidence threshold is the effective cap, since it removes candidates before the
expensive step (conf 0.5 halves the worst case; a plausible scene with only ~38 candidates is
unaffected either way). Pre-filtering boxes (or a hard cap on candidates before NMS) would bound it.
Pure scene sponge: black-box `hil_latency_blocks` also has 0 detections under every setting.

## How big can a physical thermal sponge be?
On this pipeline, a person-and-warm-object scene in a cool room costs about +4 to +6 ms/frame (+7-8 %),
or about +1 % energy at 8.7 fps. That is roughly 3 times the single-person effect (+1.2 ms) and
equal to a handful of people. The theoretical ceiling for arbitrary images is 10-15x, but it needs
a high-frequency adversarial texture that no real object produces at 160x120. It does not change
power in mW, it changes time.

How to build the best plausible scene for real: a ~21 C room with 4-5 people-sized warm bodies
(35-44 C surface), two or three at different depths so they are separate objects, plus 2-3 small hot
spots (42 C), standing about 1.5 m from the camera, mild mist/blur. The generator parameters are in
`banks/hil_latency_layout/manifest.json`. Verify by recording with the real camera: the frame should
show 4 'person' detections. Warm laptops/mugs trigger detections of other classes too (not
tested on hardware).

## Unverified / caveats
- Rendered scenes, not real camera data; the Lepton's blur/noise is modelled. Detection counts on real
  captures of such a scene are not verified.
- Search was short (5 min each); the latency run was still improving at generation 14 and the
  resume did not beat 9.1 ms. Searches (b),(c) found nothing, which does not prove nothing exists.
- Re-measurement uses one 96-frame bank per scene, with the first 16 frames timed per round.
- Cloud regressions come from noisy search-time evaluations (see noise floor) and from adaptively
  chosen points; read the sign and rough size, not p-values.
- Energy per frame in paced mode is per 115 ms period, including idle.

## Banks (hand-off format, on the Jetson under `~/synthlab/sponge-search/banks/`)
`hil_latency_layout` (best plausible), `hil_decouple_layout` (HT up, no cost), `hil_stealth_layout`,
`hil_energy_layout`, `hil_latency_blocks`, `wb_panel`, `wb_pixels` (upper bounds), `proxy_det_layout`,
`proxy_cand_panel`. Each has `frames.npy` (96,120,160) uint16 TLinear + `manifest.json`.

## Figures (`results/sponge-search/`)
`search_progress.png`, `best_scenes.png`, `stage_breakdown.png`, `cost_vs_entropy_cloud.png`.

## Reproduce
`synthlab/sponge_search/jobs/chunk_{a..e}.sh` (run in the Jetson lock), then
`python3 -m synthlab.sponge_search.analysis` and `.figures` locally. Tests: `pytest tests`.
