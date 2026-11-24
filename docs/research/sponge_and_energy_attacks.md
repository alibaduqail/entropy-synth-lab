# Sponge and energy-latency attacks: where can input-dependent cost come from?

Research track, night of 2026-11-23. Every source below was fetched tonight unless marked
"(not re-fetched)" or "(unverified)". The project's earlier note
`docs/reports/Thermal edge latency literature.md` (original project, read-only) already covers
Sponge Examples, Phantom Sponges, Overload, SlowLiDAR, SlowTrack and Underload with numbers; this
file does not repeat those tables. It adds what that note lacks: three prior-art threats, the
exact Ultralytics mechanics, the physical basis for data-dependent power, and the 2025-2026
literature.

## Short answers to the key questions

**Where can input-dependent cost come from in a fixed-graph detector such as YOLOv8?**

| Source | Mechanism | Evidence | Expected size on this project's stack |
|---|---|---|---|
| NMS candidate count | Candidates above `conf` enter NMS; pairwise IoU is worst-case O(N^2) | Overload (CVPR 2024), Phantom Sponges (WACV 2023), survey [S1] | Small. YOLOv8 at 640 px has 8,400 slots (YOLOv5: 25,200); GPU NMS cannot be pushed past a 100-500 ms budget [E1] |
| Per-detection post-processing | Box scaling, `Results` objects, `.cpu()` copies, drawing, logging | Liu et al. [L1]: post-processing time vs detected-object count r = 0.43 (YOLOv3) to 0.98 (Faster R-CNN) on a Jetson AGX Xavier | This is what the team measured. The audit track reports a step (0 vs >=1 detection, about 2 ms), not a count slope |
| Tracker association | Cost grows with number of tracks/detections | SlowTrack (AAAI 2024); TrackFlood [T1]: detector 0.99-1.01x, ByteTrack end-to-end 1.09-1.47x on Jetson AGX Orin | Zero: the thermal pipeline has no tracker |
| Activation sparsity | Zero-skipping hardware does less work on sparse activations | Shumailov et al. 2021; Mueller and Quiring [M1] | Near zero: YOLOv8 uses SiLU (no exact zeros) and the Orin GPU does not skip zeros |
| Data-dependent switching power | Bit flips in datapaths depend on operand values even when the instruction stream is identical | Gregersen et al. [G1]: almost 40 % GPU power swing for same-shape GEMM; Bhalachandra et al. [B1]: up to 67 % on an A100 | Unknown at Jetson scale. These are synthetic extremes (zeros vs random). No Jetson-class or mW-scale measurement was found |
| Dynamic shapes / pre-processing | Letterbox and resize cost depends on input size, not content | Liu et al. [L1] (image size effect) | Zero: Lepton frames are always 160x120 |
| DVFS / governor | Load-dependent frequency scaling turns small utilisation changes into power changes and adds state jumps | NVIDIA docs (earlier project note); audit track found a mid-run state jump of 30.8 -> 37.0 ms | A confound, not a mechanism. Must be pinned or logged |
| HL / HT metric code itself | The entropy metric's own CPU cost depends on point count / neighbourhood density | Sam's ~350 ms per window; audit track: BIE 34.5 ms/frame | Largest input-dependent term in the LiDAR chain, and it is the measurement instrument, not the perception system |

**What effect sizes are reported on Jetson-class hardware?** Adversarially optimised: Overload
reaches about 11x on Jetson Xavier NX for YOLOv5 (earlier note), but for YOLOv8s on Jetson Orin NX
the same attack moves latency only 30.1 -> 33.2 ms (another table: 18.2 -> 21.7 ms with TensorRT)
[U1]. Natural inputs: Liu et al. report correlations, not multiples, and rendered rain *reduced*
time [L1]. No paper reports Jetson *power* under either attack or natural variation.

**Has a physical, environmental, non-adversarial version been shown?** Partly, and the team's
claim must be narrowed (see "Novelty" below). Physical latency attacks exist but all use optimised
projected patterns: SlowPerception (projector, 2.5 s average added latency) and DetStorm (IEEE S&P
2025, +506 % detected objects, delays up to 8.1 s) [S1]. Natural scene dependence of detector
latency on a Jetson, including a weather sweep, was published in 2022 [L1]. What nobody has shown:
a real (not rendered) environmental obscurant, measured on LWIR and LiDAR, with power logged, and
framed as an availability question.

**Is there a physical basis for input-dependent power with a fixed graph?** Yes. CMOS dynamic
power is proportional to switching activity, and two HPC papers measured large swings for
fixed-size matrix multiplication by changing only the values [G1][B1]. So the null hypothesis
"fixed compute graph => constant power" is not guaranteed by physics. It is an empirical claim
that needs a matched-detection-count test.

## Sources read tonight

- **[L1] Liu, Wang, Shi, "Understanding Time Variations of DNN Inference in Autonomous Driving",
  arXiv 2209.05487 (2022).** https://arxiv.org/pdf/2209.05487 . Jetson AGX Xavier, MAXN, FP32.
  Correlation of post-processing time with number of detected objects: Faster R-CNN 0.98,
  Mask R-CNN 0.98, YOLOv3 0.43, SSD-MobileNetv2 0.91, SSD-Inceptionv2 0.96. Tests all-zero,
  all-255 and random-pixel inputs. Renders rain at 0/25/50/100/150/200 mm/h into KITTI: mean and
  variance of end-to-end time *decrease* with rain because proposals and detections drop.
  "Insight 1": the scenario affects inference-time variation through the number of detected
  objects. Peer-review venue not confirmed tonight (unverified).
- **[S1] Gu, Gao, Saremi, Chakraborty, "Deep Learning Latency Attacks and Defenses: A
  Cross-Domain Survey of Availability Threats", arXiv 2609.36732 (29 Sep 2026, formatted for ACM
  TECS, 59 pages).** https://arxiv.org/abs/2609.36732 . Full text read. Points used here:
  (a) it defines the attack mechanism as an adversarial input matched in visible size to a benign
  one, and explicitly sets aside "naturally hard instances"; (b) physical delivery is limited to
  projector attacks and "require[s] active infrastructure"; (c) it calls NMS runtime
  data-dependent and a timing side channel, and asks for constant-time NMS; (d) it lists physical
  realisability as an open challenge.
- **[E1] Monteuuis, Chen, Petit, "Latency NMS Attacks: Is It Real Life or Is It Just Fantasy?",
  NeurIPS 2025** (read via [S1], not fetched directly). Re-evaluates Daedalus, Overload, Phantom
  Sponges and Beyond Phantom Sponges on 7 hardware platforms and 15 YOLO models. Proposal counts
  needed to break a 100-500 ms budget are unreachable when NMS runs on GPU. Model caps: about
  25,000 proposals for YOLOv5 and 8,000 for YOLOv8, both under Ultralytics `max_nms` = 30,000.
  With CPU NMS, YOLOv5 can reach 2 s; YOLOv8 stays under 500 ms. Attacks do not transfer across
  YOLO versions, model sizes or datasets.
- **[U1] "Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against
  Latency Attacks for Edge Devices" (Underload), CVPR 2025.** https://arxiv.org/html/2412.02171 .
  YOLOv8s under Overload on Jetson Orin NX: 30.1 -> 33.2 ms (and 18.2 -> 21.7 ms TensorRT).
- **[T1] Gu et al., "TrackFlood: Relocating Latency Attacks from NMS-Free Detectors to Real-Time
  Trackers", arXiv 2609.33948 (27 Sep 2026).** https://arxiv.org/abs/2609.33948 . Jetson AGX Orin,
  TensorRT FP16, YOLOv10 / YOLO26 / RT-DETR. Detector latency stays 0.99-1.01x clean; a 32/255
  stress test reaches 13.54x with OC-SORT (numbers from [S1]).
- **[M1] Mueller and Quiring, "The Impact of Uniform Inputs on Activation Sparsity and
  Energy-Latency Attacks in Computer Vision", IEEE Security and Privacy Workshops 2024.**
  https://arxiv.org/abs/2403.18587 . Flat, uniformly coloured images trigger more activations
  (convolution + batch norm + ReLU) and match optimised sponge examples at a fraction of the cost.
  [S1] summarises this as: uniform inputs are "free" sponges, bounded by the architecture's
  sparsity range. Abstract-level reading only; no energy numbers extracted.
- **[G1] Gregersen, Patel, Choukse, "Input-Dependent Power Usage in GPUs", arXiv 2409.18324
  (SC24 workshop).** https://arxiv.org/abs/2409.18324 . Same matrix shapes, different values:
  GPU power during GEMM changes "by almost 40%". Tested value distribution, bit similarity,
  placement and sparsity. Hypothesis: bit flips. Abstract-level reading; GPU model not extracted.
- **[B1] Bhalachandra, Austin, Williams, Wright, "Understanding the Impact of Input Entropy on
  FPU, CPU, and GPU Power", arXiv 2212.08805 (2022).** https://arxiv.org/abs/2212.08805 .
  "Power variations of up to 67%" on an NVIDIA A100 for the same DGEMM benchmark and size,
  changing only matrix values. Preprint; peer-review status not confirmed.
- **Ultralytics NMS source.**
  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py (main branch,
  fetched tonight; the Jetson runs 8.3.226, which may differ in detail). Defaults:
  `conf_thres=0.25`, `max_det=300`, `max_nms=30000`, `max_time_img=0.05`, `max_wh=7680`.
  Order: `xc = prediction[:, 4:mi].amax(1) > conf_thres` -> per image keep candidates -> sort
  and truncate only if n > `max_nms` -> class-offset boxes -> `torchvision.ops.nms` (only if
  torchvision is already imported, else pure-torch `TorchNMS`) -> `i[:max_det]`. Time limit
  `2.0 + max_time_img * bs` seconds. If the output's last dimension is 6 or `end2end` is set,
  NMS is skipped. `max_det` caps what leaves NMS, not the work NMS does [S1].
  Zero-candidate frames: the Jetson's installed Ultralytics 8.3.226 has the early-exit test
  `if not x.shape[0]:` at `utils/nms.py` line 108, before box conversion and NMS, and the same
  `xc`, `time_limit` and `max_nms` lines as above (grep saved in
  `results/research/jetson_ultralytics_nms_grep.txt`). That fits the audit track's 0-vs->=1
  detection step.
- Other items named in [S1] and not fetched directly: SlowPerception (Ma et al., arXiv
  2406.05800); DetStorm (Muller et al., "Investigating physical latency attacks against
  camera-based perception", IEEE S&P 2025); Groundswell (Xia, Gao, Shi, VehicleSec 2026: sticker
  or projection proposed, evaluation digital only); SlowLiDAR (Liu et al., CVPR 2023);
  CP-FREEZER (cooperative perception, >90x, arXiv 2508.01062); Krithivasan et al., "Sparsity
  Turns Adversarial" (IEEE TCAD 2020); te Lintelo et al., SkipSponge (2025: at most about 3 %
  energy increase on image classifiers, could not reproduce the original results).
- Brachemi Meftah et al., "Energy-Latency Attacks: A New Adversarial Threat to Deep Learning",
  arXiv 2503.04963 (2025): a second survey; search-snippet level only (unverified).

## Novelty: is "no one has shown that yet" still true?

Not as the team currently words it. Three statements, in decreasing order of safety:

1. Safe: "No published work measures board power on a Jetson while a real environmental
   obscurant (fog or mist) changes the input to a thermal and LiDAR perception pipeline." Nothing
   found contradicts this.
2. Needs care: "A physical-world analogue of sponge examples." The 2026 survey's definition
   excludes naturally hard inputs [S1]. The framing survives only if an attacker can cheaply
   cause the condition. A fog machine or humidifier is attacker-controllable, so say that, and
   call it an environmental availability stressor rather than a sponge example.
3. False: "Nobody has measured natural scene or weather dependence of detector latency on a
   Jetson." Liu et al. did in 2022, and found that weather lowered cost [L1].

Also prior art on the words themselves: "input entropy" and "power" for a fixed workload [B1].

## What to do with this tonight

- Sponge-search track: treat the number of above-threshold candidates and the 0-vs->=1 detection
  step as the only proven levers. Report NMS time and post-NMS Python time separately.
- Add three control frame types to every thermal bank run: all-constant, iid noise, and a real
  frame, all with zero detections. If VDD_IN differs between them at equal detection count, that
  is evidence for data-dependent switching power [G1][B1]; if not, that is a clean null.
- Record direction, not just magnitude. Liu et al. and the audit track both see obscuration
  lowering cost because detections vanish.
- Reword the novelty claim on the slides to statement 1 above.

## What to do with this next month

- Read [E1] and DetStorm in full; both were only read through the survey.
- Run the matched-detection-count power test with the external AVHzY meter, because the expected
  effect (tens of mW) is at or below one INA3221 step (see `sim_to_real_measurement.md`).
- Export YOLOv8n to TensorRT with NMS in-graph and as end-to-end (NMS-free) to show which part of
  the effect is removable.
- If a tracker is ever added to the pipeline, re-run: that is where the current literature says
  the input-dependent cost has moved [T1].
- Cite [L1], [B1], [G1], [S1], [E1] in the paper's related work; reviewers will know them.
