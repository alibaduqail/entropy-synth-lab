# Shared brief for every agent working in this repo

Overnight run of 2026-11-23 -> 24. The owner (Ali) is asleep: never ask for input, never
wait on a human. Make a reasonable choice, write down the assumption, keep going.

## What the parent research project is
"Characterisation of Entropy-Driven Computational Overhead in Multi-Modal Edge Perception
Systems" (team: Nanditha lead/power, Sam LiDAR, Ali thermal). Question: does environmental
disorder (fog, clutter, heat) raise a Jetson's power and per-frame latency even when the real
scene is unchanged - a physical-world analogue of Shumailov et al. 2021 "Sponge Examples".
- Hyp A: scene complexity raises LiDAR entropy HL, thermal entropy HT, and power.
- Hyp B (the security claim): fog raises HL while HT stays flat, and power still rises.
- Hyp C: power-vs-entropy has a knee.
- Null: fixed compute graph => power/latency constant; a flat result is publishable if honest.
- HL = eigen-entropy of local point neighbourhoods (dos Santos 2019); HT = block image
  entropy BIE (Ayunts 2024). Conditions A0-A3 (empty -> people + warm objects), B0-B3 (fog).
- Hardware: Jetson Orin Nano Super 8GB (MAXN_SUPER), FLIR Lepton 3.5 on PureThermal 3
  (160x120, 8.7 fps, 57 deg HFOV, radiometric TLinear 0.01 K/count), Livox Mid-360
  (360 x [-7,+52] deg, ~200k pts/s, 10 Hz, non-repetitive), TI IWR6843ISK 60 GHz radar,
  AVHzY CT-3 power meter (not attached tonight; tegrastats VDD_IN is available).
- Known facts so far: Ali's thermal pipeline is YOLOv8n PyTorch CUDA FP32 (not TensorRT);
  a 279-trial synthetic entropy sweep found r(HT, yolo_ms)=0.70 and r(detections, yolo_ms)=0.87;
  Sam found YOLO does the same work every frame and HL costs ~350 ms per 1 s window;
  fog trials (Experiment B) have never been run for real.

## Tonight's goal
Find the best ways to GENERATE SYNTHETIC SENSOR DATA (thermal first; LiDAR and radar too)
and run it through real processing on the real Jetson, so the team can test the hypotheses
without a person standing in a room. Try several approaches, compare them honestly, and tie
every result back to hypotheses A/B/C and the null. Also flag anything wrong with the
experiment design itself.

## Where things are
- This repo: `entropy-synth-lab` (GitHub, private). You work in YOUR OWN git worktree and
  branch (given in your task). Commit often, small commits, push your branch
  (`git push -u origin <branch>`). Do NOT touch other tracks' directories or `main`.
- The original project (READ-ONLY, never edit, never commit its data here):
  `/Users/alibaduqail/Documents/thermal camera research work/` - `scripts/` (experimentA.py
  has the real BIE + YOLO pipeline; synthetic_experiment.py is the earlier Jetson harness),
  `data/` (real Lepton recordings, 7 GB), `docs/`, `deliverables/thermal_synth_mvp/`.
- Room photos: `reference_local/room_*.jpg` in the main checkout
  (`/Users/alibaduqail/Documents/thermal camera research work/entropy-synth-lab/reference_local/`).
- Room geometry: `scene/room_v0.json` (provisional); `scene/room_v1.json` may appear later
  from the room-3d track - always load the scene from a path argument, default to the newest.
- Google Drive folder "ThreatDet" (Drive MCP tools, if you have them): Guide.docx id
  1NJ_P5SZV2KFqWdKSwjzUvVtkWRCvgIQ8; "LiDAR Track Summary" id
  1fmh7hpix7DB21n6afcnFl-VPxFvEy4u7TxKPH39PjN8; Sam's compute_hl.py id
  1OPwrSpwYq7h8qh-jWq-t8oH7xJGlKeQz. Read-only. Never copy teammates' files into git.

## Jetson rules (one shared device - this matters)
- Reach it only through `tools/jetson.sh` (USB-C link, 192.168.55.1). It has internet.
- Your remote folder: `/home/jetsonnano/synthlab/<your-track>/`. Nothing else is yours.
- READ-ONLY on the Jetson: `~/thermal-camera-radiometric`, `~/thermal_replay_batches`,
  `~/experimentA_archive`, `~/thermal-synth-mvp`, `~/yolov8n.pt` (copy, don't modify).
- ANY timed or power-measured run MUST go through `tools/jetson.sh measured '<cmd>'`, and any
  long GPU/CPU job through `tools/jetson.sh heavy '<cmd>'` (same exclusive flock). This keeps
  measurements from overlapping. Keep each locked command under ~20 minutes so others get turns;
  split long sweeps into chunks. Light commands (ls, cat, rsync, pip) use `run`.
- Do not change the power mode (MAXN_SUPER), do not reboot, do not touch nvpmodel/jetson_clocks,
  no sudo. No live camera recording. `pip install --user` is fine; do not upgrade torch/numpy/
  ultralytics/opencv (torch 2.8.0 CUDA, numpy 1.26.4, cv2 4.11, ultralytics 8.3.226, scipy 1.8).
- Power = tegrastats VDD_IN (module input, mW). Log it at a fixed interval alongside timings.

## Engineering rules
- Python 3.10-compatible; numpy/scipy/opencv/torch only unless a dependency is clearly worth it.
- Files under 500 lines. Put code in `synthlab/<track>/`, tests in `tests/`, findings in
  `docs/<track>.md`, small result files + PNG figures in `results/<track>/`. No big binaries in git
  (>5 MB): keep them on disk under `data_local/` (gitignored) or on the Jetson.
- Run your tests before each commit. Fixed seeds; record seeds and parameters in outputs.
- Commit messages: plain, no `Co-Authored-By` trailer.
- Honesty rules: every number in your doc must come from a file in `results/`. Say what is
  measured vs assumed vs simulated. Report null/negative results plainly. If something failed,
  say so and say why.

## What to hand back
Your final message is read by the lead who builds a 9 AM slideshow. Make it a tight report:
what you built, what you measured (numbers with units and n), what it means for hypotheses
A/B/C/null, what is unverified, the 2-3 best figures (paths), and the branch name + last commit.
Also keep `docs/<track>.md` current as you go, so partial progress survives.

## Data hand-off formats (so tracks interoperate)
- Thermal frame bank: `~/synthlab/<track>/banks/<bank_name>/frames.npy` on the Jetson =
  uint16 array (N,120,160) in Lepton TLinear units (0.01 K per count, e.g. 29515 = 22.0 C),
  plus `manifest.json` (generator, parameters, seed, n_frames, optional per-frame labels:
  n_people, boxes, fog_level, lamp_on ...). N is typically 96-256.
- LiDAR bank: `~/synthlab/<track>/banks/<bank_name>/frame_00000.npz ...` each one 100 ms
  Mid-360 frame with `xyz` float32 (M,3) in sensor frame, `intensity` uint8, `tag` uint8
  (Livox tag byte; bits 2-3 = particle/fog flag), plus `manifest.json`.
- Shared Jetson harness (owned by the jetson-harness track) is installed at
  `~/synthlab/harness/` on the Jetson; `python3 ~/synthlab/harness/run_thermal_bank.py --help`
  and `run_lidar_bank.py --help`. If it is not there yet, keep building and check again later.
