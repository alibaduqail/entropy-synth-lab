# jetson-harness: shared measurement harness and cross-approach comparison

Branch `jetson-harness`. Code in `synthlab/harness/`, installed on the Jetson at
`~/synthlab/harness/`. Results in `results/jetson-harness/`. Every number below is taken from
a file in that folder (named next to the number).

## 1. What the harness does

| Script | Input | What runs on the Jetson | Output |
|---|---|---|---|
| `run_thermal_bank.py` | `frames.npy` uint16 (N,120,160) TLinear counts + `manifest.json` | TLinear→°C, fixed 10–50 °C → 8-bit, block image entropy (HT), YOLOv8n, NMS | `frames.csv`, `tegrastats.csv`, `summary.json` |
| `run_lidar_bank.py` | `frame_*.npz` 100 ms frames (`xyz`, `tag`) + `manifest.json` | 1 s windows → ROI → 5 cm voxels → radius kNN → eigenvalues → H_DS, H_eigen; optional Euclidean clustering | `windows.csv`, `tegrastats.csv`, `summary.json` |
| `run_matrix.py` | many banks | one randomised round over all banks, model loaded once | one trial folder per bank |
| `build_trt_engine.py` | ONNX | TensorRT FP16 engine for `--backend tensorrt` | `models/*.engine` |
| `make_control_banks.py` | recorded frames, earlier generator | control and real-anchor banks | `~/synthlab/jetson-harness/banks/` |

Usage (all timed runs hold the shared lock):

```
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_thermal_bank.py --bank <bank dir> --out <new dir> \
    [--seconds 60 --warmup 15 --fps 8.7 --unpaced --backend pytorch|tensorrt|none --no-bie --repeats N]'
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_lidar_bank.py --bank <bank dir> --out <new dir> \
    [--window 1.0 --workload entropy|cluster|both --seconds 60 --warmup 10 --unpaced --z-min -0.85 --roi ...]'
tools/jetson.sh measured 'python3 ~/synthlab/harness/run_matrix.py --kind thermal|lidar --banks <dirs...> \
    --round R --seed S --out <dir> -- <options passed to the bank runner>'
```

The thermal functions are copied from the original project's `scripts/experimentA.py`
(`raw_to_celsius`, `radiometric_bie_image`, `block_image_entropy`, `yolo_frame_from_thermal`),
and YOLO is called through the same Ultralytics `predict()` call (`imgsz=640`, `conf=0.25`,
device 0). The LiDAR entropy stage re-implements the method in the LiDAR track's
`compute_hl.py` (same parameters: 5 cm leaf, 0.20 m radius, k_max 150, k_min 10, ROI
±6 m × ±6 m × −2…3 m, floor cut −0.85 m); the teammate's file is not in this repo.

### Per-frame columns (thermal)

`convert_ms` (TLinear→°C and 8-bit map), `bie_ms` (entropy stage), `prep_ms` (3-channel copy),
`yolo_pre_ms`, `yolo_inf_ms` (network forward, CUDA-synchronised), `yolo_post_ms` (NMS and
box rescale), `yolo_wall_ms` (whole `predict()` call), `yolo_extract_ms` (boxes to CPU),
`total_ms`, `total_no_bie_ms`, `HT`, `n_det`, `n_person`, `any_det`, `n_candidates`
(anchors above the confidence threshold before NMS; counted outside all timings), `max_conf`.

### What is measured and what is not

- Measured: processing of frames already in RAM, on the real Jetson, with the real code path.
- Not measured: camera or LiDAR acquisition, USB/network transport, rosbag decoding, disk I/O,
  preview windows, saving arrays.
- Power is `tegrastats` `VDD_IN`: module input power sampled every 200 ms. It is not wall
  power and no external meter was attached. Energy per frame is the trapezoid integral of
  `VDD_IN` over the measured window divided by frames processed; in paced mode this includes
  the idle time between frames.
- Each trial: 2–3 s idle (idle power), warmup (not recorded), then the measured window.
  `summary.json` has `ok`/`problems` from telemetry coverage checks (sample count ≥ 80 % of
  expected, no gap > 1 s, window fully covered, wall clock steady).
- Two pacing modes: paced at 8.7 fps (the Lepton rate; the GPU idles most of each period) and
  unpaced (back to back; the device is busy all the time).
- Two YOLO backends: PyTorch CUDA FP32 (what the original thermal pipeline uses) and a
  TensorRT 10.3 FP16 engine at the same 480×640 input (what the project Guide specifies).
- Device state recorded with every run (`run_config.json`): power mode MAXN_SUPER, CPU
  governor `schedutil`, GPU governor `nvhost_podgov`. Neither was changed (no sudo). `jtop`
  runs as a background service on the device and was left alone.

## 2. Null-hypothesis test (step 2)

### 2a. Noise floor (measured, `results/jetson-harness/step2_noise_floor.csv`, `fig_noise_floor.png`)

Same bank (`real_heatlamp_off`, real Lepton recording) repeated 8 times per configuration,
15 s measured window per trial, 4 s warmup, 2 s idle. n = 8 trials x 4 configurations, all 32
passed the telemetry checks. Values are mean / trial-to-trial SD.

| config | total ms/frame | YOLO inference ms | NMS/post ms | BIE ms | VDD_IN W | mJ/frame |
|---|---|---|---|---|---|---|
| PyTorch FP32, unpaced | 63.88 / 0.12 | 30.58 / 0.21 | 1.70 / 0.03 | 26.07 / 0.16 | 7.251 / 0.128 | 464.7 / 8.2 |
| PyTorch FP32, paced 8.7 fps | 73.78 / 0.39 | 30.82 / 0.03 | 2.07 / 0.06 | 34.17 / 0.34 | 5.812 / 0.059 | 668.1 / 6.8 |
| TensorRT FP16, unpaced | 42.45 / 0.12 | 9.64 / 0.01 | 1.71 / 0.03 | 25.85 / 0.06 | 6.502 / 0.063 | 277.1 / 3.3 |
| TensorRT FP16, paced 8.7 fps | 69.69 / 1.00 | 9.78 / 0.005 | 2.75 / 0.02 | 49.01 / 0.95 | 5.149 / 0.010 | 591.8 / 1.2 |

- Trial-to-trial latency CV is 0.2-1.4 %; power CV is 0.2-1.8 %. With 3 repeats per bank the
  minimum detectable difference (80 % power, alpha 0.05) is 0.28 ms and 0.29 W (PyTorch unpaced).
- VDD_IN is quantised at ~39.5 mW per step. The PyTorch-unpaced trial SD (128 mW) is ~3 steps,
  so trial means are not resolution-limited; TensorRT-paced SD (10 mW) is below one step, so
  differences under ~40 mW in that mode cannot be trusted from tegrastats alone.
- BIE (the entropy measurement itself) is 41 % (PyTorch unpaced) to 70 % (TensorRT paced) of
  per-frame time. In paced mode BIE is slower (34-49 ms vs 26 ms) because the CPU governor
  (`schedutil`) drops clocks in the idle gaps between frames.
- Chip temperature drifted < 1.4 C within a configuration; power-vs-temperature slope was
  0.002-0.35 W/C with R^2 <= 0.12 (not a dominant term at this drift).

RESULTS_PLACEHOLDER
