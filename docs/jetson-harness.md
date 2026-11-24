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

RESULTS_PLACEHOLDER
