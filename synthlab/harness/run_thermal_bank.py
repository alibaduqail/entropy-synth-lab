#!/usr/bin/env python3
"""Loop a synthetic (or recorded) thermal frame bank through the real thermal pipeline on
the Jetson and record per-frame stage timings with tegrastats power alongside.

  bank dir : frames.npy  uint16 (N,120,160) Lepton TLinear counts (0.01 K/count)
             manifest.json (optional; copied into the output)
  pipeline : TLinear -> Celsius, fixed 10..50 C -> 8-bit, block image entropy (HT),
             3-channel frame, YOLOv8n (PyTorch CUDA FP32 as in the original project, or a
             TensorRT FP16 engine), NMS, result extraction.
  outputs  : <out>/frames.csv, tegrastats.csv, summary.json, bank_manifest.json
             (with --repeats N > 1: <out>/trial_00 ... trial_NN, plus <out>/repeats.json)

Timed runs must hold the shared lock:
  tools/jetson.sh measured 'python3 ~/synthlab/harness/run_thermal_bank.py --bank B --out O'
Excluded from the measurement: camera acquisition, USB transfer, disk reads (bank is in RAM).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry as tm  # noqa: E402
import thermal_pipeline as tp  # noqa: E402

HARNESS = Path(__file__).resolve().parent
DEFAULT_PT = Path("/home/jetsonnano/yolov8n.pt")
DEFAULT_ENGINE = HARNESS / "models" / "yolov8n_fp16_480x640.engine"
STAGES = ["convert_ms", "bie_ms", "prep_ms", "yolo_pre_ms", "yolo_inf_ms", "yolo_post_ms",
          "yolo_wall_ms", "yolo_extract_ms", "total_ms"]


class Detector:
    """YOLOv8n through Ultralytics predict(), the same call the original pipeline makes.
    Ultralytics' own profilers synchronise CUDA, so inference and postprocess (NMS) times
    are separated cleanly. Pre-NMS candidates are counted outside the timed section."""

    def __init__(self, backend, model_path, conf=0.25, device="0", count_candidates=True):
        self.backend, self.conf, self.device = backend, conf, device
        self.count_candidates = count_candidates
        self.model = self.torch = None
        self.raw = None
        if backend == "none":
            return
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.model = YOLO(str(model_path), task="detect")
        # .pt: rectangular letterbox of a 160x120 frame at imgsz 640 gives 480x640.
        # engine: fixed 480x640 input, so both backends see the identical tensor shape.
        self.imgsz = 640 if backend == "pytorch" else (480, 640)
        self(np.zeros((120, 160, 3), np.uint8))  # builds the predictor
        predictor = self.model.predictor
        original = predictor.postprocess

        def postprocess(preds, *args, **kwargs):
            self.raw = preds
            return original(preds, *args, **kwargs)

        predictor.postprocess = postprocess
        self.info = {"backend": backend, "model": str(model_path), "imgsz": self.imgsz,
                     "conf": conf, "device": str(predictor.device),
                     "fp16": bool(getattr(predictor.model, "fp16", False)),
                     "is_pytorch": bool(getattr(predictor.model, "pt", False)),
                     "is_engine": bool(getattr(predictor.model, "engine", False))}

    def sync(self):
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def __call__(self, frame):
        if self.model is None:
            return {"yolo_pre_ms": 0.0, "yolo_inf_ms": 0.0, "yolo_post_ms": 0.0,
                    "yolo_wall_ms": 0.0, "yolo_extract_ms": 0.0, "n_det": 0, "n_person": 0,
                    "max_conf": float("nan"), "n_candidates": -1}
        self.raw = None
        self.sync()
        start = time.perf_counter()
        result = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        self.sync()
        predicted = time.perf_counter()
        boxes = result.boxes
        if boxes is not None and len(boxes):
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            n_det, n_person, max_conf = len(classes), int((classes == 0).sum()), float(confs.max())
        else:
            n_det, n_person, max_conf = 0, 0, float("nan")
        extracted = time.perf_counter()
        out = {"yolo_pre_ms": float(result.speed.get("preprocess", 0.0)),
               "yolo_inf_ms": float(result.speed.get("inference", 0.0)),
               "yolo_post_ms": float(result.speed.get("postprocess", 0.0)),
               "yolo_wall_ms": (predicted - start) * 1e3,
               "yolo_extract_ms": (extracted - predicted) * 1e3,
               "n_det": n_det, "n_person": n_person, "max_conf": max_conf,
               "n_candidates": -1}
        raw = self.raw[0] if isinstance(self.raw, (list, tuple)) else self.raw
        if self.count_candidates and raw is not None and getattr(raw, "ndim", 0) == 3:
            # (1, 4+nc, anchors): anchors whose best class score passes the threshold
            # are what NMS has to sort and compare. Not included in any timing column.
            out["n_candidates"] = int((raw[0, 4:].amax(0) > self.conf).sum().item())
        return out


def process_frame(raw, detector):
    t0 = time.perf_counter()
    temp_c = tp.raw_to_celsius(raw)
    gray = tp.radiometric_bie_image(raw)
    t1 = time.perf_counter()
    bie = tp.block_image_entropy(gray)
    t2 = time.perf_counter()
    frame = tp.yolo_frame_from_thermal(gray)
    t3 = time.perf_counter()
    det = detector(frame)
    t4 = time.perf_counter()
    return {"convert_ms": (t1 - t0) * 1e3, "bie_ms": (t2 - t1) * 1e3,
            "prep_ms": (t3 - t2) * 1e3, **det, "total_ms": (t4 - t0) * 1e3,
            "HT": bie["value"], "mean_c": float(temp_c.mean()), "max_c": float(temp_c.max()),
            "gray_sd": bie["global_sd"]}


def run_trial(bank, detector, out, args, bank_info):
    """One warmup + measured window. Returns the summary dict (also written to disk)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    use_tegra = (not args.no_tegrastats) and tm.tegrastats_available()
    interval_s = args.tegra_interval_ms / 1000.0
    logger = tm.TegraLogger(out / "tegrastats.csv", args.tegra_interval_ms) if use_tegra else None
    period = 1.0 / args.fps
    n = len(bank)
    rows = []
    if logger:
        logger.start()
    try:
        idle_start = time.time()
        time.sleep(args.idle_seconds)
        idle_end = time.time()
        warm_start, index = time.monotonic(), 0
        while time.monotonic() - warm_start < args.warmup:
            if not args.unpaced:
                tm.wait_until(warm_start + index * period)
            process_frame(bank[index % n], detector)
            index += 1
        warm_frames = index
        base_mono, base_unix = time.monotonic(), time.time()
        target = None if args.unpaced else int(round(args.seconds * args.fps))
        index = 0
        while True:
            if args.unpaced:
                if time.monotonic() - base_mono >= args.seconds:
                    break
                late = 0.0
            else:
                if index >= target:
                    break
                late = tm.wait_until(base_mono + index * period)
            offset = time.monotonic() - base_mono
            row = {"frame": index, "bank_frame": index % n, "unix_time": time.time(),
                   "offset_s": offset, "lateness_ms": late * 1e3}
            row.update(process_frame(bank[index % n], detector))
            rows.append(row)
            index += 1
        if not args.unpaced:
            tm.wait_until(base_mono + target * period)
        end_mono, end_unix = time.monotonic(), time.time()
        time.sleep(max(0.5, 2 * interval_s))
    finally:
        if logger:
            logger.stop()
    with open(out / "frames.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    duration = end_mono - base_mono
    col = lambda key: np.asarray([r[key] for r in rows], dtype=float)
    summary = {"status": "complete", **bank_info,
               "backend": args.backend, "mode": "unpaced" if args.unpaced else "paced",
               "target_fps": None if args.unpaced else args.fps,
               "seconds": args.seconds, "warmup_s": args.warmup, "warmup_frames": warm_frames,
               "frames": len(rows), "duration_s": duration, "achieved_fps": len(rows) / duration,
               "bank_loops": len(rows) / n,
               "late_frames": int((col("lateness_ms") > period * 1e3).sum()),
               "HT": tm.dist(col("HT")), "n_det": tm.dist(col("n_det")),
               "n_person": tm.dist(col("n_person")), "n_candidates": tm.dist(col("n_candidates")),
               "stages_ms": {key: tm.dist(col(key)) for key in STAGES},
               "measurement_start_unix": base_unix, "measurement_end_unix": end_unix,
               "detector": getattr(detector, "info", {"backend": "none"}),
               "excluded": "camera acquisition, USB transfer, disk I/O; bank looped from RAM",
               "power_note": "tegrastats VDD_IN = module input power, not wall power"}
    problems = []
    if abs((end_unix - base_unix) - duration) > 0.25:
        problems.append("wall clock jumped during the measured window")
    if logger:
        tele = tm.summarize(out / "tegrastats.csv", base_unix, end_unix, interval_s, len(rows))
        idle = tm.idle_power(out / "tegrastats.csv", idle_start, idle_end)
        summary["telemetry"] = tele
        problems += tele["problems"]
        if "energy_j" in tele:
            mean = lambda key: (tele["metrics"].get(key) or {}).get("mean")
            summary.update(power_w=tele["power_w_time_weighted"], energy_j=tele["energy_j"],
                           energy_per_frame_j=tele["energy_per_unit_j"], idle_power_w=idle,
                           gpu_pct=mean("gpu_pct"), cpu_pct=mean("cpu_pct"),
                           ram_mb=mean("ram_mb"), temp_gpu_c=mean("temp_gpu_c"),
                           temp_cpu_c=mean("temp_cpu_c"),
                           power_cpu_gpu_cv_w=mean("vdd_cpu_gpu_cv_w"))
            if idle is not None:
                summary["energy_above_idle_per_frame_j"] = (
                    (tele["power_w_time_weighted"] - idle) * duration / len(rows))
    else:
        problems.append("tegrastats not available: no power data")
    summary["problems"] = problems
    summary["ok"] = not problems
    tm.write_json(out / "summary.json", summary)
    return summary


def one_line(summary):
    stage = summary["stages_ms"]
    power = summary.get("power_w")
    return (f"{summary['bank_name']} [{summary['backend']}/{summary['mode']}] "
            f"frames={summary['frames']} fps={summary['achieved_fps']:.2f} "
            f"HT={summary['HT']['mean']:.4f} det={summary['n_det']['mean']:.2f} "
            f"cand={summary['n_candidates']['mean']:.1f} "
            f"inf={stage['yolo_inf_ms']['mean']:.2f}ms post={stage['yolo_post_ms']['mean']:.2f}ms "
            f"bie={stage['bie_ms']['mean']:.2f}ms total={stage['total_ms']['mean']:.2f}ms "
            f"(p95 {stage['total_ms']['p95']:.2f}) "
            + (f"P={power:.3f}W E/frame={summary['energy_per_frame_j']*1e3:.1f}mJ "
               if power is not None else "P=n/a ")
            + ("OK" if summary["ok"] else "PROBLEMS: " + "; ".join(summary["problems"])))


def load_bank(folder):
    folder = Path(folder).expanduser()
    path = folder / "frames.npy"
    bank = tp.validate_bank(np.load(path, allow_pickle=False))
    manifest = {}
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text())
    info = {"bank_name": folder.name, "bank_dir": str(folder), "bank_frames": int(len(bank)),
            "bank_sha256": tm.sha256(path),
            "bank_track": folder.parent.parent.name if folder.parent.name == "banks" else None,
            "bank_generator": manifest.get("generator") if isinstance(manifest, dict) else None}
    return np.ascontiguousarray(bank), manifest, info


def make_detector(args):
    if args.backend == "none":
        return Detector("none", None)
    path = args.model or (DEFAULT_PT if args.backend == "pytorch" else DEFAULT_ENGINE)
    if not Path(path).is_file():
        raise SystemExit(f"model file not found: {path}")
    return Detector(args.backend, path, args.conf, args.device, not args.no_candidates)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, type=Path, help="dir with frames.npy (+manifest.json)")
    ap.add_argument("--out", required=True, type=Path, help="new output dir")
    ap.add_argument("--seconds", type=float, default=60, help="measured window per trial")
    ap.add_argument("--warmup", type=float, default=15, help="unrecorded warmup before each trial")
    ap.add_argument("--fps", type=float, default=8.7, help="pacing rate (Lepton 3.5 = 8.7)")
    ap.add_argument("--unpaced", action="store_true", help="max throughput, no sleeping")
    ap.add_argument("--backend", choices=["pytorch", "tensorrt", "none"], default="pytorch",
                    help="pytorch = CUDA FP32 (original pipeline); tensorrt = FP16 engine; "
                         "none = skip YOLO (convert + BIE only)")
    ap.add_argument("--model", type=Path, default=None, help="override weights/engine path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the trial N times (noise floor)")
    ap.add_argument("--idle-seconds", type=float, default=3, help="idle power window before warmup")
    ap.add_argument("--tegra-interval-ms", type=int, default=200)
    ap.add_argument("--no-tegrastats", action="store_true")
    ap.add_argument("--no-candidates", action="store_true", help="skip the pre-NMS candidate count")
    ap.add_argument("--max-frames", type=int, default=0, help="use only the first K bank frames")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seconds <= 0 or args.warmup < 0 or not 0 < args.fps <= 60 or args.repeats < 1:
        raise SystemExit("need seconds > 0, warmup >= 0, 0 < fps <= 60, repeats >= 1")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output dir {args.out} is not empty; choose a new one")
    bank, manifest, info = load_bank(args.bank)
    if args.max_frames:
        bank = bank[:args.max_frames]
    args.out.mkdir(parents=True, exist_ok=True)
    tm.write_json(args.out / "bank_manifest.json", manifest)
    tm.write_json(args.out / "run_config.json",
                  {"args": {k: str(v) for k, v in vars(args).items()}, **info,
                   "environment": tm.environment(),
                   "code_sha256": {p.name: tm.sha256(p) for p in sorted(HARNESS.glob("*.py"))}})
    detector = make_detector(args)
    for raw in bank[:min(len(bank), 8)]:
        process_frame(raw, detector)
    summaries = []
    for repeat in range(args.repeats):
        folder = args.out if args.repeats == 1 else args.out / f"trial_{repeat:02d}"
        summary = run_trial(bank, detector, folder, args, info)
        summary["repeat"] = repeat
        summaries.append(summary)
        print(one_line(summary), flush=True)
    if args.repeats > 1:
        keys = ["power_w", "energy_per_frame_j", "achieved_fps"]
        table = {k: tm.dist([s[k] for s in summaries if s.get(k) is not None]) for k in keys}
        for stage in STAGES:
            table[stage + "_mean"] = tm.dist([s["stages_ms"][stage]["mean"] for s in summaries])
        tm.write_json(args.out / "repeats.json",
                      {"n_trials": len(summaries), "trial_to_trial": table, **info})
    return 0 if all(s["ok"] for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
