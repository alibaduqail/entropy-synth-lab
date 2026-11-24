"""Jetson measurement rig: the real Experiment-A per-frame pipeline, kept loaded.

Imports the deployed `experimentA.py` on the Jetson READ-ONLY (never copied or edited) and
calls its own functions per frame, in the order its capture loop does:
TLinear -> Celsius -> fixed 10-50 C 8-bit image -> BIE (HT) -> 3-channel frame ->
`run_yolo` (Ultralytics predict, imgsz 640, conf 0.25, defaults iou 0.7 / max_det 300)
-> per-frame CSV row. Power is tegrastats VDD_IN sampled by a separate process.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import threading
import time

import numpy as np

EXPA_PATHS = [
    "~/thermal-camera-radiometric/jetson_experiment_A_thermal_yolo_handoff/experimentA.py",
    "~/experimentA_archive/experimentA.py",
]
RE_VDD = re.compile(r"VDD_IN (\d+)mW")
RE_GPU = re.compile(r"VDD_CPU_GPU_CV (\d+)mW")
RE_TJ = re.compile(r"tj@([\d.]+)C")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
STAGES = ["bie_ms", "prep_ms", "pre_ms", "inf_ms", "post_ms", "overhead_ms", "log_ms"]


def load_expa(path=None):
    for p in ([path] if path else EXPA_PATHS):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("experimentA_deployed", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, p
    raise FileNotFoundError("deployed experimentA.py not found")


class PowerLog:
    """tegrastats in its own process; lines are timestamped on arrival, parsed lazily."""

    def __init__(self, interval_ms=50):
        self.interval_ms = interval_ms
        self.rows = []
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.rows.append((time.time(), line))

    def window(self, t0, t1):
        sel = [ln for (t, ln) in self.rows if t0 <= t <= t1]
        out = {"n_power": len(sel)}
        for key, rx in (("vdd_in_mw", RE_VDD), ("vdd_gpu_mw", RE_GPU), ("tj_c", RE_TJ),
                        ("gr3d_pct", RE_GR3D)):
            vals = [float(m.group(1)) for m in (rx.search(ln) for ln in sel) if m]
            out[key] = float(np.mean(vals)) if vals else float("nan")
        if len(self.rows) > 200000:
            self.rows = self.rows[-1000:]
        return out

    def close(self):
        self.proc.terminate()


class Rig:
    def __init__(self, weights="~/yolov8n.pt", expa_path=None, power_interval_ms=50,
                 imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.E, self.expa_path = load_expa(expa_path)
        self.model = YOLO(os.path.expanduser(weights))
        self.imgsz, self.conf = imgsz, conf
        self.power = PowerLog(power_interval_ms)
        self._devnull = open(os.devnull, "w", newline="")
        self._writer = csv.writer(self._devnull)
        self._cap = None
        self._hooked = False
        dummy = np.full((120, 160), 30000, np.uint16)
        for _ in range(12):  # load weights, CUDA init, cuDNN autotune
            self._one(dummy, {})
        head = self.model.model.model[-1]
        head.register_forward_hook(self._hook)
        self._hooked = True

    def _hook(self, module, inputs, output):
        if self._cap is not None:
            self._cap.append(output[0] if isinstance(output, (tuple, list)) else output)

    def _run_yolo(self, frame, overrides):
        """Same body as experimentA.run_yolo, plus optional predict overrides (ablations)."""
        E = self.E
        if not overrides:
            return E.run_yolo(self.model, frame, self.imgsz, self.conf, None, [0])[1]
        start = time.perf_counter()
        kwargs = {"source": frame, "imgsz": self.imgsz, "conf": self.conf, "verbose": False}
        kwargs.update(overrides)
        results = self.model.predict(**kwargs)
        metrics = E.summarize_yolo_result(results[0], [0])
        total = (time.perf_counter() - start) * 1000.0
        internal = sum(float(metrics[k] or 0.0)
                       for k in ("preprocess_ms", "inference_ms", "postprocess_ms"))
        metrics["total_ms"] = total
        metrics["overhead_ms"] = max(0.0, total - internal)
        return metrics

    def _one(self, raw, overrides):
        E = self.E
        t0 = time.perf_counter()
        temp_c = E.raw_to_celsius(raw, 0.01)
        tmin, tmax, tmean = float(np.min(temp_c)), float(np.max(temp_c)), float(np.mean(temp_c))
        bie_frame = E.radiometric_bie_image(raw, 10.0, 50.0, 0.01)
        bie = E.block_image_entropy(bie_frame, 15, 1.0)
        t1 = time.perf_counter()
        frame = E.yolo_frame_from_thermal(bie_frame)
        t2 = time.perf_counter()
        m = self._run_yolo(frame, overrides)
        t3 = time.perf_counter()
        self._writer.writerow([time.time(), tmin, tmax, tmean, bie["value"], bie["adp"],
                               bie["mean_block_entropy"], m["count"], m["threat_count"],
                               m["max_confidence"], m["class_counts_json"],
                               m["detections_json"], m["total_ms"]])
        t4 = time.perf_counter()
        return {
            "bie_ms": (t1 - t0) * 1e3, "prep_ms": (t2 - t1) * 1e3,
            "pre_ms": m["preprocess_ms"], "inf_ms": m["inference_ms"],
            "post_ms": m["postprocess_ms"], "overhead_ms": m["overhead_ms"],
            "log_ms": (t4 - t3) * 1e3, "frame_ms": (t4 - t0) * 1e3,
            "yolo_total_ms": m["total_ms"], "n_det": int(m["count"] or 0),
            "ht": bie["value"], "bie_frame": bie_frame, "classes": m["class_counts_json"],
        }

    def info(self, frames, overrides=None):
        """Untimed deterministic pass: HT, Shannon entropy, detections, pre-NMS candidates."""
        overrides = overrides or {}
        conf = overrides.get("conf", self.conf)
        ht, sh, nd, nc, s35, s50 = [], [], [], [], [], []
        classes = ""
        for raw in frames:
            self._cap = []
            r = self._one(raw, overrides)
            pred = self._cap[-1]
            self._cap = None
            score = pred[0, 4:].amax(0)
            nc.append(int((score > conf).sum().item()))
            s35.append(int((score > 0.10).sum().item()))
            s50.append(int((score > 0.50).sum().item()))
            ht.append(r["ht"])
            sh.append(self.E.shannon_entropy(r["bie_frame"]))
            nd.append(r["n_det"])
            classes = r["classes"]
        return {"ht": float(np.mean(ht)), "shannon": float(np.mean(sh)),
                "n_det": float(np.mean(nd)), "n_cand": float(np.mean(nc)),
                "n_cand10": float(np.mean(s35)), "n_cand50": float(np.mean(s50)),
                "classes": classes}

    def measure(self, frames, k=24, warm=3, fps=None, overrides=None, with_info=True):
        """Process the bank cyclically for k timed frames; returns per-frame means."""
        overrides = overrides or {}
        n = len(frames)
        out = self.info(frames, overrides) if with_info else {}
        for i in range(warm):
            self._one(frames[i % n], overrides)
        acc = {s: [] for s in STAGES + ["frame_ms", "yolo_total_ms"]}
        period = (1.0 / fps) if fps else None
        t_wall0 = time.time()
        p0 = time.perf_counter()
        for i in range(k):
            r = self._one(frames[i % n], overrides)
            for s in acc:
                acc[s].append(r[s])
            if period:
                rest = p0 + (i + 1) * period - time.perf_counter()
                if rest > 0:
                    time.sleep(rest)
        dur = time.perf_counter() - p0
        t_wall1 = time.time()
        for s, v in acc.items():
            out[s] = float(np.mean(v))
        out["frame_ms_median"] = float(np.median(acc["frame_ms"]))
        out["frame_ms_p95"] = float(np.percentile(acc["frame_ms"], 95))
        out["frame_ms_sd"] = float(np.std(acc["frame_ms"]))
        out.update(self.power.window(t_wall0, t_wall1))
        out["window_s"] = dur
        out["k"] = k
        out["fps"] = fps or 0.0
        out["energy_mj"] = out["vdd_in_mw"] * dur / k  # mW * s = mJ, per frame
        out["t"] = t_wall0
        return out

    def close(self):
        self.power.close()
