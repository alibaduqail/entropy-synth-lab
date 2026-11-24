"""tegrastats logging, parsing and summary statistics shared by both bank runners.

The logger and parser follow the earlier project harness (scripts/experimentA.py
start_tegrastats and scripts/replay_heat_lamp.py parse_tegra): every tegrastats line is
stamped with the host Unix time when it is read, because tegrastats itself only prints
whole seconds. VDD_IN is the module input power, not wall power.
"""

import csv
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


def tegrastats_available():
    return shutil.which("tegrastats") is not None


class TegraLogger:
    """Run tegrastats in the background and write `unix_time,tegrastats_line` rows."""

    def __init__(self, path, interval_ms=200):
        self.path, self.interval_ms = Path(path), int(interval_ms)
        self.proc = self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)

        def reader():
            with open(self.path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["unix_time", "tegrastats_line"])
                while not self.stop_event.is_set():
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    writer.writerow([f"{time.time():.4f}", line.strip()])
                    handle.flush()

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.proc is None:
            return
        self.stop_event.set()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def parse_line(line):
    """One tegrastats line -> dict of numbers, or None if it is not a stats line."""
    ram = re.search(r"RAM (\d+)/(\d+)MB", line)
    gpu = re.search(r"GR3D_FREQ (\d+)%", line)
    cpu = re.search(r"CPU \[([^]]+)\]", line)
    if not (ram and gpu and cpu):
        return None
    cores = [float(v) for v in re.findall(r"(\d+)%", cpu.group(1))]
    freqs = [float(v) for v in re.findall(r"@(\d+)", cpu.group(1))]
    if not cores:
        return None
    sample = {"ram_mb": float(ram.group(1)), "gpu_pct": float(gpu.group(1)),
              "cpu_pct": float(np.mean(cores)), "cpu_max_core_pct": float(max(cores))}
    if freqs:
        sample["cpu_freq_mhz"] = float(np.mean(freqs))
    for name, value in re.findall(r"(\w+)@([\d.]+)C", line):
        sample[f"temp_{name}_c"] = float(value)
    for name, value in re.findall(r"\b(VDD_\w+)\s+(\d+)mW/", line):
        sample[f"{name.lower()}_w"] = float(value) / 1000.0
    return sample


def parse_file(path):
    samples = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            sample = parse_line(row.get("tegrastats_line") or "")
            if sample is not None:
                sample["unix_time"] = float(row["unix_time"])
                samples.append(sample)
    return samples


def dist(values):
    """Mean, SD and percentiles of a 1-D sample (None if empty)."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    p = np.percentile(x, [50, 95, 99])
    return {"n": int(x.size), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "p50": float(p[0]), "p95": float(p[1]),
            "p99": float(p[2]), "max": float(x.max())}


def summarize(path, start, end, interval_s, n_units=None):
    """Telemetry inside [start, end] (Unix s): per-metric stats, trapezoid energy of
    VDD_IN, and coverage checks. `n_units` (frames or windows) gives energy per unit."""
    samples = parse_file(path)
    inside = [s for s in samples if start <= s["unix_time"] <= end]
    stamps = np.asarray([s["unix_time"] for s in samples])
    duration = end - start
    expected = duration / interval_s
    problems = []
    if stamps.size == 0:
        problems.append("no telemetry samples")
    elif np.any(np.diff(stamps) <= 0):
        problems.append("non-monotonic telemetry timestamps")
    if len(inside) < 0.8 * max(1.0, expected - 1):
        problems.append(f"sample count {len(inside)} below 80% of expected {expected:.0f}")
    near = stamps[(stamps >= start - 3 * interval_s) & (stamps <= end + 3 * interval_s)]
    max_gap = float(np.max(np.diff(near))) if near.size > 1 else float("nan")
    if near.size < 2 or near.min() > start + 3 * interval_s or near.max() < end - 3 * interval_s:
        problems.append("telemetry does not span the measured window")
    elif max_gap > max(1.0, 5 * interval_s):
        problems.append(f"telemetry gap of {max_gap:.2f} s")
    keys = sorted({k for s in inside for k in s} - {"unix_time"})
    metrics = {k: dist(s[k] for s in inside if k in s) for k in keys}
    out = {"samples": len(inside), "expected_samples": float(expected),
           "coverage_frac": float(len(inside) / expected) if expected > 0 else 0.0,
           "max_gap_s": max_gap, "interval_s": interval_s, "metrics": metrics}
    power = [(s["unix_time"], s["vdd_in_w"]) for s in samples if "vdd_in_w" in s]
    if len(power) >= 2 and duration > 0:
        t, w = (np.asarray(v) for v in zip(*power))
        grid = np.r_[start, t[(t > start) & (t < end)], end]
        watts = np.interp(grid, t, w)
        energy = float(np.sum((watts[1:] + watts[:-1]) * 0.5 * np.diff(grid)))
        out.update(energy_j=energy, power_w_time_weighted=energy / duration)
        if n_units:
            out["energy_per_unit_j"] = energy / n_units
    else:
        problems.append("VDD_IN power missing")
    out["problems"] = problems
    out["ok"] = not problems
    return out


def idle_power(path, start, end):
    """Mean VDD_IN (W) over an idle window, or None."""
    values = [s["vdd_in_w"] for s in parse_file(path)
              if start <= s["unix_time"] <= end and "vdd_in_w" in s]
    return float(np.mean(values)) if values else None


def wait_until(deadline):
    """Sleep until a time.monotonic() deadline; return how late we were (s)."""
    remaining = deadline - time.monotonic()
    while remaining > 0:
        time.sleep(remaining)
        remaining = deadline - time.monotonic()
    return max(0.0, time.monotonic() - deadline)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_jsonable) + "\n")


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def environment():
    """Software/hardware state recorded with every run."""
    info = {"unix_time": time.time()}
    for name in ["numpy", "scipy", "torch", "cv2", "ultralytics", "tensorrt"]:
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    for key, command in {"nvpmodel": ["nvpmodel", "-q"], "uname": ["uname", "-r"]}.items():
        try:
            info[key] = subprocess.run(command, capture_output=True, text=True,
                                       timeout=5).stdout.strip()
        except Exception:
            info[key] = None
    return info
