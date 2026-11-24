"""Read-only access to the real Lepton recordings, with a local stacked cache.

A trial directory holds ``thermal_raw/raw_%06d.npy`` (uint16, 120x160, TLinear 0.01 K/count)
and ``trial_data.csv`` (per-frame HT, YOLO detections, timings) written by experimentA.py.
Nothing here writes to the original project; stacks are cached under ``data_local/``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "THERMAL_DATA_ROOT",
        "/Users/alibaduqail/Documents/thermal camera research work/data",
    )
)
CACHE_DIR = Path(os.environ.get("THERMAL_CACHE_DIR", REPO / "data_local" / "stacks"))

# session name -> (relative directory, human description)
SESSIONS = {
    "hl_off": ("experiment_A_heat_lamp/lamp_off", "2026-10-08 evening, empty room, lamp off, 60 s"),
    "hl_on": ("experiment_A_heat_lamp/lamp_on", "2026-10-08 night, person + heat lamp on, 60 s"),
    "w120": ("experiment_A_warmup60_capture120", "2026-10-09, 60 s warm-up, 120 s capture"),
    "rj": ("experiment_A_radiometric_jetson", "first radiometric Jetson session, 60 s"),
}
K_PER_COUNT = 0.01


def counts_to_c(raw):
    return np.asarray(raw, dtype=np.float32) * K_PER_COUNT - 273.15


def c_to_counts(temp_c):
    return (np.asarray(temp_c, dtype=np.float64) + 273.15) / K_PER_COUNT


def list_trials(session, data_root=DEFAULT_DATA_ROOT):
    """Return sorted trial ids (e.g. 'EXP-A_A0_run01') that have saved raw frames."""
    root = Path(data_root) / SESSIONS[session][0]
    out = []
    for d in sorted(root.glob("EXP-A_A?_run??")):
        if (d / "thermal_raw").is_dir() and any((d / "thermal_raw").iterdir()):
            out.append(d.name)
    return out


def trial_dir(session, trial_id, data_root=DEFAULT_DATA_ROOT):
    return Path(data_root) / SESSIONS[session][0] / trial_id


def load_stack(session, trial_id, data_root=DEFAULT_DATA_ROOT, use_cache=True):
    """Return the uint16 (N,120,160) stack of raw TLinear frames of one trial."""
    cache = CACHE_DIR / session / f"{trial_id}.npy"
    if use_cache and cache.exists():
        return np.load(cache)
    files = sorted((trial_dir(session, trial_id, data_root) / "thermal_raw").glob("raw_*.npy"))
    if not files:
        raise FileNotFoundError(f"no raw frames for {session}/{trial_id}")
    stack = np.stack([np.load(f) for f in files]).astype(np.uint16)
    if stack.shape[1:] != (120, 160):
        raise ValueError(f"unexpected frame shape {stack.shape}")
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, stack)
    return stack


def load_trial_table(session, trial_id, data_root=DEFAULT_DATA_ROOT):
    """Per-frame rows of trial_data.csv as a dict of lists (numbers parsed where possible)."""
    path = trial_dir(session, trial_id, data_root) / "trial_data.csv"
    cols = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def recorded_detections(table):
    """Per-frame list of (class_name, conf, [x0,y0,x1,y1]) recorded on the Jetson."""
    out = []
    for s in table["yolo_detections_json"]:
        dets = json.loads(s) if s else []
        out.append([(d["class_name"], d["confidence"], d["xyxy"]) for d in dets])
    return out


def fcol(table, name):
    return np.array([float(v) if v not in ("", "None") else np.nan for v in table[name]])
