"""Write frame banks in the shared hand-off format (frames.npy uint16 (N,120,160) + manifest.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import ht_stack


def write_bank(out_dir, frames, manifest):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint16 or frames.ndim != 3 or frames.shape[1:] != (120, 160):
        raise ValueError(f"bank frames must be uint16 (N,120,160), got {frames.dtype} {frames.shape}")
    man = dict(manifest)
    man["n_frames"] = int(len(frames))
    hts = ht_stack(frames)
    man["HT_mean"] = float(hts.mean())
    man["HT_sd"] = float(hts.std())
    man["temp_mean_c"] = float(frames.mean() * 0.01 - 273.15)
    man["temp_max_c"] = float(frames.max() * 0.01 - 273.15)
    np.save(out / "frames.npy", frames)
    (out / "manifest.json").write_text(json.dumps(man))
    return man


def real_bank(session, trial_id, n_frames=128, start=None, seed=0):
    """A bank of PURE REAL consecutive frames from one recorded trial (anchor banks)."""
    from . import dataio
    st = dataio.load_stack(session, trial_id)
    rng = np.random.default_rng(seed)
    s = int(rng.integers(0, len(st) - n_frames + 1)) if start is None else start
    frames = st[s:s + n_frames]
    tab = dataio.load_trial_table(session, trial_id)
    det = dataio.recorded_detections(tab)[s:s + n_frames]
    labels = [{"recorded_n_det": len(d), "recorded_n_person": sum(c == "person" for c, _, _ in d)}
              for d in det]
    man = {"generator": "REAL recording (no synthesis)", "units": "TLinear 0.01 K/count",
           "source_session": session, "source_trial": trial_id, "start_frame": s, "seed": seed,
           "labels": labels,
           "note": "labels are the YOLO outputs recorded on the Jetson at capture time, not ground truth"}
    return frames, man
