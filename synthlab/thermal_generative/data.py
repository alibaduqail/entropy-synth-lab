"""Dataset index, trial-level split, pack builder and loader for the real Lepton recordings.

Labels are TRIAL-level: person (A1 vs A0), lamp (heat-lamp session lamp_on), session id.
Split is by whole trial (frames inside a trial are near-duplicates at 8.7 fps).

Build the pack on the Jetson (reads the read-only recordings, writes only to our folder):
    python3 -m synthlab.thermal_generative.data --out ~/synthlab/thermal-generative/data/pack.npz
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

JETSON_ROOT = "~/thermal-camera-radiometric/jetson_experiment_A_thermal_yolo_handoff"
# (session name, session id, sub-folder, lamp flag for A1 trials in that folder)
SOURCES = [
    ("heat_lamp", 0, "experiment_A_heat_lamp/lamp_off", 0),
    ("heat_lamp", 0, "experiment_A_heat_lamp/lamp_on", 1),
    ("warmup120", 1, "data/experiment_A_warmup60_capture120", 0),
    ("radiometric", 2, "experiment_A_radiometric", 0),
]
N_SESSIONS = 3
SESSION_NAMES = ["heat_lamp", "warmup120", "radiometric"]


def split_of(session, person, run, runs_in_group):
    """Deterministic whole-trial hold-out: every 5th run of the 25-run groups, and the
    last run of each small (3-6 run) group."""
    if len(runs_in_group) >= 10:
        return "test" if run % 5 == 0 else "train"
    return "test" if run == max(runs_in_group) else "train"


def index_trials(root=JETSON_ROOT):
    root = Path(os.path.expanduser(root))
    groups = {}
    for sname, sid, sub, lamp in SOURCES:
        for d in sorted((root / sub).glob("EXP-A_A[01]_run*")):
            raw = d / "thermal_raw"
            n = len(list(raw.glob("raw_*.npy"))) if raw.is_dir() else 0
            if n < 100:
                continue
            person = int(d.name.split("_")[1][1])
            run = int(d.name.split("run")[1])
            t = dict(name=f"{sname}/{d.name}", path=str(d), session=sid, session_name=sname,
                     person=person, lamp=lamp if person else 0, run=run, n_raw=n)
            if sub.endswith("lamp_on"):
                t["lamp"] = 1
            groups.setdefault((sub, person), []).append(t)
    trials = []
    for (_, person), ts in groups.items():
        runs = [t["run"] for t in ts]
        for t in ts:
            t["split"] = split_of(t["session"], person, t["run"], runs)
            trials.append(t)
    return trials


def build_pack(out, stride=3, root=JETSON_ROOT):
    trials = index_trials(root)
    frames, tidx, fid, ht, ythreat, yms = [], [], [], [], [], []
    for ti, t in enumerate(trials):
        rows = {}
        with open(Path(t["path"]) / "trial_data.csv") as f:
            for r in csv.DictReader(f):
                rows[int(r["frame_id"])] = r
        files = sorted((Path(t["path"]) / "thermal_raw").glob("raw_*.npy"))
        kept = 0
        for p in files[::stride]:
            k = int(p.stem.split("_")[1])
            a = np.load(p)
            if a.shape != (120, 160) or a.dtype != np.uint16:
                continue
            r = rows.get(k, {})
            frames.append(a)
            tidx.append(ti)
            fid.append(k)
            ht.append(float(r.get("HT") or "nan"))
            ythreat.append(float(r.get("yolo_threat_count") or "nan"))
            yms.append(float(r.get("yolo_inference_ms") or "nan"))
            kept += 1
        t["n"] = kept
        print(t["name"], t["split"], kept, flush=True)
    out = Path(os.path.expanduser(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=np.stack(frames), trial=np.array(tidx, np.int32),
             frame_id=np.array(fid, np.int32), ht=np.array(ht, np.float32),
             yolo_threat=np.array(ythreat, np.float32), yolo_ms=np.array(yms, np.float32),
             trials=json.dumps(trials), stride=stride)
    print("wrote", out, len(frames), "frames")


class Pack:
    """In-memory view of the pack with per-frame labels derived from trial labels."""

    def __init__(self, path):
        z = np.load(os.path.expanduser(str(path)), allow_pickle=False)
        self.frames = z["frames"]
        self.trial = z["trial"]
        self.frame_id = z["frame_id"]
        self.ht = z["ht"]
        self.yolo_threat = z["yolo_threat"]
        self.yolo_ms = z["yolo_ms"]
        self.trials = json.loads(str(z["trials"]))
        g = lambda k: np.array([self.trials[i][k] for i in self.trial])
        self.person = g("person").astype(np.int64)
        self.lamp = g("lamp").astype(np.int64)
        self.session = g("session").astype(np.int64)
        self.is_test = g("split") == "test"

    def mask(self, split=None, person=None, lamp=None, session=None):
        m = np.ones(len(self.frames), bool)
        if split is not None:
            m &= self.is_test == (split == "test")
        if person is not None:
            m &= self.person == person
        if lamp is not None:
            m &= self.lamp == lamp
        if session is not None:
            m &= self.session == session
        return m

    def summary(self):
        out = {}
        for i, t in enumerate(self.trials):
            k = f"{t['session_name']}|person={t['person']}|lamp={t['lamp']}|{t['split']}"
            o = out.setdefault(k, dict(trials=0, frames=0))
            o["trials"] += 1
            o["frames"] += int((self.trial == i).sum())
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--root", default=JETSON_ROOT)
    a = ap.parse_args()
    build_pack(a.out, a.stride, a.root)
