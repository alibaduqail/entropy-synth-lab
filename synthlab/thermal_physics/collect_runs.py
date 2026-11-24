"""Collect shared-harness run folders into one compact JSON (runs on the Jetson).

  python3 -m synthlab.thermal_physics.collect_runs --runs ~/synthlab/thermal-physics/runs --prefix hb_ --out X.json
Each run folder was written by ~/synthlab/harness/run_thermal_bank.py (owner: jetson-harness).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def collect_one(d):
    s = json.load(open(os.path.join(d, "summary.json")))
    row = dict(run=os.path.basename(d.rstrip("/")), bank=s.get("bank_name"), mode=s.get("mode"), backend=s.get("backend"),
               frames=s.get("frames"), achieved_fps=s.get("achieved_fps"), power_w=s.get("power_w"),
               idle_power_w=s.get("idle_power_w"), energy_per_frame_j=s.get("energy_per_frame_j"),
               gpu_pct=s.get("gpu_pct"), cpu_pct=s.get("cpu_pct"), temp_gpu_c=s.get("temp_gpu_c"),
               ok=s.get("ok"), problems=s.get("problems"), start_unix=s.get("measurement_start_unix"))
    for k in ("HT", "n_det", "n_person", "n_candidates"):
        if isinstance(s.get(k), dict):
            row[f"{k.lower() if k != 'HT' else 'ht'}_mean"] = s[k].get("mean")
            row[f"{k.lower() if k != 'HT' else 'ht'}_sd"] = s[k].get("sd")
    for k, v in (s.get("stages_ms") or {}).items():
        if isinstance(v, dict):
            row[f"{k}_mean"] = v.get("mean")
            row[f"{k}_p95"] = v.get("p95")
            row[f"{k}_sd"] = v.get("sd")
    p = os.path.join(d, "frames.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        conf = np.array([_f(r.get("max_conf")) for r in rows])
        npers = np.array([_f(r.get("n_person")) for r in rows])
        ndet = np.array([_f(r.get("n_det")) for r in rows])
        row.update(frac_frames_with_person=float(np.mean(npers > 0)), frac_frames_with_det=float(np.mean(ndet > 0)),
                   max_conf_mean=float(np.nanmean(conf)) if np.isfinite(conf).any() else None,
                   n_rows=len(rows))
        # per-frame correlation inside the run (frame-level HT vs time)
        ht = np.array([_f(r.get("HT")) for r in rows])
        tot = np.array([_f(r.get("total_ms")) for r in rows])
        inf = np.array([_f(r.get("yolo_inf_ms")) for r in rows])
        post = np.array([_f(r.get("yolo_post_ms")) for r in rows])
        if np.std(ht) > 0:
            row["within_run_r_ht_total_ms"] = float(np.corrcoef(ht, tot)[0, 1])
        if np.std(ndet) > 0:
            row["within_run_r_ndet_post_ms"] = float(np.corrcoef(ndet, post)[0, 1])
            row["within_run_r_ndet_inf_ms"] = float(np.corrcoef(ndet, inf)[0, 1])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for d in sorted(glob.glob(os.path.join(os.path.expanduser(a.runs), a.prefix + "*", ""))):
        subs = sorted(glob.glob(os.path.join(d, "trial_*", ""))) or [d]
        for sd in subs:
            if os.path.exists(os.path.join(sd, "summary.json")):
                r = collect_one(sd)
                r["run"] = os.path.relpath(sd, os.path.expanduser(a.runs)).rstrip("/")
                out.append(r)
    json.dump(dict(rows=out), open(os.path.expanduser(a.out), "w"), indent=1)
    print(len(out), "runs collected")
