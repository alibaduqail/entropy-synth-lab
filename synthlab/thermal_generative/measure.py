"""Run our banks through the shared harness (resumable; run under `tools/jetson.sh measured`),
then collect the per-bank cost table.

    python3 -m synthlab.thermal_generative.measure run --minutes 15 [--unpaced] [--rep 0]
    python3 -m synthlab.thermal_generative.measure collect   # -> out/harness_runs.csv
Order of banks is a seeded shuffle (different per repetition) so slow drift does not line up with HT.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HOME = os.path.expanduser("~/synthlab/thermal-generative")
HARNESS = os.path.expanduser("~/synthlab/harness/run_thermal_bank.py")


def run(a):
    banks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{HOME}/banks/tg_*/manifest.json"))
    mode = "unpaced" if a.unpaced else "paced"
    order = np.random.default_rng(100 + a.rep + (50 if a.unpaced else 0)).permutation(len(banks))
    t_end = time.time() + a.minutes * 60
    for i in order:
        b = banks[i]
        out = f"{HOME}/runs/{mode}_rep{a.rep}/{b}"
        if os.path.exists(f"{out}/summary.json"):
            continue
        if time.time() + a.seconds + a.warmup + 15 > t_end:
            print("time budget reached; rerun to continue")
            return
        if os.path.isdir(out):
            subprocess.run(["rm", "-rf", out])
        cmd = [sys.executable, HARNESS, "--bank", f"{HOME}/banks/{b}", "--out", out, "--seconds", str(a.seconds),
               "--warmup", str(a.warmup)] + (["--unpaced"] if a.unpaced else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(b, mode, "rc", r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:160], flush=True)
        if r.returncode != 0:
            print(r.stderr[-600:], flush=True)
    print("measure complete")


def collect(a):
    rows = []
    for sp in sorted(glob.glob(f"{HOME}/runs/*/*/summary.json")):
        d = os.path.dirname(sp)
        bank = os.path.basename(d)
        s = json.load(open(sp))
        man = json.load(open(f"{HOME}/banks/{bank}/manifest.json"))
        fr = list(csv.DictReader(open(f"{d}/frames.csv")))
        col = lambda k: np.array([float(r[k]) for r in fr if r.get(k) not in (None, "")])
        row = dict(bank=bank, run=os.path.basename(os.path.dirname(d)), n_rows=len(fr),
                   generator=man.get("generator", ""), summary_ok=s.get("ok"))
        for k in ("HT", "total_ms", "bie_ms", "yolo_inf_ms", "yolo_post_ms", "n_det", "n_person", "n_candidates"):
            v = col(k)
            if len(v):
                row[k + "_mean"] = float(v.mean())
                if k.endswith("_ms"):
                    row[k + "_p95"] = float(np.percentile(v, 95))
        row["person_frame_rate"] = float(np.mean(col("n_person") > 0)) if len(col("n_person")) else float("nan")
        for k, v in s.items():
            if isinstance(v, (int, float)) and ("power" in k or "energy" in k):
                row[k] = v
        rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(f"{HOME}/out/harness_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(len(rows), "runs ->", f"{HOME}/out/harness_runs.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "collect"])
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--unpaced", action="store_true")
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    dict(run=run, collect=collect)[a.cmd](a)
