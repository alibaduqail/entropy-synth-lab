"""Generate and process every condition, timing each stage and logging power.

For each (repeat, condition), in a seeded shuffled order:
  idle   - sleep, power only (baseline)
  gen    - simulate ``n_frames`` raw ADC cubes (timed per frame)
  free   - run the processing chain flat out over those frames for ``proc_seconds``
  paced  - run the chain at the radar's real frame rate for ``paced_seconds``
  (untimed) - entropies / diagnostics on the same frames
Optionally a "stress" sweep: the busiest condition processed with lowered CFAR thresholds
to span a wide range of detection counts (cost-vs-detections curve).

Usage:  python3 -m synthlab.radar_sim.run_conditions --out results/radar-sim/jetson_bench
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from . import conditions as cond
from . import config, metrics, process, simulate
from . import scene as sc
from .power import TegraLogger

STAGE_COLS = process.STAGES + ["fixed", "data_dependent", "total"]


def _proc_pass(proc, frames, seconds, paced_period=None):
    rows = []
    t_start = time.perf_counter()
    i = 0
    while True:
        now = time.perf_counter()
        if now - t_start >= seconds:
            break
        if paced_period is not None:
            target = t_start + i * paced_period
            if target > now:
                time.sleep(target - now)
        res = proc.process(frames[i % len(frames)])
        row = {k: res.timings_ms[k] for k in STAGE_COLS}
        row.update(frame=i, n_det=res.n_det, n_clusters=res.n_clusters, n_tracks=res.n_tracks)
        rows.append(row)
        i += 1
    return rows


def _stats(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "n": int(len(v))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--preset", default="people_counting_6m")
    ap.add_argument("--conditions", default=",".join(cond.CONDITIONS))
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--idle-seconds", type=float, default=3.0)
    ap.add_argument("--proc-seconds", type=float, default=6.0)
    ap.add_argument("--paced-seconds", type=float, default=6.0)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    cfg = config.preset(a.preset)
    scene = sc.load_scene(a.scene)
    names = a.conditions.split(",")
    rng = np.random.default_rng(a.seed)
    power = TegraLogger(100).start()
    time.sleep(1.0)
    frame_rows, summary = [], []
    # warm-up (FFT plans, caches) so the first condition is not penalised
    wsim = simulate.RadarSimulator(cfg, scene, seed=a.seed)
    wproc = process.Processor(cfg, keep_maps=False)
    for i in range(10):
        wproc.process(wsim.frame(i))

    for rep in range(a.repeats):
        order = list(rng.permutation(names))
        for name in order:
            c = cond.build_condition(name, scene, seed=a.seed + rep)
            entry = {"condition": name, "repeat": rep, "description": c["description"]}
            t0 = time.time(); time.sleep(a.idle_seconds); t1 = time.time()
            entry["idle"] = power.window(t0, t1)
            sim = simulate.RadarSimulator(cfg, scene, c["people"], c["extra_objects"], c["fog"],
                                          seed=a.seed + rep)
            frames, gen_ms = [], []
            t0 = time.time()
            for i in range(a.n_frames):
                ta = time.perf_counter()
                frames.append(sim.frame(i))
                gen_ms.append((time.perf_counter() - ta) * 1e3)
            t1 = time.time()
            entry["gen"] = dict(power.window(t0, t1), ms_per_frame=float(np.mean(gen_ms)),
                                ms_std=float(np.std(gen_ms)), n=len(gen_ms),
                                n_static_scatterers=len(sim.static),
                                n_moving_scatterers=sum(len(p) for p in c["people"]))
            for mode, secs, period in (("free", a.proc_seconds, None),
                                       ("paced", a.paced_seconds, cfg.frame_period_s)):
                proc = process.Processor(cfg, keep_maps=False)
                t0 = time.time()
                rows = _proc_pass(proc, frames, secs, period)
                t1 = time.time()
                e = power.window(t0, t1)
                for k in STAGE_COLS + ["n_det", "n_clusters", "n_tracks"]:
                    e[k] = _stats(rows, k)
                e["fps"] = len(rows) / (t1 - t0)
                entry[mode] = e
                for r in rows:
                    r.update(condition=name, repeat=rep, mode=mode)
                frame_rows.extend(rows)
            proc = process.Processor(cfg, keep_maps=True)
            ents = [metrics.frame_entropies(proc.process(f), cfg) for f in frames]
            entry["entropy"] = {k: float(np.mean([e[k] for e in ents])) for k in ents[0]}
            summary.append(entry)
            print("rep %d %-3s gen %.1f ms  proc %.2f ms (fixed %.2f, data %.3f)  ndet %.1f  "
                  "P_free %s mW P_paced %s mW" % (
                      rep, name, entry["gen"]["ms_per_frame"], entry["free"]["total"]["mean"],
                      entry["free"]["fixed"]["mean"], entry["free"]["data_dependent"]["mean"],
                      entry["free"]["n_det"]["mean"], entry["free"]["vdd_in_mw_mean"],
                      entry["paced"]["vdd_in_mw_mean"]), flush=True)
            if a.stress and name == "M3":
                entry["stress"] = []
                for thr in (15, 12, 9, 7, 5, 4, 3, 2, 1):
                    proc = process.Processor(cfg, cfar_thr_db=thr, doppler_thr_db=max(thr - 3, 0.5),
                                             keep_maps=False, max_points=4000)
                    t0 = time.time()
                    rows = _proc_pass(proc, frames, 3.0)
                    t1 = time.time()
                    e = dict(power.window(t0, t1), cfar_thr_db=thr)
                    for k in STAGE_COLS + ["n_det", "n_clusters"]:
                        e[k] = _stats(rows, k)
                    entry["stress"].append(e)
                    for r in rows:
                        r.update(condition="M3_thr%d" % thr, repeat=rep, mode="stress")
                    frame_rows.extend(rows)
                    print("   stress thr %2d dB: ndet %.0f total %.2f ms (data %.2f) P %s" % (
                        thr, e["n_det"]["mean"], e["total"]["mean"], e["data_dependent"]["mean"],
                        e["vdd_in_mw_mean"]), flush=True)
    power.stop()
    cols = ["condition", "repeat", "mode", "frame", "n_det", "n_clusters", "n_tracks"] + STAGE_COLS
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in frame_rows:
            w.writerow({k: (round(r[k], 5) if isinstance(r[k], float) else r[k]) for k in cols})
    meta = {"args": vars(a), "radar": cfg.summary(), "scene": os.path.basename(scene["_path"]),
            "host": platform.node(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__,
            "tegrastats": power.available, "n_power_samples_total": len(power.samples),
            "summary": summary}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    with open(os.path.join(a.out, "power_trace.csv"), "w") as f:
        f.write("unix_s,vdd_in_mw,cpu_temp_c\n")
        for t, mw, tc in power.samples:
            f.write("%.3f,%.0f,%.1f\n" % (t, mw, tc))


if __name__ == "__main__":
    main()
