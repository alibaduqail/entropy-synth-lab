"""How small a movement of the person can the radar witness detect?

Each trial: one person on the mark for ``duration`` s (breathing, heartbeat, optional
postural sway); at mid-trial the whole body moves by ``delta`` (0 = no event) toward the
radar ("radial") or sideways ("lateral"). The trial is simulated at signal level, run
through the processing chain, and the stillness metrics are computed. Detection
thresholds come from the no-event trials (max over the baseline seeds).

Extra trials: fog building up during the recording (refractivity + attenuation ramp)
and a water film forming on the radome, both with a perfectly still scene.

Usage: python3 -m synthlab.radar_sim.stillness_study --out results/radar-sim
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from . import config, metrics, process, simulate
from . import scene as sc
from .fog import FOG_LEVELS, FogState
from .person import Person, radial_event

DELTAS_MM = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
METRICS = ["excursion_mm", "net_shift_mm", "lateral_shift_mm", "raw_p2p_mm",
           "range_centroid_shift_mm",
           "motion_db_max", "centroid_drift_mm"]
_CACHE = {}


def _setup(scene_path, preset):
    key = (scene_path, preset)
    if key not in _CACHE:
        cfg = config.preset(preset)
        scene = sc.load_scene(scene_path)
        proc = process.Processor(cfg, keep_maps=False)
        bsim = simulate.RadarSimulator(cfg, scene, seed=999)
        bg = np.mean([proc.process(bsim.frame(i)).zero_doppler for i in range(8)], axis=0)
        _CACHE[key] = (cfg, scene, bg)
    return _CACHE[key]


def run_trial(p: dict) -> dict:
    cfg, scene, bg = _setup(p.get("scene"), p.get("preset", "people_counting_6m"))
    pose = sc.radar_pose(scene)
    mark = scene["dynamic_slots"]["person_mark"][:2]
    person = Person(mark, face_to=pose.position, sway_rms_m=p["sway_mm"] * 1e-3, seed=p["seed"])
    dur = p.get("duration", 16.0)
    if p["delta_mm"] > 0:
        person.events.append(radial_event(person, pose.position, dur / 2, p["delta_mm"] * 1e-3,
                                          duration=p.get("event_s", 1.0),
                                          lateral=(p["direction"] == "lateral")))
    fog, ramp = FogState(), None
    if p.get("fog"):
        base = FOG_LEVELS[p["fog"]]
        fog = FogState(base.name, base.lwc_gm3, rh=base.rh, film_m=p.get("film_um", 0.0) * 1e-6)
        if p.get("fog_ramp", False):
            ramp = lambda t: float(np.clip(t / dur, 0.0, 1.0))
    sim = simulate.RadarSimulator(cfg, scene, [person], fog=fog, seed=p["seed"], fog_ramp=ramp,
                                  clutter_jitter_m=p.get("jitter_um", 0.0) * 1e-6)
    proc = process.Processor(cfg, keep_maps=False, n_sub=8)
    rec = metrics.StillnessRecorder(cfg, bg, proc.az_flat)
    for i in range(int(dur / cfg.frame_period_s)):
        rec.add(proc.process(sim.frame(i)))
    out = dict(p)
    out.update(rec.metrics())
    if p.get("keep_series"):
        out["series_mm"] = [round(float(v), 4) for v in rec.displacement_mm()["d_mm"]]
        out["motion_db"] = [round(float(v), 3) for v in rec.motion_db]
    return out


def build_grid(n_seeds: int, n_base: int, scene, preset):
    grid = []
    for sway in (2.0, 0.0):
        for s in range(n_base):
            grid.append(dict(kind="event", sway_mm=sway, delta_mm=0.0, direction="none",
                             seed=1000 + s))
        for direction in ("radial", "lateral"):
            for d in DELTAS_MM:
                for s in range(n_seeds):
                    grid.append(dict(kind="event", sway_mm=sway, delta_mm=float(d),
                                     direction=direction, seed=2000 + s))
    for s in range(6):   # still scene, medium changes (the radar should NOT report motion)
        grid.append(dict(kind="fog_static", sway_mm=2.0, delta_mm=0.0, direction="none",
                         seed=1000 + s, fog="B3"))
        grid.append(dict(kind="fog_onset", sway_mm=2.0, delta_mm=0.0, direction="none",
                         seed=1000 + s, fog="B3", fog_ramp=True))
        grid.append(dict(kind="fog_onset_ideal", sway_mm=0.0, delta_mm=0.0, direction="none",
                         seed=1000 + s, fog="B3", fog_ramp=True))
        grid.append(dict(kind="film_onset_ideal", sway_mm=0.0, delta_mm=0.0, direction="none",
                         seed=1000 + s, fog="B3", fog_ramp=True, film_um=50.0))
        grid.append(dict(kind="rig_vibration_20um", sway_mm=0.0, delta_mm=0.0, direction="none",
                         seed=1000 + s, jitter_um=20.0))
    for d, direction in ((0.0, "none"), (20.0, "radial"), (200.0, "radial"), (20.0, "lateral")):
        grid.append(dict(kind="example", sway_mm=2.0, delta_mm=d, direction=direction,
                         seed=4242, keep_series=True))
    for g in grid:
        g.update(scene=scene, preset=preset)
    return grid


def summarise(rows):
    out = {"deltas_mm": DELTAS_MM, "metrics": METRICS, "curves": {}, "thresholds": {},
           "baseline": {}, "min_detectable_mm": {}, "medium_changes": {}}
    ev = [r for r in rows if r["kind"] == "event"]
    for sway in sorted({r["sway_mm"] for r in ev}):
        base = [r for r in ev if r["sway_mm"] == sway and r["delta_mm"] == 0]
        sk = "sway_%gmm" % sway
        out["thresholds"][sk], out["baseline"][sk] = {}, {}
        out["curves"][sk], out["min_detectable_mm"][sk] = {}, {}
        for m in METRICS:
            b = np.array([r[m] for r in base], dtype=float)
            b = b[np.isfinite(b)]
            thr = float(b.max()) if len(b) else float("nan")
            out["thresholds"][sk][m] = thr
            out["baseline"][sk][m] = {"median": float(np.median(b)) if len(b) else None,
                                      "max": thr, "n": int(len(b))}
            for direction in ("radial", "lateral"):
                pd, med = [], []
                for d in DELTAS_MM:
                    v = np.array([r[m] for r in ev if r["sway_mm"] == sway
                                  and r["direction"] == direction and r["delta_mm"] == d], dtype=float)
                    pd.append(float(np.mean(np.nan_to_num(v, nan=-np.inf) > thr)))
                    med.append(float(np.nanmedian(v)) if np.isfinite(v).any() else None)
                key = "%s/%s" % (m, direction)
                out["curves"][sk][key] = {"p_detect": pd, "median": med}
                ok = [d for i, d in enumerate(DELTAS_MM) if all(x >= 0.9 for x in pd[i:])]
                out["min_detectable_mm"][sk][key] = ok[0] if ok else None
    for kind in sorted({r["kind"] for r in rows} - {"event", "example"}):
        sel = [r for r in rows if r["kind"] == kind]
        out["medium_changes"][kind] = {
            m: {"median": float(np.nanmedian([r[m] for r in sel])),
                "max": float(np.nanmax([r[m] for r in sel])), "n": len(sel)} for m in METRICS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/radar-sim")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--preset", default="people_counting_6m")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--base-seeds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    grid = build_grid(a.seeds, a.base_seeds, a.scene, a.preset)
    with Pool(a.workers) as pool:
        rows = pool.map(run_trial, grid, chunksize=2)
    examples = [r for r in rows if r["kind"] == "example"]
    cols = ["kind", "sway_mm", "delta_mm", "direction", "seed", "torso_range_m"] + METRICS + \
        ["motion_db_median", "n_det_mean", "frames_with_points_frac", "scene_decorrelation"]
    with open(os.path.join(a.out, "stillness_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in cols})
    summ = summarise(rows)
    cfg = config.preset(a.preset)
    summ["setup"] = {"n_trials": len(rows), "seeds_per_point": a.seeds, "base_seeds": a.base_seeds,
                     "duration_s": 16.0, "event_time_s": 8.0, "event_duration_s": 1.0,
                     "frame_period_s": cfg.frame_period_s, "preset": a.preset,
                     "scene": os.path.basename(sc.load_scene(a.scene)["_path"]),
                     "threshold_rule": "max of the no-event baseline trials (same sway)"}
    with open(os.path.join(a.out, "stillness_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    with open(os.path.join(a.out, "stillness_examples.json"), "w") as f:
        json.dump([{k: r[k] for k in ("delta_mm", "direction", "series_mm", "motion_db")}
                   for r in examples], f)
    print(json.dumps(summ["min_detectable_mm"], indent=1))
    print(json.dumps(summ["baseline"], indent=1))
    print(json.dumps(summ["medium_changes"], indent=1))


if __name__ == "__main__":
    main()
