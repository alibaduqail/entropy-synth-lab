"""Scene conditions (A0-A3, B0-B3, plume, arbitrary fog) and LiDAR bank generation.

Bank format (AGENTS.md hand-off): <out>/frame_00000.npz ... each one 100 ms Mid-360 frame with
xyz float32 (M,3) sensor frame, intensity uint8, tag uint8, plus <out>/manifest.json.

  python -m synthlab.lidar_sim.bank --condition B2 --windows 5 --out ~/synthlab/lidar-sim/banks/B2
  python -m synthlab.lidar_sim.bank --condition A1 --alpha 0.45 --out ... (custom fog on any scene)
"""
import argparse
import json
import os
import time

import numpy as np

from .fog import Fog, alpha_to_mor
from .scene import Box, Capsule, Scene, Sphere, person
from .sensor import SensorParams, sensor_describe, simulate_frame

# Fog levels: uniform extinction [1/m]. ASSUMED mapping (no measurement of the real mist exists):
#   B1 light haze  MOR 30 m, B2 visible fog MOR 10 m, B3 dense room-filling fog MOR 3 m.
B_ALPHA = {"B0": 0.0, "B1": 0.10, "B2": 0.30, "B3": 1.00}
# Plume-only variants: mist concentrated above dynamic_slots.fog_source, clear elsewhere.
P_ALPHA = {"P1": 1.0, "P2": 3.0, "P3": 10.0}

# Layouts are given for room_v1 (2.56 x 2.66 m, desk in the middle-north, sensor on the desk's
# south end) and scaled in xy for any other room size.
V1_SIZE = (2.56, 2.66)
EXTRA_PEOPLE = [(0.45, 1.95), (2.10, 2.15)]          # A2/A3 standing people (v1 world xy)
WALK = dict(y=0.70, x0=0.85, x1=2.15, speed=0.8)     # A2/A3 walker path (v1)


def _sc(base, xy):
    return (xy[0] * base.size[0] / V1_SIZE[0], xy[1] * base.size[1] / V1_SIZE[1])


def clutter(base):
    """Tall clutter for A3 (must rise above ~1 m: the Mid-360 at ~1.12 m with a -7 deg lower FOV
    limit cannot see low objects inside this room)."""
    def b(c, size, refl, name):
        x, y = _sc(base, c[:2])
        return Box(np.array([x, y, c[2]]), np.array(size, float), refl, name)
    px, py = _sc(base, (2.25, 1.20))
    cx, cy = _sc(base, (0.30, 2.40))
    return [b((0.18, 1.25, 0.90), (0.30, 0.60, 1.80), 0.45, "shelf"),
            b((2.20, 2.35, 0.70), (0.45, 0.45, 1.40), 0.60, "box_stack"),
            b((1.39, 2.30, 0.95), (0.30, 0.30, 0.42), 0.55, "desk_box"),
            b((0.60, 0.95, 0.80), (0.06, 0.50, 1.60), 0.35, "whiteboard"),
            Capsule(np.array([cx, cy]), 0.03, 1.85, 0.03, 0.30, "coat_stand"),
            Sphere(np.array([cx, cy, 1.60]), 0.20, 0.25, "coat"),
            Sphere(np.array([px, py, 1.35]), 0.25, 0.35, "plant"),
            Capsule(np.array([px, py]), 0.0, 1.10, 0.04, 0.30, "plant_stem")]


def build_scene(base, condition, t=0.0):
    """Scene for a condition name at time t [s] (people may move)."""
    s = base.copy()
    cond = condition.upper()
    mark = base.slots.get("person_mark", [0.45, 0.45, 0.0])
    n_people = 0
    if cond in ("A1",) or cond[0] in "BP":
        s.add(person(mark)); n_people = 1
    if cond in ("A2", "A3"):
        s.add(person(mark))
        for xy in EXTRA_PEOPLE:
            s.add(person(_sc(base, xy)))
        span = WALK["x1"] - WALK["x0"]
        u = (WALK["speed"] * t) % (2 * span)
        x = WALK["x0"] + (u if u < span else 2 * span - u)
        s.add(person(_sc(base, (x, WALK["y"])), name="walker"))
        n_people = 2 + len(EXTRA_PEOPLE)
    if cond == "A3":
        s.add(clutter(base))
    return s, n_people


def fog_for(condition, base, alpha=None, plume_alpha=None, **kw):
    cond = condition.upper()
    a = B_ALPHA.get(cond, 0.0) if alpha is None else alpha
    pa = P_ALPHA.get(cond, 0.0) if plume_alpha is None else plume_alpha
    src = base.slots.get("fog_source", {}).get("position", [1.35, 1.60, 0.30])
    return Fog(alpha=a, plume_alpha=pa, plume_center=tuple(src), **kw)


def generate(condition, n_frames, seed=0, scene_path=None, alpha=None, plume_alpha=None,
             fog_kw=None, sensor_kw=None, backend="numpy", start_frame=0, debug=True):
    """Yield (frame dict, label dict) for n_frames consecutive 100 ms frames."""
    base = Scene.load(scene_path)
    sp = SensorParams(**(sensor_kw or {}))
    fog = fog_for(condition, base, alpha, plume_alpha, **(fog_kw or {}))
    rng = np.random.default_rng(seed)
    for i in range(start_frame, start_frame + n_frames):
        t0 = time.perf_counter()
        sc, n_people = build_scene(base, condition, t=i / sp.frame_hz)
        fr = simulate_frame(sc, i, rng, sp, fog, backend, debug=debug)
        lab = dict(frame=i, n_people=n_people, fog_alpha=fog.alpha, plume_alpha=fog.plume_alpha,
                   n_points=int(fr["xyz"].shape[0]), gen_ms=(time.perf_counter() - t0) * 1e3)
        if debug:
            lab.update(n_fog_points=fr["n_fog"], n_dropped=fr["n_dropped"])
        yield fr, lab


def write_bank(out, condition, n_frames, seed=0, **kw):
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)
    labels = []
    t0 = time.perf_counter()
    for k, (fr, lab) in enumerate(generate(condition, n_frames, seed, **kw)):
        np.savez(os.path.join(out, "frame_%05d.npz" % k), xyz=fr["xyz"], intensity=fr["intensity"],
                 tag=fr["tag"])
        labels.append(lab)
    base = Scene.load(kw.get("scene_path"))
    fog = fog_for(condition, base, kw.get("alpha"), kw.get("plume_alpha"), **(kw.get("fog_kw") or {}))
    man = dict(generator="synthlab.lidar_sim.bank", sensor="Livox Mid-360 (simulated)",
               condition=condition, seed=seed, n_frames=n_frames, frame_s=0.1,
               scene=os.path.basename(base.path), backend=kw.get("backend", "numpy"),
               fog=fog.describe(), sensor_params=sensor_describe(SensorParams(**(kw.get("sensor_kw") or {}))),
               frame_convention="xyz in sensor frame: x forward, y left, z up [m]",
               tag_bits="bits 0-1 glue, bits 2-3 particle (rain/fog/dust); 0 normal, 1 medium, 2 low conf",
               gen_ms_per_frame_mean=float(np.mean([l["gen_ms"] for l in labels])),
               gen_wall_s=time.perf_counter() - t0, frames=labels)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def load_frames(bank_dir):
    bank_dir = os.path.expanduser(bank_dir)
    names = sorted(n for n in os.listdir(bank_dir) if n.startswith("frame_") and n.endswith(".npz"))
    return [dict(np.load(os.path.join(bank_dir, n))) for n in names]


def windows(frames, per_window=10):
    """Concatenate consecutive frames into accumulation windows (default 1 s)."""
    out = []
    for s in range(0, len(frames) - per_window + 1, per_window):
        ch = frames[s:s + per_window]
        out.append({k: np.concatenate([f[k] for f in ch]) for k in ("xyz", "intensity", "tag")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, help="A0-A3, B0-B3, P1-P3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=5, help="number of 1 s windows (10 frames each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--alpha", type=float, default=None, help="override uniform extinction [1/m]")
    ap.add_argument("--plume-alpha", type=float, default=None)
    ap.add_argument("--return-mode", default=None, choices=["strongest", "first"])
    ap.add_argument("--xi-r2", type=float, default=None)
    ap.add_argument("--tau-h", type=float, default=None)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    a = ap.parse_args()
    fk = {k: v for k, v in dict(return_mode=a.return_mode, xi_r2=a.xi_r2, tau_h=a.tau_h).items()
          if v is not None}
    man = write_bank(a.out, a.condition, a.windows * 10, a.seed, scene_path=a.scene, alpha=a.alpha,
                     plume_alpha=a.plume_alpha, fog_kw=fk, backend=a.backend)
    print("bank %s: %d frames, %.1f ms/frame, mean %.0f pts/frame, fog MOR %.1f m" % (
        a.out, man["n_frames"], man["gen_ms_per_frame_mean"],
        np.mean([l["n_points"] for l in man["frames"]]), alpha_to_mor(man["fog"]["alpha"])))


if __name__ == "__main__":
    main()
