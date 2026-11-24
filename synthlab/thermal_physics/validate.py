"""Real-vs-simulated comparison for matched conditions.

  python -m synthlab.thermal_physics.validate            # writes results/thermal-physics/validation.json
                                                         # + data_local/thermal-physics/validation/*.npy stacks

The real recordings are the ground truth for appearance. Their views do not correspond to
scene/room_v*.json (different framing / distance), so each real group is matched with a
hand-set "view preset": an open room at the measured background temperature with people at
a distance estimated from head width in the real frames. Preset parameters are NOT fitted to
the comparison metrics (only air temperature, distance and pose were set by eye, once).
"""
import argparse
import glob
import json
import os

import numpy as np

from . import bie, measure
from .scenarios import generate_bank


def open_room(air_c, cam_to_wall=3.0, cam_height=1.12, tilt=0.0):
    surf = lambda e: dict(emissivity=e, temp_c=air_c)                      # noqa: E731
    return dict(units="m", up="z", room=dict(size_xyz=[6.0, cam_to_wall + 2.5, 3.0], air_temp_c=air_c, surfaces=dict(
        floor=surf(0.95), ceiling=surf(0.92), wall_north=surf(0.92), wall_south=surf(0.92),
        wall_west=surf(0.92), wall_east=surf(0.92))), objects=[], dynamic_slots={},
        sensors=dict(lepton35=dict(position=[3.0, cam_to_wall, cam_height], look_dir=[0.0, -1.0, tilt],
                                   hfov_deg=57.0, width=160, height=120, fps=8.7)), _path="open_room(preset)")


def presets():
    """name -> (real run glob, room dict or None (= scene JSON), spec)."""
    y = 3.0
    return {
        "flat_wall_A0": ("experiment_A_warmup60_capture120/EXP-A_A0_run0[23]", open_room(21.75, 1.5),
                         dict(n_people=0, first_ffc_s=30.0, seed=11)),
        "empty_room_A0": ("experiment_A_heat_lamp/lamp_off/EXP-A_A0_run0[12]", None,
                          dict(n_people=0, air_c=24.8, first_ffc_s=1e12, seed=12)),
        "two_seated_A1": ("experiment_A_warmup60_capture120/EXP-A_A1_run0[12]", open_room(22.0, y),
                          dict(n_people=2, poses=["sitting"], person_xy=[(3.19, y - 0.80), (2.81, y - 0.84)],
                               first_ffc_s=1e12, seed=13)),
        "standing_person_A1": ("experiment_A_radiometric_jetson/EXP-A_A1_run0[12]", open_room(26.0, y + 0.5, tilt=-0.08),
                               dict(n_people=1, poses=["walking"], person_depth=(1.6, 2.4), first_ffc_s=1e12, seed=14)),
        "person_holding_lamp_A1": ("experiment_A_heat_lamp/lamp_on/EXP-A_A1_run0[25]", open_room(24.3, y + 0.5, tilt=-0.05),
                                   dict(n_people=1, poses=["arms_raised"], person_xy=[(3.0, y + 0.5 - 2.4)], lamp_held=True,
                                        lamp_temp_c=95.0, first_ffc_s=1e12, seed=15)),
    }


def wasserstein1(a, b, n=2000):
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def mean_radial_psd(x, step):
    acc = None
    for f in x[::step]:
        fr, p = measure.radial_psd(f)
        acc = p if acc is None else acc + p
    return fr, acc / len(x[::step])


def compare(real, sim, t_real, t_sim, static):
    rc, sc_ = real.astype(np.float64) / 100 - 273.15, sim.astype(np.float64) / 100 - 273.15
    step_r, step_s = max(1, len(real) // 60), max(1, len(sim) // 60)
    fr, pr = mean_radial_psd(real, step_r)
    _, ps = mean_radial_psd(sim, step_s)
    ok = np.isfinite(pr) & np.isfinite(ps) & (pr > 0) & (ps > 0)
    ht_r, ht_s = bie.ht_series(real[::max(1, len(real) // 150)]), bie.ht_series(sim[::max(1, len(sim) // 150)])
    sub_r, sub_s = rc[::step_r].ravel(), sc_[::step_s].ravel()
    out = dict(
        n_real=int(len(real)), n_sim=int(len(sim)),
        mean_c=dict(real=float(rc.mean()), sim=float(sc_.mean())),
        max_c=dict(real=float(np.percentile(rc.max((1, 2)), 50)), sim=float(np.percentile(sc_.max((1, 2)), 50))),
        frame_spatial_sd_k=dict(real=float(np.median(rc.std((1, 2)))), sim=float(np.median(sc_.std((1, 2))))),
        hist_w1_k=wasserstein1(sub_r, sub_s),
        hist_w1_mean_removed_k=wasserstein1(sub_r - sub_r.mean(), sub_s - sub_s.mean()),
        radial_psd_rms_log10_ratio=float(np.sqrt(np.mean(np.log10(ps[ok] / pr[ok]) ** 2))),
        radial_psd_log10_ratio_low_mid_high=[float(np.mean(np.log10(ps[ok][s] / pr[ok][s])))
                                             for s in (slice(0, 5), slice(5, 15), slice(15, 30))],
        ht=dict(real_mean=float(ht_r.mean()), real_sd=float(ht_r.std()), sim_mean=float(ht_s.mean()),
                sim_sd=float(ht_s.std()), ratio=float(ht_s.mean() / ht_r.mean())),
    )
    curves = dict(freq=fr, psd_real=pr, psd_sim=ps, ht_real=ht_r, ht_sim=ht_s,
                  hist_real=np.histogram(sub_r, bins=np.arange(15, 60, 0.1))[0],
                  hist_sim=np.histogram(sub_s, bins=np.arange(15, 60, 0.1))[0])
    if static:
        out["sensor_real"] = measure.summarize(real, t_real, flat=True, ht_step=8)
        out["sensor_sim"] = measure.summarize(sim, t_sim, flat=True, ht_step=8)
        out["frac_pixels_unchanged"] = dict(real=float((np.diff(real.astype(int), axis=0) == 0).mean()),
                                            sim=float((np.diff(sim.astype(int), axis=0) == 0).mean()))
        for nm, (x, t) in (("real", (real, t_real)), ("sim", (sim, t_sim))):
            f, p = measure.temporal_psd(x, t, measure.find_ffc(t))
            if f is not None:
                curves[f"tfreq_{nm}"], curves[f"tpsd_{nm}"] = f, p
    return out, curves


def run(data_root, out_dir, stack_dir, n_sim=None, only=None):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stack_dir, exist_ok=True)
    results = {}
    for name, (pattern, room, spec) in presets().items():
        if only and name not in only:
            continue
        dirs = sorted(glob.glob(os.path.join(data_root, pattern)))
        xs, ts = zip(*[measure.load_run(d)[:2] for d in dirs])
        real, t_real = xs[0], ts[0]                      # first run for time-resolved stats
        static = spec["n_people"] == 0
        n = n_sim or (len(real) if static else 256)
        sim, man = generate_bank(spec, n_frames=n, room=room)
        t_sim = np.array([f["t"] for f in man["frames"]])
        res, curves = compare(real, sim, t_real, t_sim, static)
        if len(xs) > 1:                                  # run-to-run spread of the real data itself
            ht2 = bie.ht_series(xs[1][::max(1, len(xs[1]) // 150)])
            res["ht"]["real_second_run_mean"] = float(ht2.mean())
            res["real_vs_real_hist_w1_k"] = wasserstein1((xs[0][::20].astype(float) / 100 - 273.15).ravel(),
                                                         (xs[1][::20].astype(float) / 100 - 273.15).ravel())
        res["real_runs"] = [os.path.relpath(d, data_root) for d in dirs]
        res["sim_spec"] = spec
        res["sim_room"] = "scene JSON (newest)" if room is None else "open_room preset"
        results[name] = res
        np.savez_compressed(os.path.join(out_dir, f"validation_curves_{name}.npz"), **curves)
        np.save(os.path.join(stack_dir, f"sim_{name}.npy"), sim)
        np.save(os.path.join(stack_dir, f"real_{name}.npy"), real)
        with open(os.path.join(stack_dir, f"sim_{name}.manifest.json"), "w") as f:
            json.dump(man, f, default=float)
        print(name, json.dumps({k: res[k] for k in ("mean_c", "max_c", "frame_spatial_sd_k", "hist_w1_k",
                                                     "hist_w1_mean_removed_k", "radial_psd_rms_log10_ratio", "ht")},
                               default=float), flush=True)
    path = os.path.join(out_dir, "validation.json")
    if only and os.path.exists(path):
        old = json.load(open(path))
        old.update(results)
        results = old
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=float)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=measure.DATA_ROOT)
    ap.add_argument("--out", default="results/thermal-physics")
    ap.add_argument("--stacks", default="data_local/thermal-physics/validation")
    ap.add_argument("--n-sim", type=int, default=None)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(a.data_root, a.out, a.stacks, a.n_sim, a.only)
