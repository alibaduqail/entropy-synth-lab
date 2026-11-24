"""How flat is HT in fog? Sweep LWIR extinction and mist-air temperature offset.

  python -m synthlab.thermal_physics.fog_sweep --out results/thermal-physics/fog_sweep.json

Hypothesis B assumes thermal entropy HT stays flat in fog. Physics says fog pulls every pixel
toward the mist temperature (L = tau L_scene + (1-tau) B(T_mist)), so contrast - and HT - must
fall once tau drops, and a mist-air temperature offset (evaporative cooling) adds a depth-
dependent gradient that can push HT the other way in an empty room.
"""
import argparse
import json

import numpy as np

from . import bie
from .scenarios import FOG_BETA, K_LWIR_M2_PER_G, Scenario
from .sensor import Lepton35, SensorParams

BETAS = [0.0, 0.035, 0.07, 0.14, 0.21, 0.35, 0.56, 1.0, 2.0]
OFFSETS = [0.0, -0.5, -1.0, -2.0]


def measure_config(scene_name, beta, offset, plume, seed, n_frames, n_renders, scene_path=None):
    spec = dict(seed=seed, fog_beta_per_m=beta, mist_offset_k=offset, fog_turbulence=0.25 if beta > 0 else 0.0)
    if scene_name == "person":
        spec.update(n_people=1, poses=["standing"], person_at_mark=True)
    if plume:
        spec.update(fog_plume=True, fog_turbulence=0.5)
    scn = Scenario(spec, room_path=scene_path)
    cam = Lepton35(SensorParams(seed=seed), first_ffc_s=1e12)
    hts, sds, means, rad, tau = [], [], [], None, 1.0
    per = max(1, n_frames // n_renders)
    for i in range(n_frames):
        if i % per == 0:
            out = scn.render(cam.t)
            rad = scn.renderer.bk.to_numpy(out["radiance"])
            tau = out.get("tau_mean", 1.0)
        f, _ = cam.capture(rad)
        hts.append(bie.ht_of_raw(f))
        sds.append(float(f.std()) / 100.0)
        means.append(float(f.mean()) / 100.0 - 273.15)
    return dict(scene=scene_name, beta_per_m=beta, lwc_g_m3=beta / K_LWIR_M2_PER_G, mist_offset_k=offset, plume=bool(plume),
                tau_mean=float(tau), ht_mean=float(np.mean(hts)), ht_sd=float(np.std(hts)), n_frames=n_frames,
                frame_sd_k=float(np.mean(sds)), mean_c=float(np.mean(means)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/thermal-physics/fog_sweep.json")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--renders", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scene", default=None)
    a = ap.parse_args(argv)
    rows = []
    for scene_name in ("person", "empty"):
        for off in OFFSETS:
            for beta in BETAS:
                rows.append(measure_config(scene_name, beta, off, False, a.seed, a.n, a.renders, a.scene))
                print(json.dumps(rows[-1]), flush=True)
        for beta in BETAS[1:]:
            rows.append(measure_config(scene_name, beta, -1.0, True, a.seed, a.n, a.renders, a.scene))
            print(json.dumps(rows[-1]), flush=True)
    with open(a.out, "w") as f:
        json.dump(dict(betas=BETAS, offsets_k=OFFSETS, fog_levels={str(k): v for k, v in FOG_BETA.items()},
                       k_lwir_m2_per_g=K_LWIR_M2_PER_G, seed=a.seed, rows=rows,
                       note="simulated; scene = newest room JSON; sensor noise on; HT via experimentA BIE"), f, indent=1)


if __name__ == "__main__":
    main()
