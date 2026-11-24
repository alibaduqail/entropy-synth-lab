"""Time the renderer + sensor model (frames/s) on this machine, numpy CPU vs torch CUDA.

  python3 -m synthlab.thermal_physics.bench --out bench.json [--n 40] [--scene room.json]
Run on the Jetson through `tools/jetson.sh measured`.
"""
import argparse
import json
import platform
import time

import numpy as np

from .scenarios import CONDITIONS, Scenario
from .sensor import Lepton35


def time_condition(name, backend, n, scene, supersample, device=None):
    spec = dict(CONDITIONS[name], seed=1)
    t0 = time.perf_counter()
    scn = Scenario(spec, room_path=scene, backend=backend, device=device, supersample=supersample)
    cam = Lepton35(supersample=supersample)
    scn.renderer.bk.sync()
    setup = time.perf_counter() - t0
    for _ in range(3):                                   # warm-up (CUDA kernels, caches)
        out = scn.render(cam.t)
        cam.capture(scn.renderer.bk.to_numpy(out["radiance"]))
    scn.renderer.bk.sync()
    r_ms, s_ms = [], []
    for _ in range(n):
        a = time.perf_counter()
        out = scn.render(cam.t)
        rad = scn.renderer.bk.to_numpy(out["radiance"])  # includes GPU sync + copy
        b = time.perf_counter()
        cam.capture(rad)
        c = time.perf_counter()
        r_ms.append(1e3 * (b - a))
        s_ms.append(1e3 * (c - b))
    tot = np.array(r_ms) + np.array(s_ms)
    return dict(condition=name, backend=backend, device=str(scn.renderer.bk.device), supersample=supersample, n=n,
                setup_s=setup, render_ms_mean=float(np.mean(r_ms)), render_ms_p95=float(np.percentile(r_ms, 95)),
                sensor_ms_mean=float(np.mean(s_ms)), total_ms_mean=float(tot.mean()), fps=float(1e3 / tot.mean()),
                n_dynamic_prims=len(scn.dynamic(0.0) or []), n_static_prims=scn.n_static_prims)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--conditions", nargs="*", default=["A0_empty", "A1_one_person", "A2_three_people",
                                                        "A3_people_warm_objects", "B2_fog2_person", "C16_objects"])
    ap.add_argument("--backends", nargs="*", default=["numpy", "torch"])
    ap.add_argument("--supersample", nargs="*", type=int, default=[2, 1])
    a = ap.parse_args(argv)
    rows = []
    for backend in a.backends:
        if backend == "torch":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("torch present but CUDA unavailable: timing torch on CPU")
            except ImportError:
                print("torch not installed: skipping")
                continue
        for ss in a.supersample:
            for name in a.conditions:
                r = time_condition(name, backend, a.n, a.scene, ss)
                rows.append(r)
                print(json.dumps(r), flush=True)
    with open(a.out, "w") as f:
        json.dump(dict(machine=platform.platform(), processor=platform.machine(), rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
