"""Generate study-condition frame banks in the hand-off format (AGENTS.md).

  python3 -m synthlab.thermal_physics.make_banks --out ~/synthlab/thermal-physics/banks \\
      [--conditions A0_empty B2_fog2_person ...] [--n 128] [--backend numpy|torch] [--scene room.json]

Each bank: <out>/<name>/frames.npy (uint16 (N,120,160) TLinear) + manifest.json with per-frame labels.
"""
import argparse
import json
import os
import time

import numpy as np

from . import bie
from .scenarios import CONDITIONS, generate_bank


def write_bank(out_dir, name, frames, manifest):
    d = os.path.join(os.path.expanduser(out_dir), name)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "frames.npy"), frames)
    ht = bie.ht_series(frames[:: max(1, len(frames) // 32)])
    manifest = dict(manifest, bank_name=name, ht_mean=float(ht.mean()), ht_sd=float(ht.std()),
                    mean_c=float(frames.mean() / 100 - 273.15), max_c=float(frames.max() / 100 - 273.15))
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=float)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="*", default=None, help="default: all study conditions")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--scene", default=None, help="room JSON path (default: newest room_v*.json)")
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    names = a.conditions or list(CONDITIONS)
    summary = {}
    for name in names:
        spec = dict(CONDITIONS[name], seed=a.seed)
        t0 = time.time()
        frames, man = generate_bank(spec, n_frames=a.n, room_path=a.scene, backend=a.backend, device=a.device,
                                    supersample=a.supersample, noise=not a.no_noise)
        man = write_bank(a.out, a.prefix + name, frames, man)
        summary[name] = dict(ht_mean=man["ht_mean"], mean_c=man["mean_c"], max_c=man["max_c"],
                             render_ms=man["timing"]["render_ms_per_frame"], sensor_ms=man["timing"]["sensor_ms_per_frame"],
                             wall_s=time.time() - t0)
        print(name, json.dumps(summary[name]), flush=True)
    with open(os.path.join(os.path.expanduser(a.out), f"{a.prefix}generation_summary_{a.backend}.json"), "w") as f:
        json.dump(dict(args=vars(a), banks=summary), f, indent=1)


if __name__ == "__main__":
    main()
