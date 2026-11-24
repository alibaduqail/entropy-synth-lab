#!/usr/bin/env python3
"""Write tiny made-up banks for self-testing the harness (not research data).

  python3 make_test_banks.py <out_dir>   ->  <out_dir>/thermal_flat, thermal_blobs,
                                             lidar_room, lidar_room_fog
"""

import json
import sys
from pathlib import Path

import numpy as np


def thermal_bank(n=24, blobs=0, seed=0):
    rng = np.random.default_rng(seed)
    kelvin = 295.15 + rng.normal(0, 0.05, (n, 120, 160))
    yy, xx = np.mgrid[:120, :160]
    for i in range(n):
        for _ in range(blobs):
            cy, cx = rng.uniform(20, 100), rng.uniform(20, 140)
            kelvin[i] += 12.0 * np.exp(-(((yy - cy) / 18) ** 2 + ((xx - cx) / 7) ** 2))
    return np.rint(kelvin / 0.01).astype(np.uint16)


def lidar_frames(n=20, points=20000, fog=0.0, seed=0):
    """A 6 x 5 x 2.6 m box room seen from a sensor 1 m above the floor, plus optional
    uniformly scattered 'fog' returns flagged in tag bits 2-3."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        az = rng.uniform(-np.pi, np.pi, points)
        el = np.radians(rng.uniform(-7, 52, points))
        d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
        lo, hi = np.array([-3.0, -2.5, -1.0]), np.array([3.0, 2.5, 1.6])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0, hi / d, lo / d)
        r = np.nanmin(t, axis=1) + rng.normal(0, 0.01, points)
        tag = np.zeros(points, np.uint8)
        if fog > 0:
            hit = rng.random(points) < fog
            r = np.where(hit, rng.uniform(0.3, 1.0, points) * r, r)
            tag[hit] = 1 << 2
        frames.append(((d * r[:, None]).astype(np.float32), tag))
    return frames


def write(out):
    out = Path(out)
    for name, blobs in [("thermal_flat", 0), ("thermal_blobs", 3)]:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "frames.npy", thermal_bank(blobs=blobs, seed=1))
        (folder / "manifest.json").write_text(json.dumps(
            {"generator": "harness self-test (made-up)", "seed": 1, "n_frames": 24,
             "blobs": blobs}))
    for name, fog in [("lidar_room", 0.0), ("lidar_room_fog", 0.3)]:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, (xyz, tag) in enumerate(lidar_frames(fog=fog, seed=2)):
            np.savez(folder / f"frame_{i:05d}.npz", xyz=xyz, tag=tag,
                     intensity=np.full(len(xyz), 100, np.uint8))
        (folder / "manifest.json").write_text(json.dumps(
            {"generator": "harness self-test (made-up)", "seed": 2, "n_frames": 20, "fog": fog}))


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "selftest_banks")
