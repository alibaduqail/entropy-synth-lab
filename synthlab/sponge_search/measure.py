"""Interleaved repeated measurement of named frame banks (runs ON the Jetson, in the lock).

  python3 -m synthlab.sponge_search.measure --out res.json --rounds 20 --k 60 \
      base=~/x/a0_bank.npy sponge=~/y/frames.npy [--fps 8.7] [--variant max_det=50]

Each round visits every bank once in a fresh random order, so slow drift (temperature,
DVFS) affects all banks alike. Listing the same file twice gives an A/A noise-floor test.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_overrides(text):
    out = {}
    for item in filter(None, text.split(",")):
        k, v = item.split("=")
        out[k] = (v == "True") if v in ("True", "False") else (float(v) if "." in v else int(v))
    return out


def main():
    from .pipeline import Rig

    ap = argparse.ArgumentParser()
    ap.add_argument("banks", nargs="+", help="name=path.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--variant", default="", help="predict overrides, e.g. max_det=50,conf=0.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=16)
    args = ap.parse_args()

    banks = {}
    for item in args.banks:
        name, path = item.split("=", 1)
        if not os.path.exists(os.path.expanduser(path)):
            print("skip missing bank", name, path, flush=True)
            continue
        banks[name] = np.load(os.path.expanduser(path))[: args.max_frames]
    overrides = parse_overrides(args.variant)
    rig = Rig()
    rng = np.random.default_rng(args.seed)
    rows = []
    names = list(banks)
    for r in range(args.rounds):
        for j in rng.permutation(len(names)):
            name = names[j]
            m = rig.measure(banks[name], k=args.k, fps=args.fps or None, overrides=overrides,
                            with_info=(r == 0))
            m.update(bank=name, round=r)
            rows.append(m)
        print("round", r, " ".join(
            f"{x['bank']}={x['frame_ms']:.2f}ms/{x['vdd_in_mw']:.0f}mW" for x in rows[-len(names):]),
            flush=True)
    rig.close()
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "expa": rig.expa_path, "rows": rows}, open(out, "w"))


if __name__ == "__main__":
    main()
