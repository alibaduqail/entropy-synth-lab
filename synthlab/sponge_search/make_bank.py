"""Export a search result as a hand-off thermal bank (frames.npy + manifest.json).

  python3 -m synthlab.sponge_search.make_bank --run <run_dir> --name <bank> [--n 96]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import scenes
from .search import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--root", default="~/synthlab/sponge-search/banks")
    ap.add_argument("--baseline", default="~/synthlab/sponge-search/real_bg/a0_bank.npy")
    args = ap.parse_args()
    run = os.path.expanduser(args.run)
    cfg = json.load(open(os.path.join(run, "config.json")))
    x = np.load(os.path.join(run, "best_x.npy"))
    background = scenes.raw_to_celsius(np.load(os.path.expanduser(args.baseline))).mean(0)
    frames, scene = render(cfg["tier"], x, args.n, seed=4242, background=background)
    out = os.path.join(os.path.expanduser(args.root), args.name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "frames.npy"), frames)
    np.save(os.path.join(out, "x.npy"), x)
    manifest = {"generator": "synthlab.sponge_search (CMA-ES hardware-in-the-loop)",
                "parameters": {"mode": cfg["mode"], "tier": cfg["tier"], "run": run,
                               "scene": scene}, "seed": 4242, "n_frames": int(len(frames)),
                "description": scenes.describe(scene) if scene.get("tier") == "layout" else []}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(out, frames.shape)


if __name__ == "__main__":
    main()
