"""Hardware-in-the-loop black-box search (runs ON the Jetson, inside the measurement lock).

Every candidate scene is rendered, pushed through the real pipeline by `pipeline.Rig`, and
scored on measured cost. State (CMA-ES object) is pickled each generation so a search can
be resumed across <= 20 minute lock chunks. Every evaluation is appended to evals.jsonl.

Modes (all maximised):
  latency : mean ms/frame minus the same-generation empty-room baseline
  energy  : mJ/frame (VDD_IN x time) minus baseline
  stealth : energy, penalised when HT leaves the real empty-room band
  decouple: HT, penalised when latency rises above baseline + tol
  cand    : deterministic proxy - pre-NMS candidate count (no timing; fast)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from . import scenes

HT_BAND = (0.011, 0.020)  # real radiometric A0 (empty room) frame-level HT range on this rig


def render(tier, x, n_frames, seed, background=None):
    if tier == "layout":
        return scenes.render_layout(x, n_frames, seed)
    if tier == "layout_real":
        return scenes.render_layout(x, n_frames, seed, background=background)
    if tier == "panel":
        return scenes.render_panel(x, n_frames, seed)
    if tier == "blocks":  # unconstrained low-res 8-bit image, nearest-upsampled
        img = np.kron(np.rint(np.clip(x, 0, 1).reshape(15, 20) * 255), np.ones((8, 8)))
        raw = scenes.u8_to_raw(img.astype(np.uint8))
        return np.repeat(raw[None], n_frames, 0), {"tier": "blocks"}
    raise ValueError(tier)


def dim_of(tier):
    return {"layout": scenes.DIM_LAYOUT, "layout_real": scenes.DIM_LAYOUT,
            "panel": scenes.DIM_PANEL, "blocks": 300}[tier]


def score(mode, m, base, lam_ht=5.0, lam_ms=50.0, tol_ms=0.4, shape=0.0):
    d_ms = m.get("frame_ms", 0.0) - base.get("frame_ms", 0.0)
    d_mj = m.get("energy_mj", 0.0) - base.get("energy_mj", 0.0)
    ht = m["ht"]
    viol = (max(0.0, ht - HT_BAND[1]) + max(0.0, HT_BAND[0] - ht)) / 0.001
    bonus = shape * m.get("n_cand", 0.0)
    if mode == "latency":
        return d_ms + bonus
    if mode == "energy":
        return d_mj + bonus
    if mode == "stealth":
        return d_mj - lam_ht * viol + bonus
    if mode == "decouple":
        return ht / 0.001 - lam_ms * max(0.0, d_ms - tol_ms)
    if mode == "cand":
        return m["n_cand"] + 0.01 * m.get("n_cand10", 0.0)
    if mode == "cand_stealth":
        return m["n_cand"] + 0.01 * m.get("n_cand10", 0.0) - 20.0 * viol
    raise ValueError(mode)


def main():
    import cma
    from .pipeline import Rig

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory (state + evals.jsonl)")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--tier", default="layout")
    ap.add_argument("--budget-s", type=float, default=780)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--bank", type=int, default=4)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shape", type=float, default=0.0,
                    help="tie-breaker added to the objective per pre-NMS candidate")
    ap.add_argument("--x0", default="", help="npy file with a start vector")
    ap.add_argument("--baseline", default="~/synthlab/sponge-search/real_bg/a0_bank.npy")
    ap.add_argument("--diag", action="store_true", help="sep-CMA (for high dimensions)")
    args = ap.parse_args()

    t_end = time.time() + args.budget_s
    run = os.path.expanduser(args.run)
    os.makedirs(run, exist_ok=True)
    state_p = os.path.join(run, "cma.pkl")
    base_raw = np.load(os.path.expanduser(args.baseline))
    background = scenes.raw_to_celsius(base_raw).mean(0)
    proxy = args.mode.startswith("cand")
    if os.path.exists(state_p):
        with open(state_p, "rb") as fh:
            es, gen, best = pickle.load(fh)
    else:
        d = dim_of(args.tier)
        x0 = np.load(os.path.expanduser(args.x0)) if args.x0 else np.full(d, 0.5)
        opts = {"bounds": [0, 1], "popsize": args.popsize, "seed": args.seed, "verbose": -9}
        if args.diag:
            opts["CMA_diagonal"] = True
        es = cma.CMAEvolutionStrategy(x0, args.sigma, opts)
        gen, best = 0, {"f": -1e9}
        json.dump(vars(args), open(os.path.join(run, "config.json"), "w"), indent=1)
    rig = Rig()
    log = open(os.path.join(run, "evals.jsonl"), "a")
    while time.time() < t_end:
        base = {"ht": 0.0} if proxy else rig.measure(base_raw, k=args.k)
        X = es.ask()
        fs = []
        for i, x in enumerate(X):
            frames, _ = render(args.tier, x, args.bank, seed=1000 + i, background=background)
            m = rig.info(frames) if proxy else rig.measure(frames, k=args.k)
            f = score(args.mode, m, base, shape=args.shape)
            fs.append(f)
            row = {"run": os.path.basename(run), "mode": args.mode, "tier": args.tier,
                   "gen": gen, "i": i, "f": f, "m": m,
                   "base": {k: base.get(k) for k in ("frame_ms", "energy_mj", "vdd_in_mw")},
                   "x": [round(float(v), 4) for v in x]}
            log.write(json.dumps(row) + "\n")
            if f > best["f"]:
                best = {"f": f, "gen": gen, "x": np.array(x), "m": m}
                np.save(os.path.join(run, "best_x.npy"), np.array(x))
        log.flush()
        es.tell(X, [-f for f in fs])
        gen += 1
        with open(state_p, "wb") as fh:
            pickle.dump((es, gen, best), fh)
        print(f"gen {gen} best_gen {max(fs):.3f} mean {np.mean(fs):.3f} best_all {best['f']:.3f} "
              f"det {best['m'].get('n_det')} cand {best['m'].get('n_cand')} "
              f"ht {best['m'].get('ht'):.4f} base_ms {base.get('frame_ms', 0):.2f}", flush=True)
    rig.close()


if __name__ == "__main__":
    main()
