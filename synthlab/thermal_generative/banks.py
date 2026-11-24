"""Generate the full set of hand-off banks (resumable, time-boxed; run under `jetson.sh heavy`).

    python3 -m synthlab.thermal_generative.banks --ddpm ddpm_main --vae vae_main --minutes 16

Banks (N frames each, ~/synthlab/thermal-generative/banks/<name>/):
  tg_real_*      held-out REAL frames, same format, as the reference for the harness
  tg_ddpm_*      DDPM samples for each seen condition and for unseen label combinations
  tg_sweep_w*    guidance-scale sweep, tg_sweep_t* noise-temperature sweep (person+lamp, heat_lamp session)
  tg_vae_*       CVAE baseline
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from .data import Pack
from .mapping import ht_of_counts
from .sample import HOME, write_bank

S = dict(steps=50, guidance=1.0, temperature=1.0)
# name -> (person, lamp, session, overrides)
DDPM_BANKS = [
    ("tg_ddpm_A0_lampoff_s0", 0, 0, 0, {}), ("tg_ddpm_A1_lampon_s0", 1, 1, 0, {}),
    ("tg_ddpm_A0_s1", 0, 0, 1, {}), ("tg_ddpm_A1_2people_s1", 1, 0, 1, {}),
    ("tg_ddpm_A0_s2", 0, 0, 2, {}), ("tg_ddpm_A1_s2", 1, 0, 2, {}),
    ("tg_ddpm_UNSEEN_A0_lampon_s0", 0, 1, 0, {}), ("tg_ddpm_UNSEEN_A1_lampoff_s0", 1, 0, 0, {}),
] + [(f"tg_sweep_w{w:g}", 1, 1, 0, dict(guidance=w)) for w in (0.0, 0.5, 2.0, 4.0, 7.0)] \
  + [(f"tg_sweep_t{t:g}", 1, 1, 0, dict(temperature=t)) for t in (0.8, 0.9, 1.1, 1.2)]
VAE_BANKS = [("tg_vae_A0_lampoff_s0", 0, 0, 0, {}), ("tg_vae_A1_lampon_s0", 1, 1, 0, {}),
             ("tg_vae_A1_2people_s1", 1, 0, 1, {})]
REAL_BANKS = [("tg_real_A0_lampoff_s0", 0, 0, 0), ("tg_real_A1_lampon_s0", 1, 1, 0),
              ("tg_real_A0_s1", 0, 0, 1), ("tg_real_A1_2people_s1", 1, 0, 1),
              ("tg_real_A0_s2", 0, 0, 2), ("tg_real_A1_s2", 1, 0, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm", default="ddpm_main")
    ap.add_argument("--vae", default="vae_main")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--minutes", type=float, default=16.0)
    ap.add_argument("--only", default="", help="comma-separated substrings; only generate matching banks")
    a = ap.parse_args()
    keep = lambda n: (not a.only) or any(k in n for k in a.only.split(","))
    t_end = time.time() + a.minutes * 60
    pack = Pack(f"{HOME}/data/pack.npz")
    for name, p, l, s in REAL_BANKS:
        if not keep(name):
            continue
        if os.path.exists(f"{HOME}/banks/{name}/manifest.json"):
            continue
        idx = np.where(pack.mask("test", p, l, s))[0]
        idx = np.sort(np.random.default_rng(0).choice(idx, min(a.n, len(idx)), replace=False))
        fr = pack.frames[idx]
        ht = [ht_of_counts(f) for f in fr]
        write_bank(name, fr, dict(generator="REAL held-out Lepton 3.5 recordings (reference, not synthetic)",
                                  parameters=dict(person=p, lamp=l, session=s, split="test"), seed=0, n_frames=int(len(fr)),
                                  ht_mean=float(np.mean(ht)), ht_sd=float(np.std(ht)),
                                  labels=dict(person=[p] * len(fr), n_people_nominal=[(2 if s == 1 else 1) * p] * len(fr),
                                              lamp_on=[l] * len(fr), session=[s] * len(fr), ht=[float(h) for h in ht],
                                              source_trial=[pack.trials[pack.trial[i]]["name"] for i in idx],
                                              source_frame_id=[int(pack.frame_id[i]) for i in idx])))
        print("real bank", name, len(fr), flush=True)
    jobs = [(a.ddpm, b) for b in DDPM_BANKS] + [(a.vae, b) for b in VAE_BANKS]
    for ck, (name, p, l, s, over) in jobs:
        if not keep(name):
            continue
        if os.path.exists(f"{HOME}/banks/{name}/manifest.json") or not os.path.exists(f"{HOME}/ckpt/{ck}.pt"):
            continue
        if time.time() > t_end:
            print("time budget reached; rerun to continue")
            return
        kw = dict(S, **over)
        cmd = [sys.executable, "-m", "synthlab.thermal_generative.sample", "--ckpt", ck, "--bank", name, "--person", str(p),
               "--lamp", str(l), "--session", str(s), "--n", str(a.n), "--steps", str(kw["steps"]),
               "--guidance", str(kw["guidance"]), "--temperature", str(kw["temperature"]), "--seed", "11"]
        subprocess.run(cmd, check=False)
    print("banks complete")


if __name__ == "__main__":
    main()
