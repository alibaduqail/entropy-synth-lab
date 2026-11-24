"""Build the validation frame sets: held-out REAL clips, BUILD-trial real clips (the
real-vs-real floor) and SYNTHETIC clips made by a generator that never saw the held-out trials.

    python -m synthlab.thermal_datadriven.valsets

Each set is saved to data_local/thermal_datadriven/valsets/<cond>__<kind>.npz with
frames (N,120,160) uint16 and group (N,) = clip id (one real trial segment or one synthetic seed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import dataio
from .compose import Generator, SceneSpec
from .split import BUILD, KEYS, TEST

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data_local" / "thermal_datadriven" / "valsets"
CLIP = 64
# condition -> (split key, synthetic spec kwargs, clips per real trial)
CONDS = {
    "hl_A0": ("hl_off", dict(session="hl"), 2),
    "hl_A1lamp": ("hl_on", dict(session="hl", n_people=1, lamp=1.0, people_hold_lamp=True,
                                layer_tags=("hl_on",)), 2),
    "rj_A0": ("rj_A0", dict(session="rj"), 6),
    "rj_A1": ("rj_A1", dict(session="rj", n_people=1, layer_tags=("rj_A1",)), 6),
    "w120_A0": ("w120_A0", dict(session="w120"), 6),
    "w120_A2": ("w120_A1", dict(session="w120", n_people=2, layer_tags=("w120_A1",)), 5),
    # cross-session transplant: people cut from the w120 session pasted into the rj room
    "rj_A1_transplant": ("rj_A1", dict(session="rj", n_people=1, layer_tags=("w120_A1",)), 6),
}


def real_clips(key, trials, per_trial, rng):
    sess = KEYS[key][0]
    frames, group, src = [], [], []
    for t in trials:
        st = dataio.load_stack(sess, t)
        starts = np.linspace(0, len(st) - CLIP, per_trial).astype(int)
        for s in starts:
            frames.append(st[s:s + CLIP])
            group.append(np.full(CLIP, len(src)))
            src.append(f"{t}@{s}")
    return np.concatenate(frames), np.concatenate(group), src


def build_trial_ambient_sd(key):
    """Between-trial SD (K) of the whole-frame mean among BUILD trials of a session's A0."""
    sess = KEYS[key][0]
    means = [dataio.load_stack(sess, t)[::20].mean() * 0.01 for t in BUILD[key]]
    return float(np.std(means))


def main(seed=1234):
    OUT.mkdir(parents=True, exist_ok=True)
    gen = Generator()
    rng = np.random.default_rng(seed)
    amb_sd = {"hl": build_trial_ambient_sd("hl_off"), "rj": build_trial_ambient_sd("rj_A0"),
              "w120": build_trial_ambient_sd("w120_A0")}
    print("ambient sd (K) from build trials", amb_sd, flush=True)
    for cond, (key, kw, per_trial) in CONDS.items():
        fr, gr, src = real_clips(key, TEST[key], per_trial, rng)
        np.savez_compressed(OUT / f"{cond}__real_test.npz", frames=fr, group=gr, src=np.array(src))
        n_clips = len(src)
        per_b = max(1, int(np.ceil(n_clips / len(BUILD[key]))))
        fb, gb, srcb = real_clips(key, BUILD[key], per_b, rng)
        np.savez_compressed(OUT / f"{cond}__real_build.npz", frames=fb, group=gb, src=np.array(srcb))
        frames, group, seeds = [], [], []
        for c in range(n_clips):
            sd = int(seed * 1000 + c + 17 * len(cond))
            shift = float(rng.normal(0, amb_sd[kw["session"]]))
            f, _ = gen.generate(SceneSpec(n_frames=CLIP, seed=sd, ambient_shift_c=shift, **kw))
            frames.append(f); group.append(np.full(CLIP, c)); seeds.append(sd)
        np.savez_compressed(OUT / f"{cond}__synth.npz", frames=np.concatenate(frames),
                            group=np.concatenate(group), src=np.array(seeds))
        print(cond, "real_test clips", n_clips, "real_build clips", len(srcb), flush=True)


if __name__ == "__main__":
    main()
