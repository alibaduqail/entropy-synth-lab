"""Write the hand-off frame banks (local copy in data_local/, then rsync to the Jetson).

    python -m synthlab.thermal_datadriven.make_banks [--out DIR]

Synthetic banks use ONLY the BUILD-split layers/background (split.py); the real_* banks are
HELD-OUT real trials (split.TEST), straight from the recordings (read-only), 128 consecutive frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .bank import real_bank, write_bank
from .compose import Generator, SceneSpec
from .split import TEST

REPO = Path(__file__).resolve().parents[2]
N = 128
HL = dict(session="hl", n_frames=N)
# A1 = one person holding the lit lamp (as the real A1 of the heat-lamp session).
A1 = dict(HL, n_people=1, lamp=1.0, people_hold_lamp=True, layer_tags=("hl_on",))

SPECS = {
    "syn_A0": dict(HL),
    "syn_A1": dict(A1),
    "syn_A2": dict(HL, n_people=2, lamp=1.0, n_warm_objects=1),
    "syn_A3": dict(HL, n_people=4, lamp=1.0, n_warm_objects=3, warm_object_intensity=0.8),
}
for k in range(9):                                 # people sweep: no lamp, clothes-warm people only
    SPECS[f"people_{k}"] = dict(HL, n_people=k)
for w in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):                # noise sweep on the A1 scene
    SPECS[f"noise_{w:g}x".replace(".", "p")] = dict(A1, noise_mult=w)
for q in (0.0, 0.5, 1.0, 1.5):                     # lamp strength sweep, one person (HT lever)
    SPECS[f"lamp_{q:g}".replace(".", "p")] = dict(HL, n_people=1, lamp=q) if q > 0 else dict(HL, n_people=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data_local" / "thermal_datadriven" / "banks"))
    out = Path(ap.parse_args().out)
    gen = Generator()
    rows = []
    for i, (name, kw) in enumerate(SPECS.items()):
        seed = 500 if name.startswith('noise_') else 100 + i      # noise sweep: identical scene, only noise differs
        fr, man = gen.generate(SceneSpec(seed=seed, **kw))
        man["bank_name"] = name
        m = write_bank(out / name, fr, man)
        rows.append((name, m["HT_mean"], m["temp_max_c"]))
        print(f"{name:14s} HT={m['HT_mean']:.3f} max={m['temp_max_c']:.1f}C", flush=True)
    for cond, key in (("A0", "hl_off"), ("A1", "hl_on")):
        for j, trial in enumerate(TEST[key][:3]):
            fr, man = real_bank("hl_off" if cond == "A0" else "hl_on", trial, N, seed=j)
            m = write_bank(out / f"real_{cond}_{j}", fr, man)
            print(f"real_{cond}_{j:<7d} HT={m['HT_mean']:.3f} max={m['temp_max_c']:.1f}C", flush=True)


if __name__ == "__main__":
    main()
