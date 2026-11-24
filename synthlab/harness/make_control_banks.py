#!/usr/bin/env python3
"""Build the jetson-harness control banks on the Jetson (run with PYTHONDONTWRITEBYTECODE=1).

  real_*        recorded Lepton frames (read-only sources), evenly sampled: the anchor
  mvp_objects_N the earlier project's composite generator (N pasted warm figures)
  mvp_noise_S   same generator, one figure, added noise sigma S
  ctl_flat      constant 22 C;  ctl_noise_S  22 C + Gaussian noise of S kelvin

  python3 make_control_banks.py --out ~/synthlab/jetson-harness/banks [--frames 96]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HANDOFF = Path("/home/jetsonnano/thermal-camera-radiometric/jetson_experiment_A_thermal_yolo_handoff")
MVP = Path("/home/jetsonnano/thermal-synth-mvp")
REAL = {"real_radiometric_A0": ("experiment_A_radiometric", "EXP-A_A0_run*"),
        "real_radiometric_A1": ("experiment_A_radiometric", "EXP-A_A1_run*"),
        "real_heatlamp_off": ("experiment_A_heat_lamp/lamp_off", "EXP-A_A0_run*"),
        "real_heatlamp_on": ("experiment_A_heat_lamp/lamp_on", "EXP-A_A1_run*")}


def save(out, name, bank, manifest):
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    assert bank.dtype == np.uint16 and bank.shape[1:] == (120, 160), (bank.dtype, bank.shape)
    np.save(folder / "frames.npy", bank)
    manifest = {"n_frames": int(len(bank)), **manifest}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(name, bank.shape, f"mean {bank.mean() * 0.01 - 273.15:.2f} C", flush=True)


def real_bank(subdir, pattern, frames, max_runs=4):
    runs = sorted((HANDOFF / subdir).glob(pattern))[:max_runs]
    per_run = frames // len(runs)
    picked, sources = [], []
    for run in runs:
        files = sorted((run / "thermal_raw").glob("*.npy"))
        for i in np.linspace(0, len(files) - 1, per_run).round().astype(int):
            picked.append(np.load(files[i], allow_pickle=False))
            sources.append(f"{run.name}/{files[i].name}")
    return np.stack(picked).astype(np.uint16), sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1711)
    args = ap.parse_args()
    out = args.out.expanduser()
    for name, (subdir, pattern) in REAL.items():
        bank, sources = real_bank(subdir, pattern, args.frames)
        save(out, name, bank, {"generator": "recorded Lepton 3.5 frames (real, not synthetic)",
                               "kind": "real", "source_dir": str(HANDOFF / subdir),
                               "source_frames": sources})
    rng = np.random.default_rng(args.seed)
    flat = np.full((args.frames, 120, 160), 29515, np.uint16)
    save(out, "ctl_flat", flat, {"generator": "constant 22.00 C", "kind": "control"})
    for sigma in (0.5, 2.0, 8.0):
        noisy = np.rint(29515 + rng.normal(0, sigma * 100, flat.shape)).astype(np.uint16)
        save(out, f"ctl_noise_{sigma:g}K", noisy,
             {"generator": "22 C + iid Gaussian noise", "kind": "control", "sigma_k": sigma,
              "seed": args.seed})
    sys.path.insert(0, str(MVP / "scripts"))
    from PIL import Image
    import synthetic_experiment as se
    source = json.loads((MVP / "jetson_inputs" / "inputs.json").read_text())
    backgrounds = [np.asarray(Image.open(MVP / "jetson_inputs" / p).convert("L"))
                   for p in source["backgrounds"]]
    layers = [np.asarray(Image.open(MVP / "jetson_inputs" / p).convert("RGBA"))
              for p in source["layers"]]
    specs = [(f"mvp_objects_{n:02d}", n, se.BASE_NOISE) for n in se.OBJECT_LEVELS]
    specs += [(f"mvp_noise_{s:02d}", 1, float(s)) for s in se.NOISE_LEVELS if s > 0]
    for name, people, noise in specs:
        bank = np.stack([se.synthetic_counts(se.make_level_sample(
            backgrounds, layers, args.seed, i, people, noise)) for i in range(args.frames)])
        save(out, name, bank, {"generator": "earlier project composite generator "
                               "(thermal-synth-mvp synthetic_experiment.make_level_sample)",
                               "kind": "mvp-composite", "n_objects": people,
                               "noise_sigma_gray": noise, "seed": args.seed,
                               "n_people": [people] * args.frames})


if __name__ == "__main__":
    main()
