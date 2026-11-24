"""Build baseline and control banks on the Jetson (real A0 frames, flat, noise, people)."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from . import scenes

REAL = ("~/thermal-camera-radiometric/jetson_experiment_A_thermal_yolo_handoff/"
        "{run}/thermal_raw/*.npy")
REAL_RUNS = (  # formal heat-lamp batch: lamp off = empty room (A0), lamp on = A1
    ("experiment_A_heat_lamp/lamp_off/EXP-A_A0_run01", "a0_bank"),
    ("experiment_A_heat_lamp/lamp_on/EXP-A_A1_run01", "a1_bank"),
    ("experiment_A_radiometric/EXP-A_A0_run03", "a0_early_session"),
    ("data/experiment_A_warmup60_capture120/EXP-A_A1_run01", "a1_two_det"),
)


def main(out="~/synthlab/sponge-search"):
    out = os.path.expanduser(out)
    os.makedirs(out + "/real_bg", exist_ok=True)
    os.makedirs(out + "/controls", exist_ok=True)
    for run, name in REAL_RUNS:
        files = sorted(glob.glob(os.path.expanduser(REAL.format(run=run))))
        if files:
            pick = files[100:100 + 8 * 40:40]
            np.save(f"{out}/real_bg/{name}.npy", np.stack([np.load(f) for f in pick]))
    rng = np.random.default_rng(0)
    flat = np.repeat(scenes.celsius_to_raw(np.full((1, 120, 160), 27.5)), 8, 0)
    np.save(out + "/controls/flat.npy", flat)
    np.save(out + "/controls/zeros_u8.npy", np.repeat(scenes.u8_to_raw(np.zeros((1, 120, 160))), 8, 0))
    np.save(out + "/controls/white_u8.npy",
            np.repeat(scenes.u8_to_raw(np.full((1, 120, 160), 255)), 8, 0))
    np.save(out + "/controls/noise_u8.npy", scenes.u8_to_raw(rng.integers(0, 256, (8, 120, 160))))
    np.save(out + "/controls/empty_synth.npy", scenes.empty_room(8, 0)[0])
    x = np.zeros(scenes.DIM_LAYOUT)
    x[:6] = [0.8, 0.3, 0.2, 0.1, 0.0, 0.3]
    for j, (px, sz) in enumerate([(0.3, 0.75), (0.65, 0.6)]):
        x[18 + 7 * j: 25 + 7 * j] = [1, 0.0, px, 0.55, sz, 0.5, 0.5]
    np.save(out + "/controls/two_people.npy", scenes.render_layout(x, 8, 0)[0])


if __name__ == "__main__":
    main(*sys.argv[1:])
