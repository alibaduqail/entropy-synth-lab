"""Fit the session models and extract the layer library from the BUILD trials only.

    python -m synthlab.thermal_datadriven.build_models

Writes small models to results/thermal-datadriven/models/ (git) and the person-layer library
(real imagery of people) to data_local/thermal_datadriven/layers/ (not in git).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import dataio, layers, sensor_model
from .split import BUILD, KEYS

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "results" / "thermal-datadriven" / "models"
LAYER_DIR = REPO / "data_local" / "thermal_datadriven" / "layers"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LAYER_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    models = {}
    for key in ("hl_off", "rj_A0", "w120_A0"):
        sess, name, _ = KEYS[key]
        stacks = [dataio.load_stack(sess, t) for t in BUILD[key]]
        m = sensor_model.fit_session(name, stacks)
        m.save(MODEL_DIR / f"session_{name}.npz")
        models[name] = m
        summary[name] = m.stats
        print(name, json.dumps(m.stats)[:300], flush=True)
    # lamp-on background of the hl session (person always present) and heated-surface map
    on_stacks = [dataio.load_stack("hl_on", t) for t in BUILD["hl_on"]]
    hl_on_bg = layers.occupied_background(on_stacks, models["hl"].bg, t_pct=5, trial_pct=10)
    surface = np.clip(hl_on_bg - models["hl"].bg, 0, None)
    from scipy import ndimage as ndi
    surface = ndi.gaussian_filter(surface, 1.0)
    surface = np.where(surface > 40.0, surface, 0.0).astype(np.float32)   # > 0.4 K
    np.savez_compressed(MODEL_DIR / "lamp_surface_hl.npz", hl_on_bg=hl_on_bg, surface=surface)
    summary["lamp_surface_hl"] = {"pixels_gt_0p4K": int((surface > 0).sum()),
                                  "max_K": float(surface.max() * 0.01),
                                  "mean_K_over_frame": float(surface.mean() * 0.01)}
    lib = {}
    for key in ("hl_on", "rj_A1", "w120_A1"):
        sess, name, npeople = KEYS[key]
        bg = hl_on_bg if key == "hl_on" else models[name].bg
        for t in BUILD[key]:
            st = dataio.load_stack(sess, t)
            ex = layers.extract_trial(st, bg, max_people=npeople)
            metas = [[layers.layer_meta(ex["alpha"][i], ex["label"][i], ex["lamp"][i], k)
                      for k in range(1, npeople + 1)] for i in range(len(st))]
            tag = f"{key}__{t}"
            np.savez_compressed(LAYER_DIR / f"{tag}.npz", alpha=ex["alpha"], fg=ex["fg"],
                                label=ex["label"], lamp=ex["lamp"], offset=ex["offset"],
                                src_bg=bg, meta=np.array(json.dumps(metas)))
            pres = [np.mean([m[k] is not None for m in metas]) for k in range(npeople)]
            lib[tag] = {"frames": len(st), "people": npeople, "presence": [float(p) for p in pres],
                        "lamp_frames": int(sum(any(m and m["lamp_px"] > 0 for m in ms) for ms in metas))}
            print(tag, lib[tag], flush=True)
    summary["layer_library"] = lib
    (MODEL_DIR / "build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
