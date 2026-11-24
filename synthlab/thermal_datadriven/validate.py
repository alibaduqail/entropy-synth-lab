"""Held-out validation: synthetic vs held-out real, with real-build vs held-out real as the floor.

    python -m synthlab.thermal_datadriven.validate [--yolo PATH/yolov8n.pt]

Writes results/thermal-datadriven/validation/metrics.json (+ per-clip CSV) from the sets made by
valsets.py.  For every metric the same comparison is made twice:
    synth      vs real_test   (what we want to be small)
    real_build vs real_test   (how much two groups of REAL trials differ anyway)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .valsets import CONDS, OUT as SETS

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "thermal-datadriven" / "validation"
KINDS = ("real_test", "real_build", "synth")
EDGES_C = np.arange(15.0, 60.0001, 0.1)


def load(cond, kind):
    z = np.load(SETS / f"{cond}__{kind}.npz")
    return z["frames"], z["group"]


def clip_metrics(frames):
    c = frames.astype(np.float32) * 0.01 - 273.15
    hts = M.ht_stack(frames, 4)
    comp = M.block_image_entropy(M.radiometric_bie_image(frames[len(frames) // 2]))
    out = {"HT": float(hts.mean()), "HT_sd_within": float(hts.std()),
           "bie_global_sd": comp["global_sd"], "bie_mean_block_sd": comp["mean_block_sd"],
           "bie_mean_block_entropy": comp["mean_block_entropy"],
           "bie_mean_modulation_entropy": comp["mean_modulation_entropy"],
           "temp_mean_c": float(c.mean()), "temp_spatial_sd_c": float(c.std(axis=(1, 2)).mean()),
           "temp_p99_c": float(np.percentile(c, 99)), "temp_max_c": float(c.max())}
    out.update(M.temporal_noise_stats(frames))
    return out


def w1(a_hist, b_hist, edges):
    """Wasserstein-1 distance between two histograms on the same bins (in bin units of edges)."""
    ca, cb = np.cumsum(a_hist) / a_hist.sum(), np.cumsum(b_hist) / b_hist.sum()
    return float(np.sum(np.abs(ca - cb)) * (edges[1] - edges[0]))


def set_summary(frames, group):
    per = [clip_metrics(frames[group == g]) for g in np.unique(group)]
    keys = per[0].keys()
    c = frames[::3].astype(np.float32) * 0.01 - 273.15
    hist_abs = np.histogram(c, EDGES_C)[0]
    rel = c - np.median(c, axis=(1, 2), keepdims=True)
    hist_rel = np.histogram(rel, np.arange(-5, 30.0001, 0.05))[0]
    f, p = M.radial_power_spectrum(c)
    return {"n_clips": len(per), "n_frames": int(len(frames)),
            "mean": {k: float(np.mean([q[k] for q in per])) for k in keys},
            "sd": {k: float(np.std([q[k] for q in per])) for k in keys},
            "median": {k: float(np.median([q[k] for q in per])) for k in keys}}, per, \
        {"hist_abs": hist_abs, "hist_rel": hist_rel, "psd_f": f, "psd": p}


def run_yolo(model, frames, step=4):
    rows = []
    for f in frames[::step]:
        g = M.radiometric_bie_image(f)
        r = model.predict(source=np.repeat(g[:, :, None], 3, axis=2), imgsz=640, conf=0.25,
                          verbose=False, device="cpu")[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        pc = conf[cls == 0]
        rows.append((len(cls), int((cls == 0).sum()), float(conf.max()) if len(conf) else np.nan,
                     float(pc.max()) if len(pc) else np.nan))
    a = np.array(rows, float)
    return {"n_frames": len(a), "mean_detections": float(a[:, 0].mean()),
            "mean_person_detections": float(a[:, 1].mean()),
            "person_frame_rate": float((a[:, 1] > 0).mean()),
            "any_detection_rate": float((a[:, 0] > 0).mean()),
            "mean_max_conf": float(np.nanmean(a[:, 2])) if np.isfinite(a[:, 2]).any() else None,
            "mean_person_conf": float(np.nanmean(a[:, 3])) if np.isfinite(a[:, 3]).any() else None,
            "det_count_hist": np.bincount(a[:, 0].astype(int), minlength=6)[:10].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="")
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    model = None
    if args.yolo:
        from ultralytics import YOLO
        model = YOLO(args.yolo)
    out, curves = {}, {}
    for cond in args.conds.split(","):
        out[cond] = {}
        dist = {}
        for kind in KINDS:
            frames, group = load(cond, kind)
            summ, per, dist[kind] = set_summary(frames, group)
            if model is not None:
                summ["yolo_mac_cpu"] = run_yolo(model, frames)
            out[cond][kind] = summ
            curves[f"{cond}__{kind}__psd"] = dist[kind]["psd"]
            curves[f"{cond}__{kind}__hist_rel"] = dist[kind]["hist_rel"]
            curves[f"{cond}__{kind}__hist_abs"] = dist[kind]["hist_abs"]
            curves[f"{cond}__{kind}__HT"] = np.array([q["HT"] for q in per])
            curves[f"{cond}__{kind}__tsd"] = np.array([q["pixel_temporal_sd_k"] for q in per])
        curves["psd_f"] = dist["synth"]["psd_f"]
        cmp_ = {}
        for other in ("synth", "real_build"):
            a, b = dist[other], dist["real_test"]
            ok = np.isfinite(a["psd"]) & np.isfinite(b["psd"]) & (b["psd"] > 0)
            cmp_[other + "_vs_real_test"] = {
                "pixel_hist_W1_abs_K": w1(a["hist_abs"], b["hist_abs"], EDGES_C),
                "pixel_hist_W1_median_removed_K": w1(a["hist_rel"], b["hist_rel"], np.array([0, 0.05])),
                "radial_psd_rms_log10_ratio": float(np.sqrt(np.mean(np.log10(a["psd"][ok] / b["psd"][ok]) ** 2))),
                "radial_psd_highfreq_log10_ratio": float(np.mean(np.log10(a["psd"][ok][-10:] / b["psd"][ok][-10:]))),
                "HT_mean_ratio": out[cond][other]["mean"]["HT"] / out[cond]["real_test"]["mean"]["HT"],
            }
        out[cond]["comparison"] = cmp_
        print(cond, json.dumps(cmp_), flush=True)
        (RES / "metrics.json").write_text(json.dumps(out, indent=1))
        np.savez_compressed(RES / "curves.npz", **curves)


if __name__ == "__main__":
    main()
