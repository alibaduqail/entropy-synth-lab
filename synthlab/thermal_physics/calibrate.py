"""Measure the real Lepton 3.5 recordings (read-only) -> results/thermal-physics/real_calibration.json.

Usage: python -m synthlab.thermal_physics.calibrate [--data-root DIR] [--out FILE]
"""
import argparse
import glob
import json
import os

import numpy as np

from . import measure

GROUPS = {
    # name: (glob under data root, flat scene?, max runs)
    "warmup_A0_flat_wall": ("experiment_A_warmup60_capture120/EXP-A_A0_run*", True, 6),
    "lamp_off_A0": ("experiment_A_heat_lamp/lamp_off/EXP-A_A0_run*", False, 10),
    "radiometric_A0": ("experiment_A_radiometric_jetson/EXP-A_A0_run*", False, 3),
    "warmup_A1_two_seated": ("experiment_A_warmup60_capture120/EXP-A_A1_run*", False, 5),
    "lamp_on_A1": ("experiment_A_heat_lamp/lamp_on/EXP-A_A1_run*", False, 10),
    "radiometric_A1": ("experiment_A_radiometric_jetson/EXP-A_A1_run*", False, 3),
}


def _median_of(dicts, key_path):
    vals = []
    for d in dicts:
        v = d
        for k in key_path:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            vals.append(v)
    return (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)) if vals else None


def run(data_root, out_path, hist_dir=None):
    res = {"data_root": data_root, "groups": {}}
    for name, (pattern, flat, max_runs) in GROUPS.items():
        dirs = [d for d in sorted(glob.glob(os.path.join(data_root, pattern)))
                if glob.glob(os.path.join(d, "thermal_raw", "raw_*.npy"))][:max_runs]
        per_run, hists, psf = {}, [], []
        static = "A0" in name
        for d in dirs:
            x, t, _ = measure.load_run(d)
            s = measure.summarize(x, t, flat=flat, ht_step=2)
            if not static:      # moving people: frame-difference noise numbers are not sensor noise
                s["temporal"] = None
                s["drift"] = None
                e = [measure.edge_psf_sigma(x[i]) for i in range(0, len(x), 40)]
                e = [q for q in e if q]
                if e:
                    s["edge_psf"] = dict(sigma_px_p10=float(np.median([q["sigma_px_p10"] for q in e])),
                                         sigma_px_median=float(np.median([q["sigma_px_median"] for q in e])))
                    psf.append(s["edge_psf"])
            else:
                e = measure.edge_psf_sigma(np.asarray(x, np.float64).mean(0))
                if e:
                    s["edge_psf"] = e
                    psf.append(e)
            h, _ = np.histogram(x.astype(np.float64) / 100 - 273.15, bins=np.arange(10, 60.01, 0.1))
            hists.append(h)
            per_run[os.path.basename(d)] = s
            print(name, os.path.basename(d), "HT %.4f" % s["ht"]["mean"], "n_ffc", s["n_ffc"], flush=True)
        runs = list(per_run.values())
        agg = {"n_runs": len(runs)}
        for label, kp in [("nedt_total_sd", ("temporal", "total_sd")), ("pixel_temporal_sd", ("temporal", "pixel_sd")),
                          ("row_temporal_sd", ("temporal", "row_sd")), ("col_temporal_sd", ("temporal", "col_sd")),
                          ("frame_flicker_sd", ("temporal", "frame_flicker_sd")), ("ar1_rho", ("temporal", "ar1_rho")),
                          ("neigh_corr_h", ("temporal", "neigh_corr_h")), ("neigh_corr_v", ("temporal", "neigh_corr_v")),
                          ("drift_abs_counts_per_min", ("drift", "slope_counts_per_min")),
                          ("fpn_col_sd", ("fpn", "col_sd")), ("fpn_row_sd", ("fpn", "row_sd")),
                          ("fpn_pixel_sd", ("fpn", "pixel_sd")), ("fpn_lowfreq_sd", ("fpn", "lowfreq_sd")),
                          ("fpn_lowfreq_p2p", ("fpn", "lowfreq_p2p")),
                          ("ht_mean", ("ht", "mean")), ("ht_within_run_sd", ("ht", "sd")),
                          ("mean_c", ("mean_c",)), ("max_c", ("max_c",)), ("spatial_sd_frame", ("spatial_sd_frame",)),
                          ("edge_psf_sigma_p10", ("edge_psf", "sigma_px_p10")),
                          ("fps_effective", ("fps_effective",))]:
            if label == "drift_abs_counts_per_min":
                vals = [abs(r["drift"]["slope_counts_per_min"]) for r in runs if r.get("drift")]
                agg[label] = (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)) if vals else None
            else:
                agg[label] = _median_of(runs, kp)
        ffc = [f for r in runs for f in r["ffc"]]
        agg["ffc_events"] = ffc
        agg["ffc_per_min"] = float(len(ffc) / (sum(r["duration_s"] for r in runs) / 60.0)) if runs else None
        chg = [c for r in runs for c in (r.get("fpn_change_across_ffc") or []) if c]
        agg["fpn_change_across_ffc"] = chg
        agg["_format"] = "aggregates are [median, min, max, n_runs] in TLinear counts (0.01 K) unless named otherwise"
        res["groups"][name] = dict(aggregate=agg, runs=per_run)
        if hist_dir:
            np.savez_compressed(os.path.join(hist_dir, f"real_hist_{name}.npz"), hist=np.sum(hists, 0),
                                edges_c=np.arange(10, 60.01, 0.1))
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=measure.DATA_ROOT)
    ap.add_argument("--out", default="results/thermal-physics/real_calibration.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    r = run(a.data_root, a.out, hist_dir=os.path.dirname(a.out))
    for g, v in r["groups"].items():
        print("==", g)
        for k, q in v["aggregate"].items():
            if k not in ("ffc_events", "_format"):
                print("  ", k, q)
        for f in v["aggregate"]["ffc_events"]:
            print("   ffc", f)
