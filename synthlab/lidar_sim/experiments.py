"""Entropy-side experiments (no timing claims): conditions, fog sweep, model and HL sensitivity.

  python -m synthlab.lidar_sim.experiments conditions|sweep|variants|sensitivity|coverage|all

Each writes a CSV (one row per 1 s window) under results/lidar-sim/. Seeds are in every row.
"""
import csv
import json
import os
import sys

import numpy as np

from .bank import generate, windows
from .entropy import HLParams, compute_hl
from .scene import REPO
from .sensor import SensorParams, coverage_vs_time

OUT = os.path.join(REPO, "results", "lidar-sim")
SWEEP_ALPHA = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00]
KEEP = ("n_raw", "n_roi", "n_voxel", "voxel_retention", "frac_tag_particle", "frac_tag_glue", "H_DS",
        "H_eigen", "mean_L", "mean_P", "mean_S", "frac_high_S", "mean_neighbors", "mean_neighbors_all",
        "sum_neighbors", "frac_capped")


def write_csv(name, rows):
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", name, len(rows), "rows")


def run_windows(condition, n_windows, seed, hp=None, per_window=10, **gen_kw):
    """Generate n_windows consecutive windows and compute HL for each."""
    frames, labels = [], []
    for fr, lab in generate(condition, n_windows * per_window, seed, **gen_kw):
        frames.append(fr)
        labels.append(lab)
    rows = []
    for wi, w in enumerate(windows(frames, per_window)):
        r = compute_hl(w["xyz"], w["tag"], hp, timing=False)
        labs = labels[wi * per_window:(wi + 1) * per_window]
        row = dict(condition=condition, seed=seed, window=wi, fog_alpha=labs[0]["fog_alpha"],
                   plume_alpha=labs[0]["plume_alpha"], n_people=labs[0]["n_people"],
                   n_fog_points=sum(l["n_fog_points"] for l in labs),
                   n_emitted=per_window * 20000)
        row.update({k: r.get(k, "") for k in KEEP})
        rows.append(row)
    return rows, frames


def exp_conditions(n_windows=5):
    rows = []
    for c in ("A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "P1", "P2", "P3"):
        rows += run_windows(c, n_windows, seed=100)[0]
    write_csv("conditions.csv", rows)


def exp_sweep(n_windows=3):
    rows = []
    for a in SWEEP_ALPHA:
        rows += run_windows("A1", n_windows, seed=200, alpha=a)[0]
    write_csv("fog_sweep.csv", rows)


def exp_variants(n_windows=2):
    """Same sweep under the uncertain model choices: near-field overlap, pulse length, return rule."""
    variants = {
        "default(xi 0.1-0.5, tau 5ns, strongest)": {},
        "near-field open (xi 0.05-0.2)": dict(xi_r1=0.05, xi_r2=0.2),
        "Hahner/Velodyne overlap (xi 0.9-1.0)": dict(xi_r1=0.9, xi_r2=1.0),
        "pulse tau_H 2 ns": dict(tau_h=2e-9),
        "pulse tau_H 20 ns (Hahner)": dict(tau_h=20e-9),
        "first-return rule": dict(return_mode="first"),
        "no near-field gate (rho_gate 0)": dict(rho_gate=0.0),
        "strong near-field gate (rho_gate 0.01)": dict(rho_gate=0.01),
        "room_v0 geometry (closed room, no door/glass)": "v0",
    }
    v0 = os.path.join(REPO, "scene", "room_v0.json")
    rows = []
    for name, fk in variants.items():
        for a in SWEEP_ALPHA:
            kw = dict(scene_path=v0) if fk == "v0" else dict(fog_kw=fk)
            rr = run_windows("A1", n_windows, seed=300, alpha=a, **kw)[0]
            for r in rr:
                r["variant"] = name
            rows += rr
    write_csv("fog_variants.csv", rows)


def exp_sensitivity():
    """HL vs window length, voxel size and radius at three fog levels (same frames reused)."""
    rows = []
    for a in (0.0, 0.3, 1.0):
        frames = [fr for fr, _ in generate("A1", 40, seed=400, alpha=a)]
        grid = [("window_s", w, HLParams(), int(round(w * 10))) for w in (0.1, 0.2, 0.5, 1.0, 1.1, 2.0)]
        grid += [("leaf_m", v, HLParams(leaf=v), 10) for v in (0.02, 0.03, 0.05, 0.08, 0.10)]
        grid += [("radius_m", r, HLParams(radius=r), 10) for r in (0.10, 0.15, 0.20, 0.30, 0.40)]
        for pname, pval, hp, per in grid:
            for wi, w in enumerate(windows(frames, per)[:4]):
                r = compute_hl(w["xyz"], w["tag"], hp, timing=False)
                row = dict(fog_alpha=a, param=pname, value=pval, window=wi, seed=400)
                row.update({k: r.get(k, "") for k in KEEP})
                rows.append(row)
    write_csv("hl_sensitivity.csv", rows)


def exp_coverage():
    sp = SensorParams()
    cov = coverage_vs_time(sp, (0.1, 0.2, 0.5, 1.0, 2.0, 4.0))
    npts = {}
    frames = [fr for fr, _ in generate("A0", 20, seed=500)]
    for per in (1, 2, 5, 10, 20):
        w = windows(frames, per)[0]
        r = compute_hl(w["xyz"], w["tag"], timing=False)
        npts[per / 10] = dict(n_raw=r["n_raw"], n_voxel=r["n_voxel"])
    with open(os.path.join(OUT, "scan_coverage.json"), "w") as f:
        json.dump(dict(cell_deg=0.5, coverage_fraction=cov, points_vs_window_A0=npts,
                       note="analytic approximation of the Mid-360 pattern; not the recorded CSV"), f, indent=1)
    print(cov, npts)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(conditions=exp_conditions, sweep=exp_sweep, variants=exp_variants,
               sensitivity=exp_sensitivity, coverage=exp_coverage)
    for k, fn in fns.items():
        if which in (k, "all"):
            fn()


if __name__ == "__main__":
    main()
