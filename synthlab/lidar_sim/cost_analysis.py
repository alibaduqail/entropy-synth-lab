"""Summarise Jetson harness runs of the lidar-sim banks: cost vs entropy vs point count.

  python -m synthlab.lidar_sim.cost_analysis <campaign> [<campaign2> ...]

Reads data_local/runs/<campaign>/p<pass>_<bank>/{windows.csv,summary.json,bank_manifest.json},
writes results/lidar-sim/cost_runs_<campaign>.csv (one row per harness run),
cost_stats_<campaign>.json (correlations, partial correlations, regressions) and figures.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import C, OUT, save
from .scene import REPO

STAGES = ["roi_ms", "voxel_ms", "knn_ms", "eig_ms", "entropy_total_ms", "cl_pairs_ms",
          "cluster_total_ms", "total_ms"]
VALS = ["H_DS", "H_eigen", "mean_S", "frac_high_S", "n_raw", "n_roi", "n_voxel", "n_pairs",
        "n_clusters", "n_person_candidates", "frac_tag_particle", "mean_neighbors"]


def group_of(bank):
    if bank.startswith("thin_"):
        return "thinned control"
    if bank.startswith("fog_") or bank in ("B1", "B2", "B3"):
        return "uniform fog"
    if bank.startswith("P"):
        return "plume"
    return "clear scenes (A0-A3)"


def load(campaign):
    root = os.path.join(REPO, "data_local", "runs", campaign)
    runs = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.exists(os.path.join(p, "summary.json")):
            continue
        s = json.load(open(os.path.join(p, "summary.json")))
        man = json.load(open(os.path.join(p, "bank_manifest.json"))) if os.path.exists(
            os.path.join(p, "bank_manifest.json")) else {}
        with open(os.path.join(p, "windows.csv")) as f:
            w = list(csv.DictReader(f))
        bank = s["bank_name"]
        row = dict(campaign=campaign, run=d, bank=bank, group=group_of(bank),
                   pass_idx=int(d[1:d.index("_")]), mode=s["mode"], workload=s["workload"],
                   fog_alpha=man.get("fog", {}).get("alpha", 0.0),
                   plume_alpha=man.get("fog", {}).get("plume_alpha", 0.0),
                   thin=man.get("thin", 1.0), condition=man.get("condition", ""),
                   windows=len(w), overrun_windows=s.get("overrun_windows"),
                   duty_cycle=s.get("duty_cycle"), power_w=s.get("power_w"),
                   idle_power_w=s.get("idle_power_w"), energy_per_window_j=s.get("energy_per_window_j"),
                   energy_above_idle_per_window_j=s.get("energy_above_idle_per_window_j"),
                   temp_cpu_c=s.get("temp_cpu_c"), ok=s.get("ok"))
        for k in STAGES + VALS:
            x = np.array([float(r[k]) for r in w if r.get(k) not in (None, "", "nan")])
            row[k] = float(x.mean()) if x.size else float("nan")
            if k in ("entropy_total_ms", "cluster_total_ms", "total_ms"):
                row[k + "_sd"] = float(x.std()) if x.size else float("nan")
        runs.append(row)
    return runs


def pcorr(x, y, z):
    """Partial correlation of x and y controlling for z (all 1-D)."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def ols(y, cols):
    """Standardised OLS: returns R^2 and beta per column."""
    X = np.column_stack([(c - c.mean()) / c.std() for c in cols] + [np.ones(len(y))])
    ys = (y - y.mean()) / y.std()
    b, *_ = np.linalg.lstsq(X, ys, rcond=None)
    r2 = 1 - ((ys - X @ b) ** 2).sum() / (ys ** 2).sum()
    return float(r2), [float(v) for v in b[:-1]]


def stats(runs, label):
    a = lambda k, rr: np.array([r[k] for r in rr], float)
    out = {"n_runs": len(runs), "subset": label}
    if len(runs) < 6:
        return out
    for y in ("entropy_total_ms", "cluster_total_ms", "total_ms", "energy_above_idle_per_window_j",
              "power_w"):
        yy = a(y, runs)
        if not np.all(np.isfinite(yy)):
            continue
        h, nv, npairs, s = a("H_eigen", runs), a("n_voxel", runs), a("n_pairs", runs), a("mean_S", runs)
        d = {"r_H_eigen": float(np.corrcoef(yy, h)[0, 1]), "r_H_DS": float(np.corrcoef(yy, a("H_DS", runs))[0, 1]),
             "r_n_voxel": float(np.corrcoef(yy, nv)[0, 1]), "r_n_raw": float(np.corrcoef(yy, a("n_raw", runs))[0, 1]),
             "r_n_pairs": float(np.corrcoef(yy, npairs)[0, 1]),
             "partial_r_H_eigen_given_n_voxel": pcorr(yy, h, nv),
             "partial_r_n_voxel_given_H_eigen": pcorr(yy, nv, h)}
        r2a, _ = ols(yy, [nv])
        r2b, beta = ols(yy, [nv, h])
        d.update(R2_n_voxel_only=r2a, R2_n_voxel_plus_H_eigen=r2b, beta_n_voxel=beta[0], beta_H_eigen=beta[1])
        k = np.polyfit(nv, yy, 1)
        d.update(slope_per_1000_voxels=float(k[0] * 1000), intercept=float(k[1]))
        out[y] = d
    out["r_H_eigen_vs_n_voxel"] = float(np.corrcoef(a("H_eigen", runs), a("n_voxel", runs))[0, 1])
    return out


def figures(runs, campaign):
    groups = ["clear scenes (A0-A3)", "uniform fog", "plume", "thinned control"]
    col = dict(zip(groups, [C["blue"], C["orange"], C["aqua"], C["violet"]]))
    mk = dict(zip(groups, "osD^"))
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2))
    for i, (y, ylab) in enumerate([("entropy_total_ms", "HL entropy stage [ms / window]"),
                                   ("cluster_total_ms", "clustering stage [ms / window]")]):
        for j, (x, xlab) in enumerate([("H_eigen", "H_eigen"), ("n_voxel", "occupied 5 cm voxels"),
                                       ("fog_alpha", "extinction alpha [1/m]")]):
            ax = axs[i, j]
            for g in groups:
                rr = [r for r in runs if r["group"] == g]
                if x == "fog_alpha":
                    rr = [r for r in rr if g == "uniform fog" or r["bank"] in ("A1",)]
                if rr:
                    xs = [max(r[x], 0.01) if x == "fog_alpha" else r[x] for r in rr]
                    ax.scatter(xs, [r[y] for r in rr], c=col[g], marker=mk[g], s=26, label=g,
                               edgecolors=C["surface"], linewidths=0.6)
            if x == "fog_alpha":
                ax.set_xscale("log")
                xlab += "  (0.01 = clear)"
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Jetson Orin Nano, shared harness, simulated Mid-360 banks: cost per 1 s window "
                 "(one marker = one %s run)" % runs[0]["mode"], fontsize=10)
    fig.tight_layout()
    save(fig, "fig_cost_vs_entropy_and_voxels_%s.png" % campaign)

    fog = sorted([r for r in runs if r["group"] == "uniform fog" or r["bank"] == "A1"],
                 key=lambda r: r["fog_alpha"])
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))
    xs = [max(r["fog_alpha"], 0.01) for r in fog]
    for ax, (y, lab) in zip(axs, [("total_ms", "total [ms / window]"), ("power_w", "VDD_IN [W]"),
                                  ("energy_above_idle_per_window_j", "energy above idle [J / window]"),
                                  ("n_voxel", "occupied voxels")]):
        ax.scatter(xs, [r[y] for r in fog], c=C["orange"], s=24, edgecolors=C["surface"], linewidths=0.6)
        ax.set_xscale("log"); ax.set_xlabel("extinction alpha [1/m]  (0.01 = clear)"); ax.set_ylabel(lab)
    fig.suptitle("Fog sweep on the Jetson (%s): time, power, energy, voxels" % runs[0]["mode"], fontsize=10)
    fig.tight_layout()
    save(fig, "fig_cost_vs_fog_%s.png" % campaign)


def main():
    for campaign in sys.argv[1:]:
        runs = load(campaign)
        keys = list(runs[0].keys())
        with open(os.path.join(OUT, "cost_runs_%s.csv" % campaign), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(runs)
        nothin = [r for r in runs if r["group"] != "thinned control"]
        fogonly = [r for r in runs if r["group"] == "uniform fog" or r["bank"] == "A1"]
        st = {"all_runs": stats(runs, "all banks"), "without_thinned": stats(nothin, "no thinned controls"),
              "fog_sweep_only": stats(fogonly, "uniform fog sweep + clear A1")}
        with open(os.path.join(OUT, "cost_stats_%s.json" % campaign), "w") as f:
            json.dump(st, f, indent=1)
        print(json.dumps(st, indent=1))
        figures(runs, campaign)


if __name__ == "__main__":
    main()
