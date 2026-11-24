"""Figures for docs/lidar-sim.md, all drawn from files in results/lidar-sim/.

  python -m synthlab.lidar_sim.figures clouds|sweep|conditions|variants|sensitivity|scan|all
"""
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bank import generate
from .scene import REPO, Scene
from .sensor import SensorParams, scan_dirs

OUT = os.path.join(REPO, "results", "lidar-sim")
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948", ink="#0b0b0b", ink2="#52514e",
         surface="#fcfcfb", grid="#e4e3df")
ORDER = [C[k] for k in ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")]

plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "lines.linewidth": 2, "lines.markersize": 5, "savefig.dpi": 150,
                     "legend.frameon": False})


def read(name):
    with open(os.path.join(OUT, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def agg(rows, key, val):
    g = defaultdict(list)
    for r in rows:
        if isinstance(r.get(val), float):
            g[r[key]].append(r[val])
    ks = sorted(g)
    return ks, np.array([np.mean(g[k]) for k in ks]), np.array([np.std(g[k]) for k in ks])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_clouds():
    base = Scene.load()
    cases = [("B0 clear", "B0"), ("B1  alpha 0.10 /m (MOR 30 m)", "B1"),
             ("B2  alpha 0.30 /m (MOR 10 m)", "B2"), ("B3  alpha 1.0 /m (MOR 3 m)", "B3"),
             ("P2  plume only (core 3 /m)", "P2")]
    fig, axs = plt.subplots(2, len(cases), figsize=(3.3 * len(cases), 6.2))
    s = base.sensor_pos
    R = base.rot_world_to_sensor()
    for j, (title, cond) in enumerate(cases):
        xyz, isf = [], []
        for fr, _ in generate(cond, 10, seed=7):
            xyz.append(fr["xyz"]); isf.append(fr["is_fog"])
        xyz, isf = np.concatenate(xyz) @ R + s, np.concatenate(isf)       # back to world frame
        sel = np.random.default_rng(0).choice(xyz.shape[0], min(40000, xyz.shape[0]), replace=False)
        p, f = xyz[sel], isf[sel]
        for i, (a, b, lab) in enumerate([(0, 1, ("x [m]", "y [m]")), (0, 2, ("x [m]", "z [m]"))]):
            ax = axs[i, j]
            if i == 0:
                m = p[:, 2] < s[2] + 1.30            # same ceiling cut as the HL ROI
            else:
                m = np.ones(p.shape[0], bool)
            ax.scatter(p[m & ~f, a], p[m & ~f, b], s=0.25, c=C["blue"], linewidths=0, rasterized=True)
            ax.scatter(p[m & f, a], p[m & f, b], s=0.25, c=C["orange"], linewidths=0, rasterized=True)
            ax.plot([s[a]], [s[b]], marker="^", ms=7, c=C["ink"], ls="none")
            lim = (base.size[a], base.size[b])
            ax.plot([0, lim[0], lim[0], 0, 0], [0, 0, lim[1], lim[1], 0], c=C["ink2"], lw=0.8)
            ax.set_xlim(-0.15, lim[0] + 0.15); ax.set_ylim(-0.15, lim[1] + 0.15)
            ax.set_aspect("equal"); ax.grid(False)
            ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1])
            if i == 0:
                ax.set_title("%s\n%d pts/s, %.0f %% fog returns" % (title, xyz.shape[0], 100 * isf.mean()))
    axs[1, 0].scatter([], [], s=12, c=C["blue"], label="surface return")
    axs[1, 0].scatter([], [], s=12, c=C["orange"], label="fog (soft) return")
    axs[1, 0].plot([], [], marker="^", c=C["ink"], ls="none", label="Mid-360")
    axs[1, 0].legend(loc="lower left", fontsize=7, markerscale=1.2)
    fig.suptitle("Simulated Mid-360, 1 s window, room_v1 with one person. Top: plan view (ceiling cut). "
                 "Bottom: side view. SIMULATION ONLY", fontsize=10)
    save(fig, "fig_clouds_clear_vs_fog.png")


def _sweep_panels(rows, axs, color, label, marker="o"):
    panels = [("H_eigen", "H_eigen"), ("H_DS", "H_DS"), ("mean_S", "mean S (scatter)"),
              ("n_voxel", "occupied 5 cm voxels"), ("n_raw", "points per 1 s window"),
              ("frac_tag_particle", "particle-tag fraction")]
    for ax, (k, lab) in zip(axs.ravel(), panels):
        x, m, sd = agg(rows, "fog_alpha", k)
        xs = [max(v, 0.005) for v in x]          # clear air drawn at the left edge of the log axis
        ax.errorbar(xs, m, sd, c=color, marker=marker, label=label, capsize=2, lw=1.6, ms=4)
        ax.set_xscale("log"); ax.set_ylabel(lab)
        ax.set_xlabel("extinction alpha [1/m]  (0.005 = clear air)")


def fig_sweep():
    rows = read("fog_sweep.csv")
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4))
    _sweep_panels(rows, axs, C["blue"], "default model")
    for ax in axs.ravel():
        for a, nm in ((0.10, "B1"), (0.30, "B2"), (1.00, "B3")):
            ax.axvline(a, c=C["ink2"], lw=0.7, ls=":")
            ax.text(a, ax.get_ylim()[1], nm, ha="center", va="bottom", fontsize=7, color=C["ink2"])
    axs[0, 0].axhline(np.log(2), c=C["ink2"], lw=0.7, ls="--")
    axs[0, 0].text(0.006, np.log(2), "ideal plane ln 2", fontsize=7, va="bottom", color=C["ink2"])
    fig.suptitle("HL metrics vs fog density (simulated, person at mark, mean +/- SD of 3 windows). "
                 "MOR = 3.0 / alpha", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_hl_vs_fog.png")


def fig_variants():
    rows = read("fog_variants.csv")
    names = []
    for r in rows:
        if r["variant"] not in names:
            names.append(r["variant"])
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.8))
    for i, nm in enumerate(names):
        _sweep_panels([r for r in rows if r["variant"] == nm], axs, ORDER[i % 8], nm,
                      marker="osD^vP*X"[i % 8])
    axs[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Fog sweep under the uncertain model assumptions (simulated)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_fog_model_variants.png")


def fig_conditions():
    rows = read("conditions.csv")
    conds = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "P1", "P2", "P3"]
    col = [C["blue"]] * 4 + [C["orange"]] * 4 + [C["aqua"]] * 3
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (k, lab) in zip(axs, [("H_eigen", "H_eigen"), ("H_DS", "H_DS"), ("mean_S", "mean S"),
                                   ("n_voxel", "occupied voxels")]):
        m = [np.mean([r[k] for r in rows if r["condition"] == c]) for c in conds]
        sd = [np.std([r[k] for r in rows if r["condition"] == c]) for c in conds]
        ax.bar(conds, m, yerr=sd, color=col, width=0.62, capsize=2)
        lo = min(m) - 0.25 * (max(m) - min(m))
        ax.set_ylim(max(0, lo), None); ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
    axs[0].bar([0], [0], color=C["blue"], label="A: scene complexity")
    axs[0].bar([0], [0], color=C["orange"], label="B: uniform fog")
    axs[0].bar([0], [0], color=C["aqua"], label="P: plume only")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Conditions (simulated, 5 windows each, mean +/- SD; y axes do not start at 0)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_conditions.png")


def fig_sensitivity():
    rows = read("hl_sensitivity.csv")
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for j, (p, lab) in enumerate([("window_s", "window length [s]"), ("leaf_m", "voxel leaf [m]"),
                                  ("radius_m", "kNN radius [m]")]):
        for i, k in enumerate(("H_eigen", "H_DS")):
            ax = axs[i, j]
            for c, a in zip(ORDER, (0.0, 0.3, 1.0)):
                rr = [r for r in rows if r["param"] == p and r["fog_alpha"] == a]
                x, m, sd = agg(rr, "value", k)
                if len(x):
                    ax.errorbar(x, m, sd, c=c, marker="o", capsize=2, label="alpha %.1f /m" % a)
            ax.set_xlabel(lab); ax.set_ylabel(k)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Sensitivity of HL to its method parameters at three fog levels (simulated)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_hl_sensitivity.png")


def fig_scan():
    sp = SensorParams()
    cov = json.load(open(os.path.join(OUT, "scan_coverage.json")))["coverage_fraction"]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.8]))
    for ax, T in zip(axs[:3], (0.1, 0.5, 1.0)):
        d, _ = scan_dirs(sp, 0, int(T * sp.points_per_s))
        az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
        el = np.degrees(np.arcsin(d[:, 2]))
        m = az < 90
        ax.scatter(az[m], el[m], s=0.3, c=C["blue"], linewidths=0, rasterized=True)
        ax.set_title("%.1f s (%d pts), first 90 deg of azimuth" % (T, d.shape[0]))
        ax.set_xlabel("azimuth [deg]"); ax.set_ylabel("elevation [deg]"); ax.grid(False)
    t = sorted(float(k) for k in cov)
    axs[3].plot(t, [100 * cov[str(k)] if str(k) in cov else 100 * cov[repr(k)] for k in t],
                marker="o", c=C["blue"])
    axs[3].set_xscale("log"); axs[3].set_xlabel("integration time [s]")
    axs[3].set_ylabel("0.5 deg cells hit [%]"); axs[3].set_ylim(0, 100)
    fig.suptitle("Analytic approximation of the Mid-360 non-repetitive scan (not the recorded pattern)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig_scan_pattern.png")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(clouds=fig_clouds, sweep=fig_sweep, conditions=fig_conditions, variants=fig_variants,
               sensitivity=fig_sensitivity, scan=fig_scan)
    for k, fn in fns.items():
        if which in (k, "all"):
            fn()


if __name__ == "__main__":
    main()
