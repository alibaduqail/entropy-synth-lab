"""Figures for docs/radar-sim.md, built from the files in results/radar-sim/.

Usage: python3 -m synthlab.radar_sim.figures --results results/radar-sim [--bench DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import conditions as cond  # noqa: E402
from . import config, metrics, process, simulate  # noqa: E402
from . import scene as sc  # noqa: E402

COND_COLORS = {"A0": "#9e9e9e", "A1": "#4c78a8", "A2": "#2f5d8a", "A3": "#1b3a5c",
               "B0": "#8fd19e", "B1": "#54a868", "B2": "#2e7d46", "B3": "#145a2a",
               "M1": "#f58518", "M3": "#c9480a"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_rd_maps(out, scene_path=None, seed=0):
    cfg = config.preset()
    scene = sc.load_scene(scene_path)
    names = ["A0", "A1", "A2", "A3", "M3", "B0", "B1", "B2", "B3", "M1"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    info = {}
    ext = [0, cfg.range_max_m, -cfg.v_max_mps, cfg.v_max_mps]
    for ax, name in zip(axes.ravel(), names):
        c = cond.build_condition(name, scene, seed=seed)
        sim = simulate.RadarSimulator(cfg, scene, c["people"], c["extra_objects"], c["fog"], seed=seed)
        proc = process.Processor(cfg)
        for i in range(12):
            res = proc.process(sim.frame(i))
        noise = np.median(res.rd_dyn)
        im = ax.imshow(10 * np.log10(res.rd_total / noise + 1e-12), origin="lower", aspect="auto",
                       extent=ext, vmin=0, vmax=90, cmap="turbo")
        if len(res.points):
            ax.scatter(res.points[:, 5], res.points[:, 3], s=14, facecolors="none",
                       edgecolors="white", linewidths=0.9)
        ent = metrics.frame_entropies(res, cfg)
        info[name] = dict(n_det=res.n_det, **ent)
        ax.set_title("%s: %d points, H=%.2f bits" % (name, res.n_det, ent["h_rd_total_bits"]),
                     fontsize=10)
        ax.set_xlim(0, 4.0); ax.set_ylim(-3, 3); ax.grid(False)
    for ax in axes[-1]:
        ax.set_xlabel("range (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("radial velocity (m/s)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="power over noise floor (dB)")
    fig.suptitle("Simulated IWR6843ISK range-Doppler maps (%s, static scene included; circles = "
                 "CFAR detections after clutter removal; H = entropy of the map)" % os.path.basename(scene["_path"]))
    fig.savefig(os.path.join(out, "rd_maps_conditions.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out, "rd_maps_info.json"), "w") as f:
        json.dump({"seed": seed, "frame": 11, "panels": info}, f, indent=1)


def fig_fog(out):
    c = json.load(open(os.path.join(out, "fog_curves.json")))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for (k, v), col in zip(c["kl"].items(), ["#3b6fb6", "#2ca02c", "#d62728"]):
        ax[0].plot(c["f_ghz"], v, label=k.replace("C", " °C"), color=col)
    ax[0].axvspan(60, 64, color="orange", alpha=0.3, label="IWR6843 band")
    ax[0].set_xlabel("frequency (GHz)"); ax[0].set_ylabel("K_l  (dB/km per g/m³)")
    ax[0].set_title("ITU-R P.840 liquid-water attenuation"); ax[0].legend()
    for (k, v), col in zip(c["two_way_db"].items(), ["#9ecae1", "#4292c6", "#084594"]):
        ax[1].loglog(c["lwc"], v, color=col, label="fog, target at %s" % k.replace("m", " m"))
        ax[1].axhline(c["o2_two_way_db"][k], color=col, ls=":", lw=1)
    ax[1].text(0.012, c["o2_two_way_db"]["4m"] * 1.15, "dotted: O₂ line, always present", fontsize=8)
    for name, m in c["levels"].items():
        if m > 0:
            ax[1].axvline(m, color="green", alpha=0.5, lw=1)
            ax[1].text(m, 2e-5, name, rotation=90, va="bottom", ha="right", fontsize=8, color="green")
    ax[1].axhline(1.0, color="red", ls="--", lw=1); ax[1].text(0.012, 1.15, "1 dB", color="red", fontsize=8)
    ax[1].set_ylim(1e-5, 3); ax[1].set_xlabel("liquid water content (g/m³)")
    ax[1].set_ylabel("two-way attenuation (dB)"); ax[1].set_title("62 GHz fog loss over room paths")
    ax[1].legend(fontsize=8, loc="lower right")
    for (k, v), col in zip(c["fog_rcs_dbsm_2m"].items(), ["#6a51a3", "#cb181d"]):
        ax[2].semilogx(c["lwc"], v, color=col, label="mist in one cell, %s drops" % k.replace("um", " µm"))
    ax[2].axhline(c["noise_equivalent_rcs_dbsm_2m"], color="k", ls="--", label="noise-equivalent RCS (2 m)")
    ax[2].axhline(-3, color="#4c78a8", lw=2, label="torso, 0.5 m²")
    ax[2].set_xlabel("liquid water content (g/m³)"); ax[2].set_ylabel("RCS (dBsm)")
    ax[2].set_title("Fog backscatter vs noise floor"); ax[2].legend(fontsize=8)
    ax[3].semilogx(c["film_um"], c["film_two_way_db"], color="#d62728")
    ax[3].set_xlabel("water film on radome (µm)"); ax[3].set_ylabel("two-way loss (dB)", color="#d62728")
    a2 = ax[3].twinx(); a2.grid(False); a2.spines["right"].set_visible(True)
    a2.semilogx(c["film_um"], c["film_shift_mm"], color="#1f77b4")
    a2.set_ylabel("apparent range shift (mm)", color="#1f77b4")
    ax[3].set_title("The real risk: condensation on the radar")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fog_attenuation.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_stillness(out):
    s = json.load(open(os.path.join(out, "stillness_summary.json")))
    ex = json.load(open(os.path.join(out, "stillness_examples.json")))
    d = s["deltas_mm"]
    fps = 1.0 / s["setup"]["frame_period_s"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    cols = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c"]
    for e, col in zip(ex, cols):
        t = np.arange(len(e["series_mm"])) / fps
        lab = "no event" if e["delta_mm"] == 0 else "%g mm %s" % (e["delta_mm"], e["direction"])
        ax[0].plot(t, np.clip(e["series_mm"], -40, 40), color=col, label=lab, lw=1.2)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("torso radial displacement from phase (mm)")
    ax[0].set_title("Phase track of the torso bin (clipped ±40 mm)"); ax[0].legend(fontsize=8)
    show = [("net_shift_mm", "torso phase: net shift", "#1f77b4", "-"),
            ("excursion_mm", "torso phase: excursion", "#17becf", "-"),
            ("lateral_shift_mm", "azimuth phase: lateral shift", "#2ca02c", "-"),
            ("motion_db_max", "non-zero-Doppler energy", "#ff7f0e", "--"),
            ("range_centroid_shift_mm", "range-profile centroid", "#9467bd", "--"),
            ("centroid_drift_mm", "CFAR point-cloud centroid", "#d62728", ":")]
    panels = [("sway_2mm", "radial", "Move toward radar, natural sway (2 mm rms)"),
              ("sway_2mm", "lateral", "Move sideways, natural sway (2 mm rms)"),
              ("sway_0mm", "radial", "Move toward radar, no sway (ideal target)")]
    for a, (sk, direction, title) in zip(ax[1:], panels):
        for m, lab, col, ls in show:
            a.semilogx(d, s["curves"][sk]["%s/%s" % (m, direction)]["p_detect"], ls, color=col,
                       marker="o", ms=4, label=lab)
        a.axvline(20, color="k", alpha=0.3, lw=1); a.axvline(200, color="k", alpha=0.3, lw=1)
        a.text(20, 1.03, "2 cm", ha="center", fontsize=8); a.text(200, 1.03, "20 cm", ha="center", fontsize=8)
        a.set_ylim(-0.03, 1.1); a.set_xlabel("size of the movement (mm)")
        a.set_ylabel("fraction of trials flagged"); a.set_title(title, fontsize=10)
    ax[1].legend(fontsize=7.5, loc="center left")
    fig.suptitle("How small a movement can the radar witness flag? (n=%d seeds per point, threshold "
                 "= max of %d no-event trials)" % (s["setup"]["seeds_per_point"], s["setup"]["base_seeds"]),
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stillness_detection_curve.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_bench(out, bench_dir, tag):
    meta = json.load(open(os.path.join(bench_dir, "summary.json")))
    summ = meta["summary"]
    names = [n for n in cond.CONDITIONS if any(e["condition"] == n for e in summ)]
    stages = process.STAGES

    def agg(name, fn):
        v = [fn(e) for e in summ if e["condition"] == name]
        v = [x for x in v if x is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, 0.0)

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    bottom = np.zeros(len(names))
    scol = plt.cm.viridis(np.linspace(0.05, 0.75, 5)).tolist() + ["#ff7f0e", "#d62728", "#8c564b"]
    for st, col in zip(stages, scol):
        v = np.array([agg(n, lambda e: e["free"][st]["mean"])[0] for n in names])
        ax[0].bar(names, v, bottom=bottom, color=col,
                  label=st + (" (data-dep.)" if st in process.DATA_STAGES else ""))
        bottom += v
    ax[0].set_ylabel("processing time per frame (ms)"); ax[0].legend(fontsize=7.5, ncol=2)
    ax[0].set_title("Per-stage cost by condition (%s)" % tag)
    ax[0].set_ylim(0, bottom.max() * 1.45)
    for mode, col, off in (("idle", "#bdbdbd", -0.27), ("paced", "#4c78a8", 0.0), ("free", "#f58518", 0.27)):
        mv = [agg(n, lambda e: e[mode]["vdd_in_mw_mean"]) for n in names]
        if not np.isfinite([m[0] for m in mv]).any():
            continue
        ax[1].bar(np.arange(len(names)) + off, [m[0] / 1e3 for m in mv], 0.27,
                  yerr=[m[1] / 1e3 for m in mv], color=col, capsize=2,
                  label={"idle": "idle before", "paced": "processing at 18.2 fps",
                         "free": "processing flat out"}[mode])
    ax[1].set_xticks(np.arange(len(names))); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("VDD_IN (W), mean ± sd over repeats"); ax[1].legend(fontsize=8)
    ax[1].set_title("Module power by condition")
    lo = ax[1].get_ylim()[1]
    ax[1].set_ylim(max(0, lo * 0.5), lo * 1.12)
    # cost vs detections, per frame
    rows = list(csv.DictReader(open(os.path.join(bench_dir, "frames.csv"))))
    for mode, mk, alpha in (("free", "o", 0.25), ("stress", "s", 0.25)):
        sel = [r for r in rows if r["mode"] == mode]
        if not sel:
            continue
        nd = np.array([float(r["n_det"]) for r in sel])
        for key, col in (("fixed", "#31688e"), ("data_dependent", "#d62728")):
            ax[2].scatter(nd + 0.5, [float(r[key]) for r in sel], s=5, marker=mk, color=col,
                          alpha=alpha, label="%s stages (%s)" % (key.replace("_", "-"), mode),
                          rasterized=True)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("CFAR detections per frame (+0.5)"); ax[2].set_ylabel("time per frame (ms)")
    ax[2].set_title("Cost vs detections (each dot = one frame)")
    lg = ax[2].legend(fontsize=7.5, markerscale=3)
    for h in lg.legend_handles:
        h.set_alpha(1)
    keys = [("h_rd_total_norm", "H(range-Doppler, full map)", "#4c78a8"),
            ("h_rd_dyn_norm", "H(range-Doppler, clutter removed)", "#f58518")]
    x = np.arange(len(names))
    for i, (k, lab, col) in enumerate(keys):
        ax[3].bar(x + (i - 1) * 0.27, [agg(n, lambda e: e["entropy"][k])[0] for n in names], 0.27,
                  color=col, label=lab + " / max")
    hp = [agg(n, lambda e: e["entropy"]["h_points_bits"])[0] for n in names]
    ax[3].bar(x + 0.27, np.array(hp) / max(max(hp), 1e-9), 0.27, color="#54a868",
              label="H(CFAR points) / %.1f bits" % max(hp))
    ax[3].set_xticks(x); ax[3].set_xticklabels(names); ax[3].set_ylabel("normalised entropy")
    ax[3].set_title("Radar entropy candidates by condition"); ax[3].legend(fontsize=7.5)
    ax[3].set_ylim(0, 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cost_entropy_%s.png" % tag), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/radar-sim")
    ap.add_argument("--bench", default=None, help="bench dir(s), comma separated tag=dir")
    ap.add_argument("--only", default="rd,fog,still,bench")
    a = ap.parse_args(argv)
    only = a.only.split(",")
    if "rd" in only:
        fig_rd_maps(a.results)
    if "fog" in only and os.path.exists(os.path.join(a.results, "fog_curves.json")):
        fig_fog(a.results)
    if "still" in only and os.path.exists(os.path.join(a.results, "stillness_summary.json")):
        fig_stillness(a.results)
    if "bench" in only and a.bench:
        for item in a.bench.split(","):
            tag, d = item.split("=")
            fig_bench(a.results, d, tag)


if __name__ == "__main__":
    main()
