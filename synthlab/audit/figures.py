#!/usr/bin/env python3
"""Figures for docs/experiment-audit.md, built only from results/experiment-audit/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthlab.audit import stats as S

C = {"objects": "#0072B2", "noise": "#D55E00", "real": "#009E73",
     "a": "#0072B2", "b": "#D55E00", "c": "#009E73", "d": "#CC79A7", "e": "#E69F00", "g": "#555555"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight"})


def fig_decomposition(out: Path, sw: dict, scen: pd.DataFrame) -> None:
    by = pd.DataFrame(sw["frame_level"]["by_count"])
    by = by[by["n_frames"] >= 200]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    stages = [("yolo_inference_ms", "YOLO inference (GPU, fixed graph)", C["a"]),
              ("yolo_postprocess_ms", "YOLO postprocess / NMS", C["b"]),
              ("yolo_overhead_ms", "result handling", C["e"])]
    bottom = np.zeros(len(by))
    for col, lab, colr in stages:
        ax[0].bar(by["yolo_count"].astype(int).astype(str), by[col], bottom=bottom, label=lab, color=colr)
        bottom += by[col].values
    ax[0].set_ylim(28, bottom.max() + 2.5)
    ax[0].set_xlabel("detections returned for the frame")
    ax[0].set_ylabel("mean time per frame (ms)  [axis starts at 28]")
    ax[0].set_title(f"Per-frame stage times, {sw['n_frames']:,} frames / 279 trials")
    ax[0].legend(fontsize=7, loc="upper left")
    s = scen.sort_values(["family", "ht"]).reset_index(drop=True)
    x = np.arange(len(s))
    ax[1].bar(x - 0.2, s["inference_ms"] - s["inference_ms"].min(), 0.4, color=C["a"],
              label="inference: change vs fastest scenario")
    ax[1].bar(x + 0.2, s["postprocess_ms"] - s["postprocess_ms"].min(), 0.4, color=C["b"],
              label="postprocess: change vs fastest scenario")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(s["scenario"].str.replace("Real ", "R:").str.replace("-person-lamp", ""),
                          rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("ms above the fastest scenario")
    ax[1].set_title("Where the scenario dependence lives")
    ax[1].legend(fontsize=7)
    fig.savefig(out / "fig_decomposition.png"); plt.close(fig)


def fig_partial(out: Path, t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for fam, g in t.groupby("family"):
        ax[0].scatter(g["ht"], g["yolo_ms"], s=9, color=C[fam], label=fam, alpha=0.7)
        ax[1].scatter(g["detections"], g["yolo_ms"], s=9, color=C[fam], alpha=0.7)
    ax[0].set_xlabel("HT (block image entropy), trial mean"); ax[0].set_ylabel("YOLO total (ms)")
    ax[0].set_title(f"r = {np.corrcoef(t['ht'], t['yolo_ms'])[0, 1]:.2f}  (looks like an entropy effect)")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("detections per frame, trial mean")
    ax[1].set_title(f"r = {np.corrcoef(t['detections'], t['yolo_ms'])[0, 1]:.2f}")
    for fam, g in t.groupby("family"):
        if g["detections"].std() == 0 or fam == "real":
            continue
        rx = S.residualize(g["ht"].values, [g["detections"].values])
        ry = S.residualize(g["yolo_ms"].values, [g["detections"].values])
        r = np.corrcoef(rx, ry)[0, 1]
        ax[2].scatter(rx, ry, s=9, color=C[fam], alpha=0.7, label=f"{fam}: partial r = {r:+.2f}")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("HT residual (detections removed)")
    ax[2].set_ylabel("YOLO ms residual (detections removed)")
    ax[2].set_title("HT beyond detection count: sign depends on family")
    ax[2].legend(fontsize=7)
    fig.savefig(out / "fig_partial_correlation.png"); plt.close(fig)


def fig_drift(out: Path, t: pd.DataFrame, sw: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].scatter(t["order"], t["module_power_w"] * 1e3, s=8, color=C["a"])
    ax[0].set_xlabel("trial order (one 6 h session)"); ax[0].set_ylabel("VDD_IN (mW)", color=C["a"])
    a2 = ax[0].twinx(); a2.spines["right"].set_visible(True)
    a2.plot(t["order"], t["tj_c"], color=C["b"], lw=1); a2.set_ylabel("chip Tj (degC)", color=C["b"])
    ax[0].set_title("Power follows chip temperature over the session")
    rp = (t["module_power_w"] - t.groupby("scenario")["module_power_w"].transform("mean")) * 1e3
    rt = t["tj_c"] - t.groupby("scenario")["tj_c"].transform("mean")
    d = sw["drift"]["power_vs_chip_temp_within_scenario"]
    ax[1].scatter(rt, rp, s=8, color=C["g"])
    xs = np.array([rt.min(), rt.max()]); ax[1].plot(xs, d["mw_per_degC"] * xs, color=C["b"])
    ax[1].set_xlabel("Tj minus scenario mean (degC)"); ax[1].set_ylabel("power minus scenario mean (mW)")
    ax[1].set_title(f"Same input: {d['mw_per_degC']:.1f} mW per degC (R2 = {d['r2']:.2f})")
    g = t.groupby("scenario")["module_power_w"].mean() * 1e3
    ax[2].hist(g - g.min(), bins=8, color=C["c"], alpha=0.8, label="scenario means (13)")
    ax[2].axvline(d["mw_per_degC"] * d["tj_range_c"], color=C["b"], ls="--",
                  label=f"thermal drift across session ({d['mw_per_degC'] * d['tj_range_c']:.0f} mW)")
    ax[2].set_xlabel("mW above the lowest scenario"); ax[2].set_ylabel("scenarios")
    ax[2].set_title("Largest input effect < session drift"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_drift.png"); plt.close(fig)


def fig_heatlamp(out: Path, h: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, (lv, rp, yl) in zip(ax, [("original_inference_ms", "replay_inference_ms",
                                     "YOLO inference only (ms)"),
                                    ("original_power_VDD_IN_w", "replay_power_VDD_IN_w",
                                     "VDD_IN (W)")]):
        a.plot(h["order"], h[lv], "o", ms=4, color=C["b"], label="live capture (2026-10-08)")
        a.plot(h["order"], h[rp], "s", ms=3, color=C["a"], label="replay of the same frames")
        a.axvline(25.5, color="k", lw=0.6, ls=":"); a.axvline(35.5, color=C["d"], lw=0.8, ls="--")
        a.set_xlabel("trial order (A0 x25, then A1 x25)"); a.set_ylabel(yl)
    ax[0].set_title("Inference: +6 ms mid-condition, live only")
    ax[1].set_title("Power: same step = 83 % of live A1-A0 gap")
    ax[0].legend(fontsize=7)
    fig.savefig(out / "fig_heatlamp_live_vs_replay.png"); plt.close(fig)


def fig_power(out: Path, vp: dict) -> None:
    pa = vp["power_analysis"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    n = pa["curve_sd20mw"]["n"]
    for (lab, pw), col in zip(pa["curve_sd20mw"]["power"].items(),
                              [C["g"], C["d"], C["e"], C["c"], C["b"], C["a"]]):
        ax[0].plot(n, pw, label=f"true difference {lab}", color=col)
    ax[0].axhline(0.8, color="k", lw=0.6, ls=":"); ax[0].axvline(3, color="k", lw=0.6, ls="--")
    ax[0].text(3.2, 0.03, "Guide: 3 repeats", fontsize=7)
    ax[0].set_xlabel("trials per condition (60 s each, interleaved)"); ax[0].set_ylabel("power (alpha = 0.05)")
    ax[0].set_title("Power to detect a VDD_IN difference, trial SD = 20 mW"); ax[0].legend(fontsize=7)
    m = pd.DataFrame(pa["mde"])
    x = np.arange(len(m)); w = 0.2
    for i, (k, col) in enumerate(zip(["mde_mw_n3", "mde_mw_n5", "mde_mw_n10", "mde_mw_n25"],
                                     [C["b"], C["e"], C["c"], C["a"]])):
        ax[1].bar(x + (i - 1.5) * w, m[k], w, color=col, label=k.replace("mde_mw_n", "n = "))
    ax[1].set_yscale("log"); ax[1].set_xticks(x)
    ax[1].set_xticklabels(["sweep\n(same input)", "replay A0", "live A0", "live A1\n(state shift)"], fontsize=7)
    for v, lab in [(36, "largest sweep effect 36 mW"), (73, "replay A1-A0 73 mW")]:
        ax[1].axhline(v, color="k", lw=0.6, ls=":"); ax[1].text(-0.45, v * 1.05, lab, fontsize=7)
    ax[1].set_ylabel("minimum detectable difference (mW, 80 % power)")
    ax[1].set_title("What each noise floor can resolve"); ax[1].legend(fontsize=7)
    fig.savefig(out / "fig_power_analysis.png"); plt.close(fig)


def fig_metric(out: Path, scen: pd.DataFrame) -> None:
    b = pd.read_csv(out / "bie_probe.csv"); e = pd.read_csv(out / "eigen_entropy_probe.csv")
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
    nz = b[b["probe"] == "noise_sigma_K"]
    ax[0].plot(nz["x"], nz["value"] / nz["value"].iloc[0], "o-", color=C["b"], label="HT (BIE)")
    ax[0].plot(nz["x"], nz["global_shannon_bits"] / nz["global_shannon_bits"].iloc[0], "s-",
               color=C["a"], label="plain Shannon entropy")
    ax[0].set_xlabel("added sensor noise sigma (K)"); ax[0].set_ylabel("relative to noise-free")
    ax[0].set_title("Toy scene: noise lowers HT"); ax[0].legend(fontsize=7)
    n = scen[scen["family"] == "noise"].copy()
    n["level"] = n["scenario"].str.extract(r"(\d+)").astype(int); n = n.sort_values("level")
    ax[1].plot(n["level"], n["ht"], "o-", color=C["b"])
    ax[1].set_xlabel("noise level in the Jetson sweep"); ax[1].set_ylabel("HT measured on the Jetson")
    ax[1].set_title("Measured: same direction")
    u = b[b["probe"] == "uniform_offset_K"]
    ax[2].plot(u["x"] + 24, u["value"], "o-", color=C["c"])
    ax[2].set_xlabel("room temperature (degC), identical scene structure"); ax[2].set_ylabel("HT")
    ax[2].set_title("HT moves with mean temperature alone")
    ax[3].plot(e["fog_fraction"], e["H_DS"], "o-", color=C["b"], label="H_DS (in use)")
    ax[3].plot(e["fog_fraction"], e["H_eigen"], "s-", color=C["a"], label="H_eigen")
    ax[3].plot(e["fog_fraction"], e["mean_S"], "^-", color=C["c"], label="mean S (sphericity)")
    ax[3].set_xlabel("fraction of neighbourhoods that are isotropic scatter")
    ax[3].set_title("LiDAR: H_DS is the least fog-sensitive"); ax[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_metric_validity.png"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/experiment-audit"))
    o = ap.parse_args().out
    sw = json.loads((o / "sweep_mechanism.json").read_text())
    vp = json.loads((o / "validity_and_power.json").read_text())
    scen = pd.read_csv(o / "sweep_scenario_effects.csv")
    t = pd.read_csv(o / "sweep_trials_derived.csv")
    fig_decomposition(o, sw, scen); fig_partial(o, t); fig_drift(o, t, sw)
    fig_heatlamp(o, pd.read_csv(o / "heatlamp_live_vs_replay.csv")); fig_power(o, vp); fig_metric(o, scen)
    print("figures written to", o)


if __name__ == "__main__":
    main()
