#!/usr/bin/env python3
"""Step-2 analysis: is the cost of a fixed compute graph constant?

  python analyze_null.py --runs data_local/runs --out results/jetson-harness

Expects (pulled from the Jetson):
  <runs>/noise/<backend>_<mode>/trial_XX/      same bank repeated (noise floor)
  <runs>/ctl/<backend>_<mode>[_nobie]/round_RR/NN_<track>__<bank>/   control banks
Writes CSVs, figures and null_test.json (all numbers quoted in the doc come from these).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect as c  # noqa: E402

METRICS = ["total_ms", "total_no_bie_ms", "bie_ms", "convert_ms", "yolo_inf_ms", "yolo_post_ms",
           "yolo_wall_ms", "power_w", "energy_per_unit_j", "gpu_pct", "temp_tj_c"]
COLORS = {"pytorch": "#1f77b4", "tensorrt": "#d62728"}
KIND_COLORS = {"real": "#2ca02c", "control": "#7f7f7f", "mvp-composite": "#ff7f0e"}


def bank_kind(bank):
    return "real" if bank.startswith("real_") else "control" if bank.startswith("ctl_") else "mvp-composite"


def mde(sd, n, alpha_z=1.96, power_z=0.84):
    """Smallest difference between two bank means detectable with 80% power at
    two-sided alpha 0.05 when each bank has n trials with trial-to-trial SD `sd`."""
    return (alpha_z + power_z) * sd * np.sqrt(2.0 / n)


def noise_floor(trials):
    rows = []
    for config, g in trials[trials.experiment == "noise"].groupby("config"):
        g = g.sort_values("t_start")
        row = {"config": config, "bank": g.bank.iloc[0], "n_trials": len(g),
               "seconds_per_trial": float(g.duration_s.mean())}
        for m in METRICS:
            x = g[m].astype(float).values
            row[m + "_mean"] = float(np.nanmean(x))
            row[m + "_sd"] = float(np.nanstd(x, ddof=1))
            if m in ("total_ms", "yolo_inf_ms", "power_w", "energy_per_unit_j"):
                row[m + "_cv_pct"] = 100 * row[m + "_sd"] / row[m + "_mean"]
                row[m + "_mde_n3"] = float(mde(row[m + "_sd"], 3))
        x = g.power_w.values
        row["power_lag1_autocorr"] = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 3 else np.nan
        beta, se, r2 = c.ols(g.power_w, [g.temp_tj_c])
        row["power_w_per_degC"], row["power_temp_r2"] = float(beta[1]), r2
        row["temp_tj_range_c"] = float(g.temp_tj_c.max() - g.temp_tj_c.min())
        rows.append(row)
    return pd.DataFrame(rows)


def by_bank(trials):
    ctl = trials[trials.experiment == "ctl"]
    agg = ctl.groupby(["config", "backend", "mode", "bie_in_loop", "bank"]).agg(
        n_trials=("power_w", "size"), HT=("HT", "mean"), n_det=("n_det", "mean"),
        n_candidates=("n_candidates", "mean"), any_det_frac=("any_det_frac", "mean"),
        rate_hz=("rate_hz", "mean"),
        **{m: (m, "mean") for m in METRICS}, **{m + "_sd": (m, "std") for m in METRICS}).reset_index()
    agg["kind"] = agg.bank.map(bank_kind)
    return agg


def spread_table(banks, floor):
    """Per config and metric: spread of bank means against the trial-to-trial noise SD."""
    rows = []
    for config, g in banks.groupby("config"):
        base = config.replace("_nobie", "")
        f = floor[floor.config == base]
        for m in ["total_ms", "total_no_bie_ms", "bie_ms", "yolo_inf_ms", "yolo_post_ms",
                  "power_w", "energy_per_unit_j"]:
            x = g[m].astype(float)
            if x.isna().all():
                continue
            sd = float(f[m + "_sd"].iloc[0]) if len(f) else np.nan
            lo, hi = g.loc[x.idxmin()], g.loc[x.idxmax()]
            rows.append({"config": config, "metric": m, "n_banks": len(g), "mean": float(x.mean()),
                         "min": float(x.min()), "min_bank": lo.bank, "max": float(x.max()),
                         "max_bank": hi.bank, "range": float(x.max() - x.min()),
                         "range_pct_of_mean": float(100 * (x.max() - x.min()) / x.mean()),
                         "noise_sd": sd,
                         "range_in_noise_sd": float((x.max() - x.min()) / sd) if sd and sd > 0 else np.nan,
                         "within_bank_sd_median": float(g[m + "_sd"].median())})
    return pd.DataFrame(rows)


def frame_level(frames):
    """Per-frame (bank-frame means) relations: which stage follows the input?"""
    rows, fits = [], []
    for config, g in frames.groupby("config"):
        for stage in ["yolo_inf_ms", "yolo_post_ms", "yolo_extract_ms", "bie_ms", "convert_ms",
                      "total_ms", "total_no_bie_ms"]:
            row = {"config": config, "stage": stage, "n_bank_frames": len(g),
                   "mean_ms": float(g[stage].mean()), "sd_across_frames_ms": float(g[stage].std())}
            for x in ["HT", "n_candidates", "n_det", "any_det", "gray_sd"]:
                if g[x].notna().sum() > 10:
                    r, ci = c.corr_ci(g[x], g[stage], n=1000)
                    row[f"r_{x}"], row[f"r_{x}_lo"], row[f"r_{x}_hi"] = r, ci[0], ci[1]
            rows.append(row)
        if g.HT.notna().sum() > 10:
            anyd = (g.n_det > 0).astype(float)
            for target in ["total_no_bie_ms", "yolo_post_ms", "yolo_inf_ms"]:
                b_raw, se_raw, r2_raw = c.ols(g[target], [g.HT])
                b, se, r2 = c.ols(g[target], [g.HT, anyd, g.n_candidates])
                step = g.loc[anyd == 1, target].mean() - g.loc[anyd == 0, target].mean()
                fits.append({"config": config, "target": target, "n": len(g),
                             "ms_per_unit_HT_raw": b_raw[1], "se_raw": se_raw[1], "r2_raw": r2_raw,
                             "ms_per_unit_HT_controlled": b[1], "se_controlled": se[1],
                             "ms_any_detection": b[2], "se_any_detection": se[2],
                             "ms_per_candidate": b[3], "se_per_candidate": se[3], "r2_full": r2,
                             "step_any_vs_none_ms": float(step),
                             "HT_range": float(g.HT.max() - g.HT.min())})
    return pd.DataFrame(rows), pd.DataFrame(fits)


def fig_stages(banks, out):
    configs = [cfg for cfg in ["pytorch_unpaced", "tensorrt_unpaced"] if cfg in set(banks.config)]
    if not configs:
        return
    fig, axes = plt.subplots(1, len(configs), figsize=(7.5 * len(configs), 5.5), sharey=True, squeeze=False)
    parts = [("convert_ms", "TLinear→8-bit", "#9ecae1"), ("bie_ms", "BIE (HT)", "#fdae6b"),
             ("yolo_pre_ms", "YOLO preprocess", "#c7c7c7"), ("yolo_inf_ms", "YOLO inference", "#3182bd"),
             ("yolo_post_ms", "YOLO NMS/post", "#e6550d")]
    for ax, cfg in zip(axes[0], configs):
        g = banks[banks.config == cfg].sort_values("HT")
        bottom = np.zeros(len(g))
        for key, label, color in parts:
            vals = g[key].values if key in g else np.zeros(len(g))
            ax.barh(g.bank, vals, left=bottom, label=label, color=color)
            bottom += vals
        ax.set_title(f"{cfg}: mean time per frame by stage (banks sorted by HT)")
        ax.set_xlabel("ms per frame")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_stage_breakdown.png", dpi=140)
    plt.close(fig)


def fig_inf_vs_post(frames, out):
    g = frames[frames.config.isin(["pytorch_unpaced", "tensorrt_unpaced"])]
    if g.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for (cfg, backend), d in g.groupby(["config", "backend"]):
        color = COLORS.get(backend, "k")
        axes[0].scatter(d.n_candidates, d.yolo_inf_ms, s=5, alpha=0.4, color=color, label=backend)
        axes[1].scatter(d.n_candidates, d.yolo_post_ms, s=5, alpha=0.4, color=color, label=backend)
        axes[2].scatter(d.HT, d.yolo_post_ms, s=5, alpha=0.4, color=color, label=backend)
    axes[0].set(xlabel="pre-NMS candidates per frame", ylabel="YOLO inference (ms)",
                title="Inference: does it follow the input?")
    axes[1].set(xlabel="pre-NMS candidates per frame", ylabel="YOLO NMS/postprocess (ms)",
                title="NMS/postprocess vs candidates")
    axes[2].set(xlabel="thermal entropy HT (BIE)", ylabel="YOLO NMS/postprocess (ms)",
                title="NMS/postprocess vs HT")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1) if "candidates" in ax.get_xlabel() else None
        ax.legend(markerscale=3, fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_inference_vs_nms.png", dpi=140)
    plt.close(fig)


def fig_power(banks, floor, out):
    configs = [cfg for cfg in ["pytorch_paced", "pytorch_unpaced", "tensorrt_paced", "tensorrt_unpaced"]
               if cfg in set(banks.config)]
    if not configs:
        return
    fig, axes = plt.subplots(2, len(configs), figsize=(4.6 * len(configs), 8), squeeze=False)
    for j, cfg in enumerate(configs):
        g = banks[banks.config == cfg]
        f = floor[floor.config == cfg]
        for i, (metric, label) in enumerate([("power_w", "module power VDD_IN (W)"),
                                             ("total_ms", "latency per frame (ms)")]):
            ax = axes[i][j]
            for kind, d in g.groupby("kind"):
                ax.errorbar(d.HT, d[metric], yerr=d[metric + "_sd"], fmt="o", ms=5, capsize=2,
                            color=KIND_COLORS[kind], label=kind)
            if len(f):
                sd, centre = float(f[metric + "_sd"].iloc[0]), float(g[metric].median())
                ax.axhspan(centre - 2 * sd, centre + 2 * sd, color="k", alpha=0.10,
                           label="±2 SD noise floor")
            ax.set(xlabel="thermal entropy HT (BIE)", ylabel=label, title=cfg)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_cost_vs_HT_controls.png", dpi=140)
    plt.close(fig)


def fig_noise(trials, out):
    g = trials[trials.experiment == "noise"]
    if g.empty:
        return
    configs = sorted(g.config.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for cfg in configs:
        d = g[g.config == cfg].sort_values("t_start")
        k = np.arange(len(d))
        axes[0].plot(k, d.total_ms / d.total_ms.mean() * 100 - 100, "o-", label=cfg)
        axes[1].plot(k, d.power_w, "o-", label=cfg)
        axes[2].plot(d.temp_tj_c, d.power_w, "o", label=cfg)
    axes[0].set(xlabel="repeat (same bank)", ylabel="latency, % from config mean",
                title="Trial-to-trial latency")
    axes[1].set(xlabel="repeat (same bank)", ylabel="VDD_IN (W)", title="Trial-to-trial power")
    axes[2].set(xlabel="junction temperature (°C)", ylabel="VDD_IN (W)", title="Power vs chip temperature")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_noise_floor.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trials = c.collect_trials(args.runs)
    trials = trials[(trials.kind == "thermal") & trials.experiment.isin(["noise", "ctl"])].copy()
    trials["config"] = trials.rel.str.split("/").str[1]
    trials.drop(columns=["dir"]).to_csv(args.out / "step2_trials.csv", index=False)
    floor = noise_floor(trials)
    floor.to_csv(args.out / "step2_noise_floor.csv", index=False)
    banks = by_bank(trials)
    banks.to_csv(args.out / "step2_by_bank.csv", index=False)
    spread = spread_table(banks, floor)
    spread.to_csv(args.out / "step2_spread_vs_noise.csv", index=False)
    frames = c.collect_frames(trials[trials.experiment == "ctl"])
    if len(frames):
        frames["config"] = frames.rel.str.split("/").str[1]
        frames = frames.groupby(["config", "backend", "mode", "bank", "bank_frame"]).mean(
            numeric_only=True).reset_index()
        frames.to_csv(args.out / "step2_by_bank_frame.csv", index=False)
        corr, fits = frame_level(frames)
        corr.to_csv(args.out / "step2_stage_correlations.csv", index=False)
        fits.to_csv(args.out / "step2_ht_regression.csv", index=False)
        fig_inf_vs_post(frames, args.out)
    fig_stages(banks, args.out)
    fig_power(banks, floor, args.out)
    fig_noise(trials, args.out)
    summary = {"n_trials": int(len(trials)), "n_ok": int(trials.ok.sum()),
               "problems": sorted(set(p for p in trials.problems if p)),
               "configs": sorted(trials.config.unique())}
    (args.out / "step2_summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(json.dumps(summary, indent=1))
    print(floor.round(4).T)
    print(spread.round(3).to_string())


if __name__ == "__main__":
    main()
