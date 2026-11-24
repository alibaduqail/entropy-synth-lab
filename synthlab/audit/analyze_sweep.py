#!/usr/bin/env python3
"""Mechanism check on the 279-trial synthetic entropy sweep (Orin Nano, 2026-11-16).

Reads (read-only) deliverables/thermal_synth_mvp/entropy_sweep/ from the thermal
project and writes derived tables to results/experiment-audit/.

Questions:
  1. Which pipeline stage carries the input dependence (inference vs postprocess/NMS)?
  2. Does HT explain latency or power once detection count is controlled for?
  3. Does the power effect survive time-order drift / chip temperature / blocking by round?
  4. How big are the effects in ms, mW, mJ per frame, and versus the same-input noise floor?
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from synthlab.audit import stats as S

DEFAULT_SRC = Path("/Users/alibaduqail/Documents/thermal camera research work/"
                   "deliverables/thermal_synth_mvp/entropy_sweep")
FRAME_COLS = ["frame", "bank_frame", "lateness_ms", "calculate_ms", "prepare_ms", "processing_ms",
              "HT", "yolo_count", "yolo_preprocess_ms", "yolo_inference_ms",
              "yolo_postprocess_ms", "yolo_total_ms", "yolo_overhead_ms"]


def load_trials(src: Path) -> pd.DataFrame:
    t = pd.read_csv(src / "summaries.csv")
    t["order"] = np.arange(len(t))
    # scenario order was reshuffled every 13 trials; recover the round as a block
    seen, rnd, rounds = set(), 0, []
    for sc in t["scenario"]:
        if sc in seen:
            rnd, seen = rnd + 1, set()
        seen.add(sc)
        rounds.append(rnd)
    t["round"] = rounds
    tg = pd.read_csv(src / "tegrastats_samples.csv")
    tg = tg[tg["measured_window"].astype(str) == "True"]
    agg = tg.groupby("trial").agg(tj_c=("temperature_tj_c", "mean"),
                                  cpu_c=("temperature_cpu_c", "mean"),
                                  p_cpu_gpu_w=("power_VDD_CPU_GPU_CV_w", "mean"),
                                  p_soc_w=("power_VDD_SOC_w", "mean"),
                                  p_in_sd_w=("power_VDD_IN_w", "std"),
                                  n_tegra=("power_VDD_IN_w", "size"),
                                  t_start=("unix_time", "min")).reset_index()
    return t.merge(agg, on="trial", how="left")


def load_frames(src: Path, trials: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for tr, sc, fam in zip(trials["trial"], trials["scenario"], trials["family"]):
        f = pd.read_csv(src / "trials" / tr / "frames.csv", usecols=FRAME_COLS)
        f["trial"], f["scenario"], f["family"] = tr, sc, fam
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def idle_power_w(src: Path) -> dict:
    vals = []
    for line in pd.read_csv(src / "idle_tegrastats.csv")["tegrastats_line"]:
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            vals.append(int(m.group(1)) / 1000.0)
    return {"mean_w": float(np.mean(vals)), "sd_w": float(np.std(vals, ddof=1)), "n": len(vals)}


def corr_block(t: pd.DataFrame, label: str) -> list:
    rows = []
    for y in ["yolo_ms", "yolo_inference_ms", "yolo_postprocess_ms", "calculate_ms",
              "processing_ms", "module_power_w"]:
        ht, det, yy = t["ht"].values, t["detections"].values, t[y].values
        if np.std(yy) == 0 or np.std(ht) == 0:
            continue
        rows.append({
            "subset": label, "n": len(t), "outcome": y,
            "r_HT": float(np.corrcoef(ht, yy)[0, 1]),
            "r_det": float(np.corrcoef(det, yy)[0, 1]) if np.std(det) > 0 else np.nan,
            "r_HT_given_det": S.partial_corr(ht, yy, [det])["r"] if np.std(det) > 0 else np.nan,
            "p_HT_given_det": S.partial_corr(ht, yy, [det])["p"] if np.std(det) > 0 else np.nan,
            "r_det_given_HT": S.partial_corr(det, yy, [ht])["r"] if np.std(det) > 0 else np.nan,
            "p_det_given_HT": S.partial_corr(det, yy, [ht])["p"] if np.std(det) > 0 else np.nan,
        })
    return rows


def regressions(t: pd.DataFrame) -> dict:
    out = {}
    rd = S.one_hot(t["round"])
    zht = (t["ht"] - t["ht"].mean()) / t["ht"].std()
    zdet = (t["detections"] - t["detections"].mean()) / t["detections"].std()
    for y, scale, unit in [("yolo_ms", 1.0, "ms"), ("yolo_postprocess_ms", 1.0, "ms"),
                           ("yolo_inference_ms", 1.0, "ms"), ("calculate_ms", 1.0, "ms"),
                           ("module_power_w", 1000.0, "mW")]:
        yy = t[y].values * scale
        base = np.column_stack([zht, zdet])
        out[y] = {
            "unit_per_1sd": unit,
            "ht_only": S.ols(yy, zht.values, ["ht_z"]),
            "det_only": S.ols(yy, zdet.values, ["det_z"]),
            "ht_det": S.ols(yy, base, ["ht_z", "det_z"]),
            "ht_det_roundFE": {k: v for k, v in S.ols(
                yy, np.column_stack([base, rd]),
                ["ht_z", "det_z"] + [f"round{i}" for i in range(rd.shape[1])]).items()
                if not k.startswith("round")},
            "ht_det_tj_order": S.ols(
                yy, np.column_stack([base, t["tj_c"], t["order"]]),
                ["ht_z", "det_z", "tj_c", "order"]),
        }
    return out


def scenario_table(t: pd.DataFrame, fps: float, idle_w: float) -> pd.DataFrame:
    g = t.groupby("scenario", sort=False)
    tab = g.agg(family=("family", "first"), n=("trial", "size"), ht=("ht", "mean"),
                detections_per_frame=("detections", "mean"),
                yolo_ms=("yolo_ms", "mean"), yolo_ms_sd=("yolo_ms", "std"),
                inference_ms=("yolo_inference_ms", "mean"),
                inference_ms_sd=("yolo_inference_ms", "std"),
                postprocess_ms=("yolo_postprocess_ms", "mean"),
                bie_calc_ms=("calculate_ms", "mean"),
                power_w=("module_power_w", "mean"), power_w_sd=("module_power_w", "std"),
                tj_c=("tj_c", "mean")).reset_index()
    ref = {"objects": "Objects 0", "noise": "Noise 0", "real": "Real A0-empty"}
    base = tab.set_index("scenario")
    tab["d_yolo_ms"] = [r.yolo_ms - base.loc[ref[r.family], "yolo_ms"] for r in tab.itertuples()]
    tab["d_power_mw"] = [1000 * (r.power_w - base.loc[ref[r.family], "power_w"])
                         for r in tab.itertuples()]
    tab["d_energy_mj_per_frame"] = tab["d_power_mw"] / fps
    tab["d_power_pct_of_module"] = 100 * tab["d_power_mw"] / 1000 / tab["power_w"]
    tab["d_power_pct_of_active"] = 100 * tab["d_power_mw"] / 1000 / (tab["power_w"] - idle_w)
    return tab.sort_values(["family", "ht"])


def frame_level(f: pd.DataFrame) -> dict:
    out = {}
    cnt = f["yolo_count"].values.astype(float)
    for y in ["yolo_inference_ms", "yolo_postprocess_ms", "yolo_overhead_ms",
              "yolo_preprocess_ms", "yolo_total_ms", "calculate_ms"]:
        fit = S.ols(f[y].values, cnt, ["per_detection_ms"])
        # same slope using only within-trial variation (trial fixed effects by centring)
        yc = f[y] - f.groupby("trial")[y].transform("mean")
        cc = f["yolo_count"] - f.groupby("trial")["yolo_count"].transform("mean")
        den = float((cc ** 2).sum())
        out[y] = {"slope_ms_per_detection": fit["per_detection_ms"]["beta"],
                  "intercept_ms": fit["intercept"]["beta"], "r2": fit["_model"]["r2"],
                  "within_trial_slope": float((cc * yc).sum() / den) if den > 0 else None}
    # zero vs non-zero detections: the NMS "step"
    z, nz = f[f["yolo_count"] == 0], f[f["yolo_count"] > 0]
    out["step_any_detection"] = {
        c: {"zero": float(z[c].mean()), "nonzero": float(nz[c].mean()),
            "diff": float(nz[c].mean() - z[c].mean())}
        for c in ["yolo_inference_ms", "yolo_postprocess_ms", "yolo_overhead_ms", "yolo_total_ms"]}
    out["n_frames"] = int(len(f))
    out["frac_frames_with_detection"] = float((f["yolo_count"] > 0).mean())
    # HT beyond detection count, frame level, within trial
    for y in ["yolo_total_ms", "yolo_inference_ms", "calculate_ms"]:
        g = f.groupby("trial")
        yc = (f[y] - g[y].transform("mean")).values
        hc = (f["HT"] - g["HT"].transform("mean")).values
        cc = (f["yolo_count"] - g["yolo_count"].transform("mean")).values.astype(float)
        out[f"within_trial_partial_r_HT_{y}_given_count"] = S.partial_corr(hc, yc, [cc])["r"]
    icc = S.icc_oneway(f["yolo_total_ms"].values, f["trial"].values)
    out["icc_yolo_total_ms_by_trial"] = icc
    # by-count stage means for the decomposition plot
    by = f.groupby("yolo_count")[["yolo_preprocess_ms", "yolo_inference_ms",
                                  "yolo_postprocess_ms", "yolo_overhead_ms",
                                  "calculate_ms"]].mean()
    by["n_frames"] = f.groupby("yolo_count").size()
    out["by_count"] = by.reset_index().to_dict(orient="list")
    return out


def drift(t: pd.DataFrame) -> dict:
    out = {}
    for y, scale in [("module_power_w", 1000.0), ("yolo_ms", 1.0), ("yolo_inference_ms", 1.0),
                     ("tj_c", 1.0), ("calculate_ms", 1.0)]:
        v = t[y].values * scale
        resid = v - t.groupby("scenario")[y].transform("mean").values * scale
        fit = S.ols(resid, t["order"].values, ["per_trial"])
        rho = S.lag1_autocorr(resid)
        out[y] = {"raw_range": float(v.max() - v.min()),
                  "scenario_centred_sd": float(resid.std(ddof=1)),
                  "trend_per_100_trials": 100 * fit["per_trial"]["beta"],
                  "trend_p": fit["per_trial"]["p"],
                  "lag1_autocorr_scenario_centred": rho,
                  "effective_n_of_279": S.effective_n(len(t), rho),
                  "first_round_minus_rest": float(resid[t["round"] == 0].mean()
                                                  - resid[t["round"] > 0].mean())}
    resid_p = (t["module_power_w"] - t.groupby("scenario")["module_power_w"].transform("mean")) * 1e3
    resid_t = t["tj_c"] - t.groupby("scenario")["tj_c"].transform("mean")
    fit = S.ols(resid_p.values, resid_t.values, ["mw_per_degC"])
    out["power_vs_chip_temp_within_scenario"] = {
        "mw_per_degC": fit["mw_per_degC"]["beta"], "se": fit["mw_per_degC"]["se"],
        "p": fit["mw_per_degC"]["p"], "r2": fit["_model"]["r2"],
        "tj_range_c": float(t["tj_c"].max() - t["tj_c"].min())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=Path("results/experiment-audit"))
    ap.add_argument("--n-perm", type=int, default=5000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fps = json.loads((a.src / "manifest.json").read_text())["fps"]

    t = load_trials(a.src)
    f = load_frames(a.src, t)
    idle = idle_power_w(a.src)

    rows = corr_block(t, "all 279 trials")
    for fam in ["objects", "noise", "real"]:
        rows += corr_block(t[t["family"] == fam], f"family={fam}")
    rows += corr_block(t[t["family"] != "real"], "synthetic only")
    pd.DataFrame(rows).to_csv(a.out / "sweep_partial_correlations.csv", index=False)

    tab = scenario_table(t, fps, idle["mean_w"])
    tab.to_csv(a.out / "sweep_scenario_effects.csv", index=False)

    pooled_sd = {y: float(np.sqrt((t.groupby("scenario")[y].var(ddof=1)
                                   * (t.groupby("scenario").size() - 1)).sum()
                                  / (len(t) - t["scenario"].nunique())))
                 for y in ["module_power_w", "yolo_ms", "yolo_inference_ms",
                           "yolo_postprocess_ms", "calculate_ms", "processing_ms"]}
    perm = {}
    for x in ["ht", "detections"]:
        for y, sc in [("module_power_w", 1000.0), ("yolo_ms", 1.0), ("yolo_inference_ms", 1.0)]:
            perm[f"{y}~{x}"] = S.blocked_permutation_slope(
                t[x].values, t[y].values * sc, t["round"].values, a.n_perm, seed=1711)
    # HT after removing detections, permuted within round
    ht_res = S.residualize(t["ht"].values, [t["detections"].values])
    for y, sc in [("module_power_w", 1000.0), ("yolo_ms", 1.0)]:
        y_res = S.residualize(t[y].values * sc, [t["detections"].values])
        perm[f"{y}~ht|detections"] = S.blocked_permutation_slope(
            ht_res, y_res, t["round"].values, a.n_perm, seed=1711)

    duty = {
        "frame_period_ms": 1000.0 / fps,
        "mean_processing_ms": float(t["processing_ms"].mean()),
        "cpu_plus_gpu_busy_fraction": float(t["processing_ms"].mean() / (1000.0 / fps)),
        "gpu_inference_fraction_of_period": float(t["yolo_inference_ms"].mean() / (1000.0 / fps)),
        "bie_calc_fraction_of_processing": float((t["calculate_ms"] / t["processing_ms"]).mean()),
        "bie_calc_ms": float(t["calculate_ms"].mean()),
        "yolo_inference_ms": float(t["yolo_inference_ms"].mean()),
        "yolo_postprocess_ms": float(t["yolo_postprocess_ms"].mean()),
        "tegrastats_gpu_pct_mean": float(t["gpu_pct"].mean()),
    }
    summary = {
        "source": str(a.src), "n_trials": int(len(t)), "n_rounds": int(t["round"].nunique()),
        "n_frames": int(len(f)), "fps": fps, "idle_power": idle,
        "active_power_above_idle_w": float(t["module_power_w"].mean() - idle["mean_w"]),
        "ranges": {y: {"min": float(t[y].min()), "max": float(t[y].max()),
                       "sd": float(t[y].std(ddof=1))}
                   for y in ["ht", "detections", "yolo_ms", "yolo_inference_ms",
                             "yolo_postprocess_ms", "calculate_ms", "module_power_w", "tj_c"]},
        "noise_floor_same_input_pooled_sd": pooled_sd,
        "tegrastats_samples_per_trial": float(t["n_tegra"].mean()),
        "within_trial_power_sample_sd_w": float(t["p_in_sd_w"].mean()),
        "regressions_per_1sd": regressions(t),
        "blocked_permutation_by_round": perm,
        "frame_level": frame_level(f),
        "drift": drift(t),
        "duty_cycle": duty,
        "seed": 1711,
    }
    (a.out / "sweep_mechanism.json").write_text(json.dumps(summary, indent=2, default=float))
    t.drop(columns=["bank_shannon_bits"]).to_csv(a.out / "sweep_trials_derived.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(tab.round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ["idle_power", "active_power_above_idle_w",
                                              "noise_floor_same_input_pooled_sd",
                                              "blocked_permutation_by_round", "drift",
                                              "duty_cycle"]}, indent=1, default=float))
    fl = dict(summary["frame_level"]); fl.pop("by_count")
    print(json.dumps(fl, indent=1, default=float))
    for y, r in summary["regressions_per_1sd"].items():
        print(y, {m: {k: (round(v["beta"], 4), round(v["p"], 4)) for k, v in fit.items()
                      if k in ("ht_z", "det_z", "tj_c", "order")} for m, fit in r.items()
                  if m != "unit_per_1sd"})


if __name__ == "__main__":
    main()
