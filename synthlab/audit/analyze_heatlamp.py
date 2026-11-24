#!/usr/bin/env python3
"""Live 50-trial heat-lamp experiment vs its frame-identical replay.

The replay batch re-ran the SAME saved frames through the SAME pipeline, so
anything that differs between live and replay for one trial is not caused by
the image content.  Writes results/experiment-audit/heatlamp_*.{csv,json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from synthlab.audit import stats as S

ROOT = Path("/Users/alibaduqail/Documents/thermal camera research work/data")
REPLAY = ROOT / "replay_batches/heat_lamp_replay_20261106T170132-0500"
LIVE = ROOT / "experiment_A_heat_lamp/analysis/heat_lamp_trial_summary.csv"


def msd(x) -> dict:
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "n": int(x.size)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=Path, default=REPLAY)
    ap.add_argument("--live", type=Path, default=LIVE)
    ap.add_argument("--out", type=Path, default=Path("results/experiment-audit"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    h = pd.read_csv(a.replay / "historical_comparison.csv")
    live = pd.read_csv(a.live)
    rep = pd.read_csv(a.replay / "trial_summary.csv")
    d = h.merge(live[["trial_id", "HT_mean", "temperature_mean_c", "threat_detection_rate_pct",
                      "steady_inference_ms"]], on="trial_id")
    d = d.merge(rep[["trial_id", "yolo_inference_ms_mean", "processing_ms_mean"]], on="trial_id")
    d["run"] = d["trial_id"].str.extract(r"run(\d+)").astype(int)
    d["phase"] = np.where(d["condition"] == "A0", "A0 (runs 1-26)",
                          np.where(d["run"] <= 10, "A1 runs 1-10", "A1 runs 11-25"))
    keep = ["order", "trial_id", "condition", "phase", "HT_mean", "threat_detection_rate_pct",
            "steady_inference_ms", "yolo_inference_ms_mean", "original_yolo_total_ms",
            "replay_yolo_total_ms", "original_power_VDD_IN_w", "replay_power_VDD_IN_w",
            "original_temperature_tj_c", "replay_temperature_tj_c", "original_gpu_pct",
            "replay_gpu_pct", "original_cpu_active_core_mean_pct",
            "replay_cpu_active_core_mean_pct"]
    d[keep].rename(columns={"steady_inference_ms": "original_inference_ms",
                            "yolo_inference_ms_mean": "replay_inference_ms"}) \
        .to_csv(a.out / "heatlamp_live_vs_replay.csv", index=False)

    phases = {}
    for ph, g in d.groupby("phase"):
        phases[ph] = {
            "live_inference_ms": msd(g["steady_inference_ms"]),
            "replay_inference_ms": msd(g["yolo_inference_ms_mean"]),
            "live_yolo_total_ms": msd(g["original_yolo_total_ms"]),
            "replay_yolo_total_ms": msd(g["replay_yolo_total_ms"]),
            "live_power_w": msd(g["original_power_VDD_IN_w"]),
            "replay_power_w": msd(g["replay_power_VDD_IN_w"]),
            "live_cpu_pct": msd(g["original_cpu_active_core_mean_pct"]),
            "live_gpu_pct": msd(g["original_gpu_pct"]),
            "live_tj_c": msd(g["original_temperature_tj_c"]),
            "HT": msd(g["HT_mean"]),
        }
    a0, a1 = d[d["condition"] == "A0"], d[d["condition"] == "A1"]
    early, late = a1[a1["run"] <= 10], a1[a1["run"] > 10]

    def contrast(col, scale=1.0):
        return {"A1_minus_A0_all": S.welch(a0[col] * scale, a1[col] * scale),
                "A1early_minus_A0": S.welch(a0[col] * scale, early[col] * scale),
                "A1late_minus_A1early_same_condition": S.welch(early[col] * scale,
                                                               late[col] * scale)}

    out = {
        "phases": phases,
        "live_power_mw": contrast("original_power_VDD_IN_w", 1000.0),
        "live_yolo_total_ms": contrast("original_yolo_total_ms"),
        "live_inference_ms": contrast("steady_inference_ms"),
        "replay_power_mw": contrast("replay_power_VDD_IN_w", 1000.0),
        "replay_yolo_total_ms": contrast("replay_yolo_total_ms"),
        "replay_inference_ms": contrast("yolo_inference_ms_mean"),
        "replay_noise_floor": {
            "power_sd_mw_A0": float(a0["replay_power_VDD_IN_w"].std(ddof=1) * 1e3),
            "power_sd_mw_A1": float(a1["replay_power_VDD_IN_w"].std(ddof=1) * 1e3),
            "yolo_ms_sd_A0": float(a0["replay_yolo_total_ms"].std(ddof=1)),
            "yolo_ms_sd_A1": float(a1["replay_yolo_total_ms"].std(ddof=1)),
            "inference_ms_sd_A0": float(a0["yolo_inference_ms_mean"].std(ddof=1)),
        },
        "live_minus_replay_offset": {
            "power_mw_A0": float((a0["original_power_VDD_IN_w"]
                                  - a0["replay_power_VDD_IN_w"]).mean() * 1e3),
            "tj_c_A0": float((a0["original_temperature_tj_c"]
                              - a0["replay_temperature_tj_c"]).mean()),
        },
        "confound_structure": {
            "order": "all 25 A0 trials first, then all 25 A1 trials (no interleaving)",
            "r_order_vs_condition": float(np.corrcoef(d["order"], d["condition"] == "A1")[0, 1]),
            "r_HT_vs_condition": float(np.corrcoef(d["HT_mean"], d["condition"] == "A1")[0, 1]),
            "within_A1_r_HT_vs_live_power": float(np.corrcoef(a1["HT_mean"],
                                                              a1["original_power_VDD_IN_w"])[0, 1]),
            "within_A1_r_HT_vs_replay_power": float(np.corrcoef(a1["HT_mean"],
                                                                a1["replay_power_VDD_IN_w"])[0, 1]),
            "within_A1_r_detrate_vs_replay_yolo": float(np.corrcoef(
                a1["threat_detection_rate_pct"], a1["replay_yolo_total_ms"])[0, 1]),
            "within_A1_r_HT_vs_replay_yolo": float(np.corrcoef(a1["HT_mean"],
                                                               a1["replay_yolo_total_ms"])[0, 1]),
            "within_A1_partial_r_HT_vs_replay_yolo_given_detrate": S.partial_corr(
                a1["HT_mean"], a1["replay_yolo_total_ms"],
                [a1["threat_detection_rate_pct"]])["r"],
        },
        "share_of_live_A1_minus_A0_power_gap_from_late_shift": None,
    }
    gap_all = out["live_power_mw"]["A1_minus_A0_all"]["diff"]
    gap_early = out["live_power_mw"]["A1early_minus_A0"]["diff"]
    out["share_of_live_A1_minus_A0_power_gap_from_late_shift"] = 1 - gap_early / gap_all
    (a.out / "heatlamp_live_vs_replay.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
