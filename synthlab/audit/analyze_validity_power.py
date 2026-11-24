#!/usr/bin/env python3
"""Metric-validity probes and the power (sample-size) analysis.

Inputs are the derived files written by analyze_sweep.py / analyze_heatlamp.py
plus controlled synthetic probes (seeded).  Outputs go to results/experiment-audit/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from synthlab.audit import metrics as M
from synthlab.audit import stats as S

SEED = 20261123


def bie_probes() -> dict:
    """How HT (BIE) responds to things that are / are not 'disorder'."""
    rng = np.random.default_rng(SEED)
    scene = M.synthetic_thermal_scene(n_people=1, seed=3)
    rows = []
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]:      # sensor noise, kelvin
        vals = []
        for _ in range(8):
            img = M.to_u8(scene + rng.normal(0, sigma, scene.shape))
            b = M.bie(img)
            b["global_shannon_bits"] = M.global_shannon_bits(img)
            vals.append(b)
        m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows.append({"probe": "noise_sigma_K", "x": sigma, **m})
    for dt in [-8, -4, 0, 4, 8, 12]:                             # uniform warming, kelvin
        img = M.to_u8(scene + dt)
        rows.append({"probe": "uniform_offset_K", "x": dt, **M.bie(img),
                     "global_shannon_bits": M.global_shannon_bits(img)})
    for n in [0, 1, 2, 4, 8]:                                    # scene content
        img = M.to_u8(M.synthetic_thermal_scene(n_people=n, seed=3))
        rows.append({"probe": "n_people", "x": n, **M.bie(img),
                     "global_shannon_bits": M.global_shannon_bits(img)})
    return {"rows": rows, "seed": SEED,
            "note": "controlled toy scene (flat 24 C room + warm ellipses); probes the formula only"}


def eigen_probes() -> dict:
    """H_DS vs H_eigen as planar points are replaced by isotropic scatter (fog)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        n = 4000
        k = int(round(frac * n))
        plane = np.column_stack([np.ones(n - k), rng.uniform(0.6, 1.0, n - k),
                                 rng.uniform(0.0, 0.03, n - k)])     # wall-like: l3 ~ 0
        fog = np.column_stack([np.ones(k), rng.uniform(0.75, 1.0, k),
                               rng.uniform(0.6, 1.0, k)])            # volumetric scatter
        e = M.eigen_entropies(np.vstack([plane, fog]))
        rows.append({"fog_fraction": frac, "H_DS": float(e["H_DS"].mean()),
                     "H_eigen": float(e["H_eigen"].mean()), "mean_S": float(e["S"].mean())})
    return {"rows": rows, "seed": SEED,
            "note": "eigenvalue triplets drawn directly; no point cloud or kNN simulated"}


def power_analysis(sweep: dict, heat: dict, scen: pd.DataFrame) -> dict:
    sd = {
        "sweep_same_input_pooled (60 s trials, interleaved)":
            1e3 * sweep["noise_floor_same_input_pooled_sd"]["module_power_w"],
        "replay_A0 (60 s trials, one batch)": heat["replay_noise_floor"]["power_sd_mw_A0"],
        "live_A0 (60 s trials, one evening)": 1e3 * heat["phases"]["A0 (runs 1-26)"]["live_power_w"]["sd"],
        "live_A1_all (includes unexplained state shift)":
            1e3 * math.sqrt(np.mean([heat["phases"][p]["live_power_w"]["sd"] ** 2
                                     for p in ("A1 runs 1-10", "A1 runs 11-25")])
                            + (heat["live_power_mw"]["A1late_minus_A1early_same_condition"]["diff"]
                               / 1e3) ** 2 * (10 * 15) / 25 ** 2),
    }
    module_w = float(scen["power_w"].mean())
    effects = {
        "replay A1(person+lamp) - A0": heat["replay_power_mw"]["A1_minus_A0_all"]["diff"],
        "sweep Real A1 - Real A0": float(scen.set_index("scenario").loc["Real A1-person-lamp", "d_power_mw"]),
        "sweep Objects 4 - Objects 0": float(scen.set_index("scenario").loc["Objects 4", "d_power_mw"]),
        "sweep Noise 32 - Noise 0": float(scen.set_index("scenario").loc["Noise 32", "d_power_mw"]),
        "0.5 % of module power": 5.0 * module_w,
        "1 % of module power": 10.0 * module_w,
        "5 % of module power": 50.0 * module_w,
    }
    table = []
    for en, dv in effects.items():
        for sn, sv in sd.items():
            table.append({"effect": en, "delta_mw": dv, "noise_floor": sn, "sd_mw": sv,
                          "d": abs(dv) / sv, "n_per_condition_80pct": S.n_for_power(dv, sv),
                          "power_at_n3": S.power_two_sample(dv, sv, 3),
                          "power_at_n5": S.power_two_sample(dv, sv, 5),
                          "power_at_n10": S.power_two_sample(dv, sv, 10)})
    mde = [{"noise_floor": sn, "sd_mw": sv,
            **{f"mde_mw_n{n}": S.min_detectable_effect(sv, n) for n in (3, 5, 10, 25)}}
           for sn, sv in sd.items()]
    lat_sd = sweep["noise_floor_same_input_pooled_sd"]
    latency = {
        "yolo_total_sd_ms": lat_sd["yolo_ms"], "inference_sd_ms": lat_sd["yolo_inference_ms"],
        "mde_yolo_total_ms_n3": S.min_detectable_effect(lat_sd["yolo_ms"], 3),
        "mde_inference_ms_n3": S.min_detectable_effect(lat_sd["yolo_inference_ms"], 3),
        "mde_inference_ms_n25": S.min_detectable_effect(lat_sd["yolo_inference_ms"], 25),
    }
    # non-interleaved sessions: a session-level offset cannot be averaged away by more frames
    tj = sweep["drift"]["power_vs_chip_temp_within_scenario"]
    session_sd = tj["mw_per_degC"] * tj["tj_range_c"] / math.sqrt(12)   # uniform over the range
    two_level = {
        "assumed_session_sd_mw_from_thermal_drift": session_sd,
        "sd_of_session_mean_5_trials": S.two_level_sd(sd["replay_A0 (60 s trials, one batch)"],
                                                      session_sd, 5),
        "sessions_per_condition_for_36mw_if_not_interleaved": S.n_for_power(
            effects["sweep Real A1 - Real A0"],
            S.two_level_sd(sd["replay_A0 (60 s trials, one batch)"], session_sd, 5)),
        "note": "assumption: one condition per session, session offset ~ chip-temperature drift only",
    }
    # battery framing
    idle = sweep["idle_power"]["mean_w"]
    worst = max(abs(v) for k, v in effects.items() if "%" not in k)
    battery = {
        "module_power_w": module_w, "idle_w": idle, "largest_measured_effect_mw": worst,
        "pct_of_module": 100 * worst / 1e3 / module_w,
        "pct_of_active_above_idle": 100 * worst / 1e3 / (module_w - idle),
        "runtime_loss_min_per_8h_jetson_only": 8 * 60 * (1 - module_w / (module_w + worst / 1e3)),
        "note": "runtime loss assumes the Jetson module is the ONLY load on the battery; "
                "idle baseline is only 9 tegrastats samples",
    }
    curve = {f"{int(dv)} mW": [S.power_two_sample(dv, 20.0, n) for n in range(2, 41)]
             for dv in (15, 20, 30, 40, 60, 75)}
    return {"noise_floors_mw": sd, "effects_mw": effects, "table": table, "mde": mde,
            "latency": latency, "two_level": two_level, "battery": battery,
            "curve_sd20mw": {"n": list(range(2, 41)), "power": curve}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/experiment-audit"))
    a = ap.parse_args()
    sweep = json.loads((a.out / "sweep_mechanism.json").read_text())
    heat = json.loads((a.out / "heatlamp_live_vs_replay.json").read_text())
    scen = pd.read_csv(a.out / "sweep_scenario_effects.csv")

    bp, ep = bie_probes(), eigen_probes()
    pd.DataFrame(bp["rows"]).to_csv(a.out / "bie_probe.csv", index=False)
    pd.DataFrame(ep["rows"]).to_csv(a.out / "eigen_entropy_probe.csv", index=False)

    trials = pd.read_csv(a.out / "sweep_trials_derived.csv")
    noise = scen[scen["family"] == "noise"].copy()
    noise["level"] = noise["scenario"].str.extract(r"(\d+)").astype(int)
    validity = {
        "sweep_noise_family_HT_by_level": noise.sort_values("level")[
            ["level", "ht", "detections_per_frame"]].to_dict(orient="list"),
        "sweep_r_noise_level_vs_HT": float(np.corrcoef(noise["level"], noise["ht"])[0, 1]),
        "bie_probe_meta": {k: bp[k] for k in ("seed", "note")},
        "eigen_probe_meta": {k: ep[k] for k in ("seed", "note")},
        "bie_cost": {"bie_calc_ms": sweep["duty_cycle"]["bie_calc_ms"],
                     "fraction_of_processing": sweep["duty_cycle"]["bie_calc_fraction_of_processing"],
                     "yolo_inference_ms": sweep["duty_cycle"]["yolo_inference_ms"],
                     "bie_time_range_across_scenarios_ms":
                         float(scen["bie_calc_ms"].max() - scen["bie_calc_ms"].min()),
                     "r_bie_time_vs_detections_trial_level":
                         float(np.corrcoef(trials["calculate_ms"], trials["detections"])[0, 1])},
    }
    pa = power_analysis(sweep, heat, scen)
    pd.DataFrame(pa["table"]).to_csv(a.out / "power_analysis_table.csv", index=False)
    (a.out / "validity_and_power.json").write_text(
        json.dumps({"validity": validity, "power_analysis": pa}, indent=2, default=float))

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(bp["rows"]).round(4).to_string(index=False))
    print(pd.DataFrame(ep["rows"]).round(4).to_string(index=False))
    print(json.dumps(validity, indent=1, default=float))
    print(pd.DataFrame(pa["table"]).round(3).to_string(index=False))
    print(pd.DataFrame(pa["mde"]).round(1).to_string(index=False))
    print(json.dumps({k: pa[k] for k in ("latency", "two_level", "battery")}, indent=1))


if __name__ == "__main__":
    main()
