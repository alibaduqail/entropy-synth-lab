"""Gather harness outputs (pulled from the Jetson) into flat tables for analysis.

  trials  : one row per measured trial (from summary.json)
  frames  : one row per (trial, bank_frame): per-frame columns averaged over bank loops
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

THERMAL_STAGES = ["convert_ms", "bie_ms", "prep_ms", "yolo_pre_ms", "yolo_inf_ms", "yolo_post_ms",
                  "yolo_wall_ms", "yolo_extract_ms", "total_ms", "total_no_bie_ms"]


def _get(d, *keys):
    for key in keys:
        d = d.get(key) if isinstance(d, dict) else None
    return d


def trial_row(folder):
    """Flatten one trial folder's summary.json (thermal or LiDAR)."""
    s = json.loads((folder / "summary.json").read_text())
    row = {"dir": str(folder), "bank": s["bank_name"], "track": s.get("bank_track"),
           "generator": s.get("bank_generator"), "mode": s["mode"], "ok": s["ok"],
           "problems": "; ".join(s["problems"]), "duration_s": s["duration_s"],
           "t_start": s["measurement_start_unix"],
           "power_w": s.get("power_w"), "idle_power_w": s.get("idle_power_w"),
           "gpu_pct": s.get("gpu_pct"), "cpu_pct": s.get("cpu_pct"), "ram_mb": s.get("ram_mb"),
           "temp_cpu_c": s.get("temp_cpu_c"),
           "temp_tj_c": _get(s, "telemetry", "metrics", "temp_tj_c", "mean"),
           "power_sd_w": _get(s, "telemetry", "metrics", "vdd_in_w", "sd"),
           "power_cpu_gpu_cv_w": s.get("power_cpu_gpu_cv_w"),
           "tele_coverage": _get(s, "telemetry", "coverage_frac")}
    if "frames" in s:  # thermal
        row.update(kind="thermal", backend=s["backend"], bie_in_loop=s.get("bie_in_loop", True),
                   units=s["frames"], rate_hz=s["achieved_fps"], HT=s["HT"]["mean"],
                   n_det=s["n_det"]["mean"], n_person=s["n_person"]["mean"],
                   n_candidates=s["n_candidates"]["mean"], any_det_frac=s.get("any_det_frac"),
                   late=s["late_frames"], energy_per_unit_j=s.get("energy_per_frame_j"),
                   energy_above_idle_j=s.get("energy_above_idle_per_frame_j"))
        for stage, d in s["stages_ms"].items():
            row[stage] = d["mean"]
            row[stage.replace("_ms", "_p95_ms")] = d["p95"]
            row[stage.replace("_ms", "_p99_ms")] = d["p99"]
    else:  # lidar
        row.update(kind="lidar", workload=s["workload"], units=s["windows"],
                   rate_hz=s["windows_per_s"], late=s["late_windows"],
                   overrun=s["overrun_windows"], duty_cycle=s["duty_cycle"],
                   energy_per_unit_j=s.get("energy_per_window_j"),
                   energy_above_idle_j=s.get("energy_above_idle_per_window_j"))
        for key, d in s["values"].items():
            row[key] = d["mean"] if d else np.nan
        for stage, d in s["stages_ms"].items():
            row[stage] = d["mean"]
            row[stage.replace("_ms", "_p95_ms")] = d["p95"]
    return row


def collect_trials(root):
    """Every summary.json under root -> DataFrame, with path-derived labels:
    experiment (first folder under root), round (round_RR) and position in the round."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        folder = path.parent
        row = trial_row(folder)
        rel = folder.relative_to(root).parts
        row["experiment"] = rel[0] if rel else ""
        row["round"] = next((int(p[6:]) for p in rel if p.startswith("round_")), -1)
        row["repeat"] = next((int(p[6:]) for p in rel if p.startswith("trial_")), -1)
        row["position"] = int(rel[-1][:2]) if rel and rel[-1][:2].isdigit() and "_" in rel[-1] else -1
        row["rel"] = "/".join(rel)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_frames(trials, columns=None):
    """Per bank-frame means over loops for each thermal trial in `trials`."""
    out = []
    for _, trial in trials.iterrows():
        path = Path(trial["dir"]) / "frames.csv"
        if not path.exists():
            continue
        frames = pd.read_csv(path)
        keep = [c for c in (columns or THERMAL_STAGES + ["HT", "n_det", "n_person", "n_candidates",
                                                           "any_det", "max_conf", "gray_sd", "mean_c"])
                if c in frames.columns]
        grouped = frames.groupby("bank_frame")[keep].mean().reset_index()
        grouped["loops"] = frames.groupby("bank_frame").size().values
        for key in ["rel", "experiment", "bank", "track", "backend", "mode", "round", "bie_in_loop"]:
            grouped[key] = trial.get(key)
        out.append(grouped)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bootstrap_ci(values, func=np.mean, n=4000, seed=0, level=0.95):
    """Percentile bootstrap CI of func over a 1-D sample."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = [func(x[rng.integers(0, x.size, x.size)]) for _ in range(n)]
    lo, hi = np.percentile(stats, [(1 - level) * 50, 100 - (1 - level) * 50])
    return float(lo), float(hi)


def corr_ci(x, y, n=4000, seed=0):
    """Pearson r with a paired percentile bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, (np.nan, np.nan)
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        i = rng.integers(0, x.size, x.size)
        if x[i].std() > 0 and y[i].std() > 0:
            boots.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return r, (float(lo), float(hi))


def ols(y, columns):
    """Least squares y ~ 1 + columns. Returns coefficients (intercept first), their
    standard errors and R^2."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in columns])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    total = ((y - y.mean()) ** 2).sum()
    return beta, np.sqrt(np.diag(cov)), float(1 - (resid @ resid) / total) if total > 0 else np.nan
