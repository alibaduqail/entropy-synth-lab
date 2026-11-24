"""Offline analysis of the sponge search: statistics (JSON) for the doc. Figures: figures.py.

Inputs (pulled from the Jetson into results/sponge-search/):
  runs/<run>/evals.jsonl     every evaluation made during a search
  measure/<name>.json        interleaved repeated measurements (measure.py)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = "results/sponge-search"
STAGES = ["bie_ms", "prep_ms", "pre_ms", "inf_ms", "post_ms", "overhead_ms", "log_ms"]


def load_cloud(root=ROOT, timed_only=True):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "runs", "*", "evals.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            m, b = r["m"], r.get("base") or {}
            if timed_only and "frame_ms" not in m:
                continue
            d = {"run": r["run"], "mode": r["mode"], "tier": r["tier"], "gen": r["gen"],
                 "f": r["f"], "ht": m["ht"], "shannon": m["shannon"], "n_det": m["n_det"],
                 "n_cand": m["n_cand"]}
            if "frame_ms" in m:
                d.update(frame_ms=m["frame_ms"], energy_mj=m["energy_mj"],
                         vdd_in_mw=m["vdd_in_mw"], d_ms=m["frame_ms"] - b["frame_ms"],
                         d_mj=m["energy_mj"] - b["energy_mj"],
                         d_mw=m["vdd_in_mw"] - b["vdd_in_mw"], **{s: m[s] for s in STAGES})
            rows.append(d)
    return rows


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def corr(x, y):
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    return {"pearson_r": float(r), "p": float(p), "spearman_r": float(rs), "n": int(len(x))}


def knee_fit(x, y, n_grid=40):
    """Two-segment continuous piecewise-linear fit vs a straight line (BIC comparison)."""
    n = len(x)
    A = np.c_[np.ones(n), x]
    res_lin = float(((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2).sum())
    best = (np.inf, None, None)
    for bp in np.quantile(x, np.linspace(0.05, 0.95, n_grid)):
        B = np.c_[np.ones(n), x, np.maximum(0.0, x - bp)]
        coef = np.linalg.lstsq(B, y, rcond=None)[0]
        res = float(((y - B @ coef) ** 2).sum())
        if res < best[0]:
            best = (res, float(bp), coef.tolist())
    bic_lin = n * np.log(res_lin / n) + 2 * np.log(n)
    bic_knee = n * np.log(best[0] / n) + 4 * np.log(n)
    return {"breakpoint": best[1], "slope_before": best[2][1],
            "slope_after": best[2][1] + best[2][2], "delta_bic_knee_minus_linear":
            float(bic_knee - bic_lin), "r2_linear": 1 - res_lin / float(((y - y.mean()) ** 2).sum()),
            "r2_knee": 1 - best[0] / float(((y - y.mean()) ** 2).sum())}


def slope_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.polyfit(x, y, 1)[0]
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    boots = [np.polyfit(x[i], y[i], 1)[0] for i in idx if np.std(x[i]) > 0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope": float(s), "ci95": [float(lo), float(hi)], "n": int(len(x))}


def cloud_stats(rows):
    y, e, pw = col(rows, "d_ms"), col(rows, "d_mj"), col(rows, "d_mw")
    ht, sh, nd, nc = (col(rows, k) for k in ("ht", "shannon", "n_det", "n_cand"))
    out = {"n_points": len(rows), "runs": sorted({r["run"] for r in rows}),
           "ranges": {"ht": [float(ht.min()), float(ht.max())],
                      "n_det": [float(nd.min()), float(nd.max())],
                      "n_cand": [float(nc.min()), float(nc.max())],
                      "d_ms": [float(y.min()), float(y.max())]}}
    for name, v in (("ht", ht), ("shannon", sh), ("n_det", nd), ("n_cand", nc)):
        out[f"d_ms_vs_{name}"] = corr(v, y)
        out[f"d_mj_vs_{name}"] = corr(v, e)
        out[f"d_mw_vs_{name}"] = corr(v, pw)
    out["ht_vs_n_det"] = corr(ht, nd)
    # multiple regression on standardised predictors: which variable carries the cost?
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    X = np.c_[np.ones(len(y)), z(ht), z(sh), z(nd), z(nc)]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    out["ols_ms_per_sd"] = {"ht": coef[1], "shannon": coef[2], "n_det": coef[3],
                            "n_cand": coef[4],
                            "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    Xd = np.c_[np.ones(len(y)), nd, nc]
    resid = y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]
    out["partial_ht_given_det_cand"] = corr(ht, resid)
    out["partial_shannon_given_det_cand"] = corr(sh, resid)
    zero = nd == 0
    out["zero_detection_subset"] = {
        "n": int(zero.sum()),
        "d_ms_mean": float(y[zero].mean()) if zero.any() else None,
        "d_ms_sd": float(y[zero].std()) if zero.any() else None,
        "d_ms_vs_ht": corr(ht[zero], y[zero]) if zero.sum() > 5 else None,
        "slope_ms_per_0.01HT": (slope_ci(ht[zero] / 0.01, y[zero]) if zero.sum() > 20 else None),
    }
    pos = nd > 0
    if pos.sum() > 20:
        out["with_detections"] = {
            "n": int(pos.sum()), "d_ms_mean": float(y[pos].mean()),
            "step_first_detection_ms": float(y[(nd > 0) & (nd <= 1.5)].mean() - y[zero].mean())
            if ((nd > 0) & (nd <= 1.5)).sum() > 3 and zero.any() else None,
            "slope_ms_per_detection": slope_ci(nd[pos], y[pos]),
            "slope_mj_per_detection": slope_ci(nd[pos], e[pos]),
        }
    out["knee_vs_ht"] = knee_fit(ht, y)
    out["knee_vs_n_det"] = knee_fit(nd, y)
    return out


def paired(path, base="base"):
    """Per-bank difference to the baseline measured in the same round, with 95% t CIs."""
    data = json.load(open(path))
    rows = data["rows"]
    names = list(dict.fromkeys(r["bank"] for r in rows))
    by = {n: sorted([r for r in rows if r["bank"] == n], key=lambda r: r["round"]) for n in names}
    out = {"args": data["args"], "banks": {}}
    keys = ["frame_ms", "vdd_in_mw", "energy_mj"] + STAGES
    for n in names:
        info = {k: by[n][0].get(k) for k in ("ht", "shannon", "n_det", "n_cand", "classes")}
        entry = {"info": info, "n_rounds": len(by[n]),
                 "mean": {k: float(np.mean([r[k] for r in by[n]])) for k in keys}}
        if n != base and base in by:
            for k in keys:
                d = np.array([a[k] - b[k] for a, b in zip(by[n], by[base])])
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                bm = np.mean([b[k] for b in by[base]])
                entry.setdefault("delta", {})[k] = {
                    "mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                    "pct": float(100 * d.mean() / bm) if bm else None,
                    "pct_ci95": [float(100 * (d.mean() - half) / bm), float(100 * (d.mean() + half) / bm)]
                    if bm else None}
        out["banks"][n] = entry
    return out


def main():
    os.makedirs(ROOT, exist_ok=True)
    rows = load_cloud()
    if rows:
        json.dump(cloud_stats(rows), open(f"{ROOT}/cloud_stats.json", "w"), indent=1, default=float)
        by_run = {}
        for r in rows:
            by_run.setdefault(r["run"], []).append(r)
        per = {k: cloud_stats(v) for k, v in by_run.items() if len(v) > 50}
        json.dump(per, open(f"{ROOT}/cloud_stats_per_run.json", "w"), indent=1, default=float)
    for path in sorted(glob.glob(f"{ROOT}/measure/*.json")):
        name = os.path.basename(path)[:-5]
        json.dump(paired(path), open(f"{ROOT}/paired_{name}.json", "w"), indent=1)
        print(name, "done")


if __name__ == "__main__":
    main()
