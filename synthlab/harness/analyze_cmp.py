#!/usr/bin/env python3
"""Step-3 analysis: one randomised-block comparison of every bank from every track.

  python analyze_cmp.py --runs data_local/runs --out results/jetson-harness

Expects <runs>/cmp/thermal/cCC/round_BB/NN_<track>__<bank>/ and <runs>/cmp/lidar/... (round =
block; every bank appears once per block) and optionally <runs>/noise/ (dedicated noise floor).
Model per metric (trial level):  y = bank effect + block effect + b_T * chip_temp + error.
The residual SD of that model is the in-situ noise floor; MDE uses it. Slopes against entropy
are fitted on block- and temperature-adjusted bank means, with a bank-level bootstrap CI.
Writes cmp_*.csv, cmp_verdicts.json and figures fig_cmp_*.png.
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

VDD_STEP_W = 0.0395   # tegrastats VDD_IN resolution (~39.5 mW per step)
T_METRICS = ["total_ms", "bie_ms", "convert_ms", "yolo_pre_ms", "yolo_inf_ms", "yolo_post_ms",
             "total_no_bie_ms", "power_w", "energy_per_unit_j"]
L_METRICS = ["total_ms", "entropy_total_ms", "knn_ms", "eig_ms", "voxel_ms", "cluster_total_ms",
             "power_w", "energy_per_unit_j"]
STAGES = ["convert_ms", "bie_ms", "yolo_pre_ms", "yolo_inf_ms", "yolo_post_ms"]


def approach(row):
    if row.track == "jetson-harness":
        b = row.bank
        return "real" if b.startswith("real_") else "control" if b.startswith("ctl_") else "mvp"
    return row.track


def fog_of(manifest):
    """Fog level from a bank manifest, whatever the generator called it (0 if none)."""
    for key in ("fog_level", "fog_alpha"):
        if isinstance(manifest.get(key), (int, float)):
            return float(manifest[key])
    fog = manifest.get("fog")
    if isinstance(fog, dict) and isinstance(fog.get("alpha"), (int, float)):
        return float(fog["alpha"])
    params = manifest.get("parameters")
    if isinstance(params, dict):
        return fog_of(params)
    frames = manifest.get("frames")
    if isinstance(frames, list) and frames and isinstance(frames[0], dict):
        vals = [f.get("fog_level", f.get("fog_alpha")) for f in frames]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            return float(np.mean(vals))
    return 0.0


def load(runs, kind):
    root = Path(runs) / "cmp" / kind
    if not root.exists():
        return pd.DataFrame()
    t = c.collect_trials(root)
    if not len(t):
        return t
    t["block"] = t["round"]
    t["approach"] = t.apply(approach, axis=1)
    t["bank_id"] = t.track + "__" + t.bank
    fogs, conds = [], []
    for d in t.dir:
        p = Path(d) / "bank_manifest.json"
        m = json.loads(p.read_text()) if p.exists() else {}
        fogs.append(fog_of(m))
        conds.append(str(m.get("condition", "")))
    t["fog"], t["condition"] = fogs, conds
    t["entropy"] = t["HT"] if kind == "thermal" else t["H_DS"]
    return t


def adjust(t, metric):
    """Fit y ~ bank + block + temp; return adjusted bank means, residual SD, temp slope."""
    d = t[["bank_id", "block", "temp_tj_c", metric]].dropna()
    banks, blocks = sorted(d.bank_id.unique()), sorted(d.block.unique())
    X = [np.ones(len(d))]
    X += [(d.bank_id == b).astype(float).values for b in banks[1:]]
    X += [(d.block == k).astype(float).values for k in blocks[1:]]
    temp = d.temp_tj_c.values - d.temp_tj_c.mean()
    X.append(temp)
    X = np.column_stack(X)
    y = d[metric].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - np.linalg.matrix_rank(X))
    sd = float(np.sqrt(resid @ resid / dof))
    nb = len(banks)
    bank_eff = np.r_[0.0, beta[1:nb]]
    block_mean = np.r_[0.0, beta[nb:nb + len(blocks) - 1]].mean()
    means = beta[0] + bank_eff + block_mean
    return pd.Series(means, index=banks), sd, float(beta[-1])


def mde(sd, n, z=2.80):
    return z * sd * np.sqrt(2.0 / n)


def slope_ci(x, y, n=4000, seed=1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or x.std() == 0:
        return np.nan, (np.nan, np.nan)
    b = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, x.size, x.size)
        if x[i].std() > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    return b, tuple(float(v) for v in np.percentile(bs, [2.5, 97.5]))


def hinge_test(x, y):
    """Linear vs one-knee piecewise-linear fit (knee on a grid); BIC difference (>6 = knee)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 8:
        return {"n": n, "knee": None, "dBIC_linear_minus_hinge": np.nan}
    rss_lin = np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2)
    best = (np.inf, None)
    for k in np.quantile(x, np.linspace(0.2, 0.8, 25)):
        X = np.column_stack([np.ones(n), x, np.clip(x - k, 0, None)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ beta) ** 2)
        if rss < best[0]:
            best = (rss, float(k))
    bic = lambda rss, p: n * np.log(max(rss, 1e-12) / n) + p * np.log(n)  # noqa: E731
    return {"n": n, "knee": best[1],
            "dBIC_linear_minus_hinge": float(bic(rss_lin, 2) - bic(best[0], 4))}


def bank_table(t, metrics, extra):
    rows = {}
    stats = {}
    for m in metrics:
        if m not in t or t[m].isna().all():
            continue
        means, sd, tslope = adjust(t, m)
        rows[m] = means
        stats[m] = {"resid_sd": sd, "temp_slope_per_C": tslope}
    tab = pd.DataFrame(rows)
    info = t.groupby("bank_id").agg(
        approach=("approach", "first"), bank=("bank", "first"), n_blocks=("block", "nunique"),
        fog=("fog", "first"), condition=("condition", "first"),
        temp_tj_c=("temp_tj_c", "mean"), all_ok=("ok", "all"),
        **{k: (k, "mean") for k in extra if k in t})
    return info.join(tab), stats


def verdict_rows(tab, stats, xcol, metrics, n_blocks, label, controls=()):
    out = []
    for m in metrics:
        if m not in tab:
            continue
        sd = stats[m]["resid_sd"]
        b, ci = slope_ci(tab[xcol], tab[m])
        xr = float(tab[xcol].max() - tab[xcol].min())
        row = {"set": label, "x": xcol, "metric": m, "n_banks": len(tab), "n_blocks": n_blocks,
               "slope_per_unit": b, "slope_lo": ci[0], "slope_hi": ci[1],
               "effect_over_range": b * xr if np.isfinite(b) else np.nan,
               "x_range": xr, "noise_sd_trial": sd, "mde_two_banks": mde(sd, n_blocks),
               "bank_spread": float(tab[m].max() - tab[m].min()),
               "temp_slope_per_C": stats[m]["temp_slope_per_C"]}
        if np.isfinite(b):
            row["r"] = float(np.corrcoef(tab[xcol], tab[m])[0, 1])
        ctl = [k for k in controls if k in tab and tab[k].std() > 0]
        if ctl:
            beta, se, r2 = c.ols(tab[m], [tab[xcol]] + [tab[k] for k in ctl])
            row.update(slope_controlled=float(beta[1]), se_controlled=float(se[1]),
                       controls="+".join(ctl), r2_full=r2)
        row.update({f"hinge_{k}": v for k, v in hinge_test(tab[xcol], tab[m]).items()})
        if m == "power_w":
            row["vdd_step_w"] = VDD_STEP_W
        out.append(row)
    return out


def fig_scatter(tab, stats, xcol, metrics, labels, out, title, n_blocks):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4))
    cmap = plt.get_cmap("tab10")
    apps = sorted(tab.approach.unique())
    for ax, m in zip(np.atleast_1d(axes), metrics):
        if m not in tab:
            continue
        scale = 1e3 if m == "energy_per_unit_j" else 1.0
        y = tab[m] * scale
        centre = float(np.median(y))
        band = 1.96 * stats[m]["resid_sd"] * scale / np.sqrt(n_blocks)
        ax.axhspan(centre - band, centre + band, color="0.85", zorder=0,
                   label="noise floor (bank mean ±1.96 SD/√blocks)")
        for i, a in enumerate(apps):
            s = tab[tab.approach == a]
            ax.scatter(s[xcol], s[m] * scale, s=28, color=cmap(i % 10), label=a, zorder=3)
        ax.set_xlabel(xcol)
        ax.set_ylabel(labels.get(m, m))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_stages(tab, out):
    s = tab.sort_values("HT")
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(s)), 4.2))
    bottom = np.zeros(len(s))
    for st in [k for k in STAGES if k in s]:
        ax.bar(np.arange(len(s)), s[st], bottom=bottom, label=st.replace("_ms", ""))
        bottom += s[st].values
    ax.set_xticks(np.arange(len(s)))
    ax.set_xticklabels([f"{a[:6]}:{b[:14]}" for a, b in zip(s.approach, s.bank)],
                       rotation=90, fontsize=6)
    ax.set_ylabel("ms / frame (block-adjusted)")
    ax.set_title("Thermal per-frame cost by stage, banks sorted by HT (left low, right high)")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def residual_vs_ht(tab, metric):
    """metric residual after regressing on any-detection fraction and candidate count."""
    ctl = [k for k in ("any_det_frac", "n_candidates") if k in tab and tab[k].std() > 0]
    beta, _, _ = c.ols(tab[metric], [tab[k] for k in ctl])
    X = np.column_stack([np.ones(len(tab))] + [tab[k].values for k in ctl])
    return tab[metric].values - X @ beta + tab[metric].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    verdicts, summary = [], {}
    labels = {"total_ms": "ms / frame", "power_w": "VDD_IN W", "energy_per_unit_j": "mJ / unit",
              "total_no_bie_ms": "ms / frame without BIE", "entropy_total_ms": "HL stage ms / window"}

    th = load(args.runs, "thermal")
    if len(th):
        th.drop(columns=["dir"]).to_csv(args.out / "cmp_thermal_trials.csv", index=False)
        nb = int(th.block.nunique())
        tab, stats = bank_table(th, T_METRICS,
                                ["entropy", "HT", "n_det", "n_candidates", "any_det_frac"])
        tab.to_csv(args.out / "cmp_thermal_banks.csv")
        controls = ("any_det_frac", "n_candidates")
        verdicts += verdict_rows(tab, stats, "HT", T_METRICS, nb, "thermal_all", controls)
        for a, g in tab.groupby("approach"):
            if len(g) >= 4:
                verdicts += verdict_rows(g, stats, "HT", ["total_ms", "power_w"], nb,
                                         f"thermal_{a}", controls)
        tab["total_ms_ctl_det"] = residual_vs_ht(tab, "total_ms")
        tab["total_no_bie_ms_ctl_det"] = residual_vs_ht(tab, "total_no_bie_ms")
        stats["total_ms_ctl_det"] = stats["total_ms"]
        stats["total_no_bie_ms_ctl_det"] = stats["total_no_bie_ms"]
        labels.update(total_ms_ctl_det="ms / frame, detections regressed out",
                      total_no_bie_ms_ctl_det="no-BIE ms, detections regressed out")
        fig_scatter(tab, stats, "HT", ["total_ms", "power_w", "energy_per_unit_j"], labels,
                    args.out / "fig_cmp_thermal_cost_vs_HT.png",
                    f"Thermal: block-adjusted bank means, {len(tab)} banks x {nb} blocks "
                    "(YOLOv8n PyTorch FP32, unpaced)", nb)
        fig_scatter(tab, stats, "HT", ["total_ms_ctl_det", "total_no_bie_ms_ctl_det", "yolo_post_ms"],
                    labels, args.out / "fig_cmp_thermal_cost_vs_HT_controlled.png",
                    "Thermal cost vs HT after regressing out any-detection fraction and "
                    "pre-NMS candidate count", nb)
        fig_stages(tab, args.out / "fig_cmp_thermal_stages.png")
        summary["thermal"] = {"n_trials": int(len(th)), "n_ok": int(th.ok.sum()), "n_banks": len(tab),
                              "n_blocks": nb, "temp_tj_range_c": float(th.temp_tj_c.max() - th.temp_tj_c.min()),
                              "problems": sorted(set(p for p in th.problems if p))[:10],
                              "noise": {m: s for m, s in stats.items() if not m.endswith("_det")}}

    li = load(args.runs, "lidar")
    if len(li):
        li.drop(columns=["dir"]).to_csv(args.out / "cmp_lidar_trials.csv", index=False)
        nb = int(li.block.nunique())
        tab, stats = bank_table(li, L_METRICS, ["entropy", "H_DS", "H_eigen", "n_voxel",
                                                "n_roi", "frac_tag_particle", "n_clusters"])
        tab = tab.rename(columns={"H_DS": "HL"})
        tab.to_csv(args.out / "cmp_lidar_banks.csv")
        verdicts += verdict_rows(tab, stats, "HL", L_METRICS, nb, "lidar_all", ("n_voxel",))
        fig_scatter(tab, stats, "HL", ["total_ms", "power_w", "energy_per_unit_j"], labels,
                    args.out / "fig_cmp_lidar_cost_vs_HL.png",
                    f"LiDAR: block-adjusted bank means, {len(tab)} banks x {nb} blocks "
                    "(HL entropy + clustering, unpaced)", nb)
        # Hyp B, LiDAR side: fog banks vs no-fog banks of the same family
        fog = tab[tab.fog > 0]
        clear = tab[tab.fog == 0]
        hb = {}
        if len(fog) and len(clear):
            for m in ["HL", "total_ms", "power_w", "energy_per_unit_j"]:
                d = float(fog[m].mean() - clear[m].mean())
                hb[m] = {"fog_mean": float(fog[m].mean()), "clear_mean": float(clear[m].mean()),
                         "diff": d}
                if m in stats:
                    hb[m]["mde_two_banks"] = mde(stats[m]["resid_sd"], nb)
        summary["lidar"] = {"n_trials": int(len(li)), "n_ok": int(li.ok.sum()), "n_banks": len(tab),
                            "n_blocks": nb, "hyp_b_fog_vs_clear": hb,
                            "noise": stats}

    v = pd.DataFrame(verdicts)
    v.to_csv(args.out / "cmp_verdicts.csv", index=False)
    summary["vdd_in_step_w"] = VDD_STEP_W
    (args.out / "cmp_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    cols = ["set", "metric", "n_banks", "slope_per_unit", "slope_lo", "slope_hi",
            "effect_over_range", "mde_two_banks", "bank_spread", "slope_controlled",
            "hinge_dBIC_linear_minus_hinge"]
    pd.set_option("display.width", 250, "display.max_columns", 20)
    if len(v):
        print(v[[k for k in cols if k in v]].round(4).to_string())


if __name__ == "__main__":
    main()
