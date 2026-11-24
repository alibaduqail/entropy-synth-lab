"""Summarise stock vs thermal-fine-tuned YOLOv8n harness runs.

usage: python analyze_compare.py <runs dir pulled from the Jetson> <out dir>
Layout read: <runs>/<mode>/<model>/round_RR/<order>_<track>__<bank>/{frames.csv,summary.json}
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = []
for summ in sorted(runs.glob("*/*/round_*/*/summary.json")):
    d = summ.parent
    mode, model, rnd = d.parts[-4], d.parts[-3], int(d.parts[-2][-2:])
    s = json.loads(summ.read_text())
    f = pd.read_csv(d / "frames.csv")
    rows.append({
        "mode": mode, "model": model, "round": rnd, "bank": d.name.split("_", 1)[1],
        "frames": len(f), "ok": s.get("ok"),
        "person_frame_frac": float((f.n_person > 0).mean()),
        "any_det_frame_frac": float((f.n_det > 0).mean()),
        "n_person": f.n_person.mean(), "n_det": f.n_det.mean(),
        "n_candidates": f.n_candidates.mean(), "max_conf": f.max_conf.mean(),
        "yolo_inf_ms": f.yolo_inf_ms.mean(), "yolo_post_ms": f.yolo_post_ms.mean(),
        "total_ms": f.total_ms.mean(), "power_w": s.get("power_w"),
        "energy_per_frame_j": s.get("energy_per_frame_j")})
trials = pd.DataFrame(rows)
trials.to_csv(out / "trials.csv", index=False)
num = [c for c in trials.columns if c not in ("mode", "model", "round", "bank", "ok", "frames")]
bank = trials.groupby(["mode", "bank", "model"])[num].mean().reset_index()
bank.to_csv(out / "by_bank.csv", index=False)

rng = np.random.default_rng(2026)
paired = []
for mode, g in bank.groupby("mode"):
    wide = g.pivot(index="bank", columns="model", values=num)
    for col in num:
        diff = (wide[col]["thermal"] - wide[col]["stock"]).dropna().to_numpy()
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        paired.append({"mode": mode, "metric": col, "banks": len(diff),
                       "stock_mean": wide[col]["stock"].mean(),
                       "thermal_mean": wide[col]["thermal"].mean(),
                       "diff_thermal_minus_stock": diff.mean(),
                       "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
pd.DataFrame(paired).to_csv(out / "paired_diff.csv", index=False)

u = bank[bank["mode"] == "unpaced"].pivot(index="bank", columns="model")
names = [b.split("__")[1] for b in u.index]
x = np.arange(len(names))
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, col, label in zip(axes, ["person_frame_frac", "yolo_post_ms", "total_ms"],
                          ["Frames with a person found", "Post-processing (ms)",
                           "Whole pipeline (ms/frame)"]):
    ax.bar(x - 0.2, u[col]["stock"], 0.4, label="Stock YOLOv8n (COCO)", color="tab:blue")
    ax.bar(x + 0.2, u[col]["thermal"], 0.4, label="Thermal fine-tune (FLIR)", color="tab:red")
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(loc="upper right")
axes[2].set_ylim(bottom=max(0, float(np.nanmin(u["total_ms"].to_numpy())) - 5))
axes[2].set_xticks(x)
axes[2].set_xticklabels(names, rotation=60, ha="right")
fig.tight_layout()
fig.savefig(out / "stock_vs_thermal.png", dpi=150)
print(pd.DataFrame(paired).to_string(index=False))
