"""Turn evaluation JSON + harness runs into the metric table (markdown/CSV) and the cost-vs-HT figure.

    python3 -m synthlab.thermal_generative.report --ddpm ddpm_main --vae vae_main --fig <dir>
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SESSION_NAMES
from .evaluate import OUT
from .sample import HOME


def load(ck):
    p = f"{OUT}/{ck}_metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def metric_table(models, out_md, out_csv):
    rows = []
    for ck in models:
        m = load(ck)
        if not m:
            continue
        for name, r in m["cells"].items():
            g, t, tr = r["gen"], r["real_test"], r["real_train"]
            rows.append(dict(
                model=ck, cell=f"{SESSION_NAMES[r['session']]} p{r['person']} l{r['lamp']}", n_gen=g["n"], n_real_test=t["n"],
                HT_real_test=t["ht_mean"], HT_real_train=tr["ht_mean"], HT_gen=g["ht_mean"], HT_sd_real_test=t["ht_sd"], HT_sd_gen=g["ht_sd"],
                HT_KS_gen_vs_test=r["ht_ks_gen_vs_test"], HT_KS_train_vs_test=r["ht_ks_train_vs_test"],
                W1_C_gen_vs_test=r["w1_c_gen_vs_test"], W1_C_train_vs_test=r["w1_c_train_vs_test"],
                spec_logdist_gen=r["spec_gen_vs_test"][0], spec_logdist_train=r["spec_train_vs_test"][0],
                HF_power_ratio_gen=r["spec_gen_vs_test"][1], HF_power_ratio_train=r["spec_train_vs_test"][1],
                noise_mK_real_test=t["noise_mk_median"], noise_mK_gen=g["noise_mk_median"],
                yolo_person_rate_real_test=t["person_frame_rate"], yolo_person_rate_gen=g["person_frame_rate"],
                yolo_dets_real_test=t["detections_per_frame"], yolo_dets_gen=g["detections_per_frame"],
                lamp_hot_real_test=t["lamp_hot_fraction"], lamp_hot_gen=g["lamp_hot_fraction"],
                NN_gen_to_train_C=r["nn_rmse_c_median"]["gen_to_train"], NN_test_to_train_C=r["nn_rmse_c_median"]["test_to_train"],
                NN_train_to_train_C=r["nn_rmse_c_median"]["train_to_other_train_trials"],
                frac_gen_closer_than_median_test=r["nn_gen_below_median_of_test"]))
    if not rows:
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    cols = ["model", "cell", "HT_real_test", "HT_gen", "HT_KS_gen_vs_test", "W1_C_gen_vs_test", "HF_power_ratio_gen",
            "noise_mK_real_test", "noise_mK_gen", "yolo_person_rate_real_test", "yolo_person_rate_gen",
            "NN_gen_to_train_C", "NN_test_to_train_C"]
    with open(out_md, "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n")
        for ck in models:
            m = load(ck)
            if m:
                f.write(f"\n{ck}: real-vs-generated classifier {json.dumps(m['real_vs_gen_accuracy'])}\n")
                f.write(f"\n{ck}: condition classifier {json.dumps(m['condition_classifier'])}\n")
                f.write(f"\n{ck}: unseen label combinations {json.dumps(m['novel'])}\n")


def kind_of(bank):
    for k, col in (("real", "#222222"), ("sweep_w", "#7570b3"), ("sweep_t", "#e7298a"), ("UNSEEN", "#e6ab02"),
                   ("vae", "#1b9e77"), ("ddpm", "#d95f02")):
        if k in bank:
            return k, col
    return "other", "#999999"


def fig_cost(csv_path, out):
    if not os.path.exists(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    runs = sorted(set(r["run"] for r in rows))
    ys = [("total_ms_mean", "total per-frame time (ms)"), ("yolo_inf_ms_mean", "YOLO inference (ms)"),
          ("yolo_post_ms_mean", "NMS / post (ms)"), ("power_w", "VDD_IN power (W)")]
    fig, ax = plt.subplots(len(runs), len(ys), figsize=(4.2 * len(ys), 3.6 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        rr = [r for r in rows if r["run"] == run]
        for j, (k, lab) in enumerate(ys):
            seen = set()
            x = np.array([float(r["HT_mean"]) for r in rr]); y = np.array([float(r.get(k) or "nan") for r in rr])
            for r, xi, yi in zip(rr, x, y):
                kd, col = kind_of(r["bank"])
                ax[i, j].scatter(xi, yi, color=col, s=28, label=None if kd in seen else kd)
                seen.add(kd)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2:
                ax[i, j].set_title(f"{run}: r={np.corrcoef(x[ok], y[ok])[0, 1]:.2f} (n={ok.sum()} banks)", fontsize=9)
            ax[i, j].set(xlabel="mean HT of bank (harness)", ylabel=lab); ax[i, j].grid(alpha=0.3)
        ax[i, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm", default="ddpm_main")
    ap.add_argument("--vae", default="vae_main")
    ap.add_argument("--fig", default=f"{HOME}/out/fig")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    metric_table([a.ddpm, a.vae], f"{a.fig}/metric_table.md", f"{a.fig}/metric_table.csv")
    fig_cost(f"{HOME}/out/harness_runs.csv", f"{a.fig}/cost_vs_ht.png")
    print("report written to", a.fig)
