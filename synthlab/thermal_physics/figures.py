"""Figures for docs/thermal-physics.md. Every plotted number is read from results/thermal-physics/.

  python -m synthlab.thermal_physics.figures validation   # contact sheet, histograms, PSDs, HT
  python -m synthlab.thermal_physics.figures fog          # HT vs fog density / mist offset
  python -m synthlab.thermal_physics.figures cost         # Jetson cost vs HT
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

RES = "results/thermal-physics"
STACKS = "data_local/thermal-physics/validation"
C_REAL, C_SIM = "#1f77b4", "#d62728"
ORDER = ["flat_wall_A0", "empty_room_A0", "two_seated_A1", "standing_person_A1", "person_holding_lamp_A1"]


def _c(x):
    return x.astype(np.float64) / 100.0 - 273.15


def contact_sheet():
    names = [n for n in ORDER if os.path.exists(f"{STACKS}/sim_{n}.npy")]
    fig, ax = plt.subplots(len(names), 6, figsize=(15, 2.25 * len(names)))
    for r, n in enumerate(names):
        real, sim = np.load(f"{STACKS}/real_{n}.npy"), np.load(f"{STACKS}/sim_{n}.npy")
        picks_r = [int(len(real) * q) for q in (0.15, 0.5, 0.85)]
        picks_s = [int(len(sim) * q) for q in (0.15, 0.5, 0.85)]
        frames = [_c(real[i]) for i in picks_r] + [_c(sim[i]) for i in picks_s]
        lo = min(np.percentile(f, 1) for f in frames)
        hi = min(max(np.percentile(f, 99.8) for f in frames), lo + 16)
        for c, f in enumerate(frames):
            im = ax[r, c].imshow(f, cmap="inferno", vmin=lo, vmax=hi)
            ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            ax[r, c].set_title(("REAL " if c < 3 else "SIMULATED ") + n if c in (0, 3) else "", fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax[r, :], fraction=0.012, pad=0.01)
        cb.set_label("deg C", fontsize=8)
    fig.suptitle("Real Lepton 3.5 frames (left 3) vs physically-based simulation (right 3), shared colour scale per row",
                 fontsize=11, y=0.995)
    fig.savefig(f"{RES}/contact_sheet_real_vs_sim.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def validation_curves():
    val = json.load(open(f"{RES}/validation.json"))
    names = [n for n in ORDER if n in val]
    fig, ax = plt.subplots(3, len(names), figsize=(3.6 * len(names), 9))
    edges = np.arange(15, 60, 0.1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    for k, n in enumerate(names):
        cv = np.load(f"{RES}/validation_curves_{n}.npz")
        v = val[n]
        hr, hs = cv["hist_real"] / cv["hist_real"].sum(), cv["hist_sim"] / cv["hist_sim"].sum()
        nz = np.nonzero((hr > 1e-6) | (hs > 1e-6))[0]
        ax[0, k].semilogy(cen, hr + 1e-7, color=C_REAL, label="real")
        ax[0, k].semilogy(cen, hs + 1e-7, color=C_SIM, label="sim")
        ax[0, k].set_xlim(cen[nz[0]] - 0.5, min(cen[nz[-1]] + 0.5, 45))
        ax[0, k].set_ylim(1e-6, 1)
        ax[0, k].set_title(f"{n}\nhistogram, W1 = {v['hist_w1_k']:.2f} K", fontsize=9)
        ax[0, k].set_xlabel("deg C")
        ax[1, k].loglog(cv["freq"], cv["psd_real"], color=C_REAL, label="real")
        ax[1, k].loglog(cv["freq"], cv["psd_sim"], color=C_SIM, label="sim")
        ax[1, k].set_title(f"radial power spectrum\nRMS log10 ratio = {v['radial_psd_rms_log10_ratio']:.2f}", fontsize=9)
        ax[1, k].set_xlabel("cycles / pixel")
        parts = ax[2, k].violinplot([cv["ht_real"], cv["ht_sim"]], showmeans=True)
        for body, col in zip(parts["bodies"], (C_REAL, C_SIM)):
            body.set_facecolor(col)
        ax[2, k].set_xticks([1, 2])
        ax[2, k].set_xticklabels(["real", "sim"])
        ax[2, k].set_title(f"HT (BIE): real {v['ht']['real_mean']:.4f}, sim {v['ht']['sim_mean']:.4f}", fontsize=9)
        ax[2, k].set_ylim(0, None)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel("pixel fraction")
    ax[1, 0].set_ylabel("power (counts^2)")
    ax[2, 0].set_ylabel("HT per frame")
    fig.tight_layout()
    fig.savefig(f"{RES}/validation_hist_psd_ht.png", dpi=110)
    plt.close(fig)
    # temporal noise spectrum (static pairs)
    stat = [n for n in names if "sensor_real" in val[n]]
    fig, ax = plt.subplots(1, len(stat), figsize=(5 * len(stat), 3.8), squeeze=False)
    for k, n in enumerate(stat):
        cv = np.load(f"{RES}/validation_curves_{n}.npz")
        if "tfreq_real" in cv:
            ax[0, k].loglog(cv["tfreq_real"][1:], cv["tpsd_real"][1:], color=C_REAL, label="real")
        if "tfreq_sim" in cv:
            ax[0, k].loglog(cv["tfreq_sim"][1:], cv["tpsd_sim"][1:], color=C_SIM, label="sim")
        ax[0, k].set_title(f"temporal noise spectrum per pixel - {n}", fontsize=10)
        ax[0, k].set_xlabel("Hz")
        ax[0, k].set_ylabel("counts^2 / Hz")
        ax[0, k].legend()
    fig.tight_layout()
    fig.savefig(f"{RES}/validation_temporal_psd.png", dpi=110)
    plt.close(fig)


def fog():
    d = json.load(open(f"{RES}/fog_sweep.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(d["offsets_k"])))
    for scene_name, a in zip(("person", "empty"), ax[:2]):
        for off, col in zip(d["offsets_k"], cols):
            rows = [r for r in d["rows"] if r["scene"] == scene_name and r["mist_offset_k"] == off and not r["plume"]]
            rows.sort(key=lambda r: r["beta_per_m"])
            a.errorbar([r["beta_per_m"] for r in rows], [r["ht_mean"] for r in rows], yerr=[r["ht_sd"] for r in rows],
                       color=col, marker="o", capsize=2, label=f"mist {off:+.1f} K vs air")
        for lvl, b in d["fog_levels"].items():
            a.axvline(b, color="0.8", lw=0.8, zorder=0)
            a.text(b, a.get_ylim()[1], f"B{lvl}", fontsize=8, ha="center", va="bottom")
        a.set_xlabel("LWIR extinction beta (1/m)   [LWC = beta / 0.14 g/m^3]")
        a.set_ylabel("HT (BIE)")
        a.set_title(f"HT vs fog density - {'one person at the mark' if scene_name == 'person' else 'empty room'}", fontsize=10)
        a.set_ylim(0, None)
    ax[0].legend(fontsize=8)
    a = ax[2]
    for scene_name, ls in (("person", "-"), ("empty", "--")):
        rows = [r for r in d["rows"] if r["scene"] == scene_name and r["mist_offset_k"] == 0.0 and not r["plume"]]
        rows.sort(key=lambda r: r["beta_per_m"])
        base = rows[0]["ht_mean"]
        a.plot([r["beta_per_m"] for r in rows], [100 * (r["ht_mean"] / base - 1) for r in rows], ls, marker="o",
               color="k", label=f"{scene_name}, uniform fog, mist = air")
        pr = [r for r in d["rows"] if r["scene"] == scene_name and r["plume"]]
        pr.sort(key=lambda r: r["beta_per_m"])
        if pr:
            a.plot([r["beta_per_m"] for r in pr], [100 * (r["ht_mean"] / base - 1) for r in pr], ls, marker="s",
                   color="tab:orange", label=f"{scene_name}, uniform + turbulent plume, mist -1 K")
    a.axhline(0, color="0.6", lw=0.8)
    a.set_xlabel("LWIR extinction beta (1/m)")
    a.set_ylabel("HT change vs no fog (%)")
    a.set_title("Hypothesis B check: is HT flat in fog?", fontsize=10)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{RES}/ht_vs_fog.png", dpi=110)
    plt.close(fig)


def cost():
    d = json.load(open(f"{RES}/jetson_bank_runs.json"))
    rows = [r for r in d["rows"] if r.get("mode") == "unpaced"] or d["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    fam = {"A": "tab:red", "B": "tab:blue", "C": "tab:green", "L": "tab:orange", "R": "k", "r": "k"}
    for r in rows:
        c = fam.get(r["bank"][0], "0.4")
        for a, key in zip(ax, ("total_ms_mean", "yolo_inf_ms_mean", "power_w")):
            if r.get(key) is None:
                continue
            a.scatter(r["ht_mean"], r[key], color=c, s=28)
            a.annotate(r["bank"].split("_")[0], (r["ht_mean"], r[key]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for a, lab in zip(ax, ("total per-frame time (ms)", "YOLOv8n inference (ms)", "VDD_IN power (W)")):
        a.set_xlabel("HT (BIE) of the bank")
        a.set_ylabel(lab)
    x = np.array([r["ht_mean"] for r in rows])
    for a, key in zip(ax, ("total_ms_mean", "yolo_inf_ms_mean", "power_w")):
        y = np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)
        ok = np.isfinite(y)
        if ok.sum() > 2:
            a.set_title(f"Pearson r = {np.corrcoef(x[ok], y[ok])[0, 1]:+.2f}  (n = {ok.sum()} banks, {rows[0].get('mode', '')})", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
               (("A0-A3 people", "tab:red"), ("B0-B3 fog", "tab:blue"), ("C00-C16 warm objects", "tab:green"),
                ("lamp off/on", "tab:orange"), ("real recordings", "k"))]
    ax[0].legend(handles=handles, fontsize=7)
    fig.suptitle("Jetson Orin Nano: cost vs thermal entropy for physically-based synthetic banks", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{RES}/cost_vs_ht.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "validation"
    if what == "validation":
        contact_sheet()
        validation_curves()
    elif what == "fog":
        fog()
    elif what == "cost":
        cost()
