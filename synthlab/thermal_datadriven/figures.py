"""Figures for docs/thermal-datadriven.md  ->  results/thermal-datadriven/*.png"""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .valsets import CONDS, OUT as SETS
from .validate import RES

OUTD = RES.parent
COL = {"real_test": "#1b6ca8", "real_build": "#7aa6c9", "synth": "#d1495b"}


def contact_sheet():
    conds = list(CONDS)
    fig, ax = plt.subplots(len(conds), 6, figsize=(13, 1.9 * len(conds)))
    for i, c in enumerate(conds):
        rt = np.load(SETS / f"{c}__real_test.npz")["frames"]
        sy = np.load(SETS / f"{c}__synth.npz")["frames"]
        for j, (src, name) in enumerate([(rt, "real"), (sy, "synth")]):
            for k in range(3):
                f = src[(k * 3 + 1) * (len(src) // 9)] * 0.01 - 273.15
                a = ax[i, j * 3 + k]
                a.imshow(f, cmap="inferno", vmin=np.percentile(f, 1), vmax=max(np.percentile(f, 99.8), np.percentile(f, 1) + 3))
                a.set_xticks([]); a.set_yticks([])
                if k == 0:
                    a.set_ylabel(c, fontsize=8)
                if i == 0:
                    a.set_title(f"{name} (held-out)" if name == "real" else "synthetic", fontsize=9)
    fig.suptitle("Held-out real (left 3) vs synthetic (right 3); per-frame colour scale", fontsize=10)
    fig.tight_layout(); fig.savefig(OUTD / "contact_sheet_real_vs_synth.png", dpi=110); plt.close(fig)


def distributions():
    z = np.load(RES / "curves.npz")
    conds = list(CONDS)
    fig, ax = plt.subplots(3, len(conds), figsize=(3.0 * len(conds), 8.2))
    edges = np.arange(-5, 30.0001, 0.05); mid = (edges[1:] + edges[:-1]) / 2
    for i, c in enumerate(conds):
        for kind in ("real_build", "real_test", "synth"):
            h = z[f"{c}__{kind}__hist_rel"]; ax[0, i].semilogy(mid, h / h.sum() + 1e-9, color=COL[kind], label=kind)
            ax[1, i].loglog(z["psd_f"], z[f"{c}__{kind}__psd"], color=COL[kind])
            ax[2, i].hist(z[f"{c}__{kind}__HT"], bins=8, color=COL[kind], alpha=0.5)
        ax[0, i].set_title(c, fontsize=9); ax[0, i].set_xlim(-2, 15); ax[0, i].set_xlabel("pixel - frame median (K)", fontsize=7)
        ax[1, i].set_xlabel("spatial frequency (cyc/px)", fontsize=7); ax[2, i].set_xlabel("HT per clip", fontsize=7)
    ax[0, 0].set_ylabel("pixel histogram"); ax[1, 0].set_ylabel("radial power"); ax[2, 0].set_ylabel("clips")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUTD / "validation_distributions.png", dpi=110); plt.close(fig)


def classifier_bars():
    p = RES / "classifier.json"
    if not p.exists():
        return
    r = json.load(open(p)); conds = list(r)
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(conds)); w = 0.2
    for k, (key, lab, col) in enumerate([("synth_vs_real_test", "synth vs held-out real", COL["synth"]),
                                         ("real_build_vs_real_test", "build real vs held-out real (floor)", COL["real_build"]),
                                         ("synth_vs_real_test_noise_features_only", "synth, noise features only", "#e9a3ad")]):
        ax.bar(x + (k - 1) * w, [r[c][key] for c in conds], w, label=lab, color=col)
    ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, fontsize=8)
    ax.set_ylabel("balanced accuracy (50% = indistinguishable)"); ax.set_ylim(0.4, 1.02); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUTD / "classifier_accuracy.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    contact_sheet(); distributions(); classifier_bars()
