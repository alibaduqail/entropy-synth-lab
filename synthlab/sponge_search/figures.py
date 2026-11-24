"""Figures for the sponge search (static PNGs, light surface, validated default palette)."""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import ROOT, STAGES, col, load_cloud

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#b7d3f6", "#5598e7", "#256abf", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 10, "savefig.dpi": 160})
STAGE_LABEL = {"bie_ms": "BIE (HT)", "prep_ms": "3-ch copy", "pre_ms": "YOLO preprocess",
               "inf_ms": "YOLO inference", "post_ms": "YOLO postprocess (NMS)",
               "overhead_ms": "result summary", "log_ms": "CSV row"}
HT_BAND = (0.011, 0.020)


def progress(rows_all, out):
    runs = [r for r in dict.fromkeys(x["run"] for x in rows_all)]
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        rr = [x for x in rows_all if x["run"] == run]
        timed = "d_ms" in rr[0]
        y = col(rr, "d_ms") if timed else col(rr, "n_det")
        ax.scatter(np.arange(len(y)), y, s=5, color=C[0], alpha=0.35, linewidths=0,
                   label="each candidate")
        ax.plot(np.maximum.accumulate(y), color=C[1], lw=2, label="best so far")
        ax.set_title(f"{run}\n({rr[0]['mode']}, {rr[0]['tier']})")
        ax.set_xlabel("evaluation #")
        ax.set_ylabel("extra ms/frame vs empty room" if timed else "detections (proxy)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def cloud(rows, out):
    ht, y, e, nd, nc, sh = (col(rows, k) for k in ("ht", "d_ms", "d_mj", "n_det", "n_cand",
                                                    "shannon"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    vmax = max(1.0, float(np.percentile(nd, 99)))
    panels = [(axes[0, 0], ht, y, "thermal entropy HT (BIE)", "extra ms/frame"),
              (axes[0, 1], nd, y, "detections per frame", "extra ms/frame"),
              (axes[1, 0], ht, e, "thermal entropy HT (BIE)", "extra mJ/frame"),
              (axes[1, 1], sh, y, "Shannon entropy of 8-bit frame (bits)", "extra ms/frame")]
    for ax, x, v, xl, yl in panels:
        sc = ax.scatter(x, v, c=nd, cmap=SEQ, vmin=0, vmax=vmax, s=7, alpha=0.7, linewidths=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axhline(0, color=INK2, lw=0.8)
        if "HT" in xl:
            ax.axvspan(*HT_BAND, color=C[3], alpha=0.25, lw=0)
            ax.text(HT_BAND[1], ax.get_ylim()[1], " real empty-room HT band", va="top",
                    fontsize=7, color=INK2)
    fig.colorbar(sc, ax=axes, label="detections per frame", shrink=0.7)
    fig.suptitle(f"Every hardware-in-the-loop evaluation (n={len(rows)}): cost vs entropy, "
                 "relative to the same-generation empty-room baseline", fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def stage_breakdown(paired_path, out, order=None, title=""):
    data = json.load(open(paired_path))["banks"]
    names = order or list(data)
    names = [n for n in names if n in data]
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(names), 4.2))
    bottom = np.zeros(len(names))
    for i, s in enumerate(STAGES):
        vals = np.array([data[n]["mean"][s] for n in names])
        ax.bar(names, vals, bottom=bottom, color=C[i], label=STAGE_LABEL[s], width=0.62,
               edgecolor=SURF, linewidth=1.5)
        bottom += vals
    for i, n in enumerate(names):
        ax.text(i, bottom[i] + 0.5, f"{bottom[i]:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("ms per frame (mean)")
    ax.set_title(title or "Where the time goes")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def scenes_grid(npz_path, out, paired_path=None, order=None):
    z = np.load(npz_path)
    info = json.load(open(paired_path))["banks"] if paired_path else {}
    names = [n for n in (order or z.files) if n in z.files]
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        temp = z[n].astype(np.float32) * 0.01 - 273.15
        im = ax.imshow(temp, cmap="inferno", vmin=10, vmax=50)
        t = n
        if n in info:
            b = info[n]
            d = b.get("delta", {}).get("frame_ms")
            t += f"\nHT {b['info']['ht']:.3f}, {b['info']['n_det']:.0f} det, {b['info']['n_cand']:.0f} cand"
            if d:
                t += f"\n{d['mean']:+.1f} ms/frame ({d['pct']:+.1f}%)"
        ax.set_title(t, fontsize=8)
    fig.colorbar(im, ax=axes, label="temperature (C), pipeline range 10-50 C", shrink=0.6)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    rows_all = load_cloud(timed_only=False)
    rows = [r for r in rows_all if "d_ms" in r]
    if rows_all:
        progress(rows_all, f"{ROOT}/search_progress.png")
    if rows:
        cloud(rows, f"{ROOT}/cost_vs_entropy_cloud.png")
    fin = f"{ROOT}/paired_finalists.json"
    if os.path.exists(fin):
        stage_breakdown(fin, f"{ROOT}/stage_breakdown.png")
        if os.path.exists(f"{ROOT}/scenes_preview.npz"):
            scenes_grid(f"{ROOT}/scenes_preview.npz", f"{ROOT}/best_scenes.png", fin)


if __name__ == "__main__":
    main()
