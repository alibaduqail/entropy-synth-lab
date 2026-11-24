"""Figures for the thermal-generative track (run on the Jetson; matplotlib Agg).

    python3 -m synthlab.thermal_generative.plots --ddpm ddpm_main --vae vae_main --fig ~/synthlab/thermal-generative/out/fig
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import SESSION_NAMES, Pack
from .evaluate import NOVEL_CELLS, OUT, REAL_CELLS, cell_name, gen_path
from .mapping import counts_to_celsius
from .sample import HOME

C_REAL, C_TRAIN, C_DDPM, C_VAE = "#222222", "#888888", "#d95f02", "#1b9e77"


def label(c):
    return f"{SESSION_NAMES[c[0]]}\nperson={c[1]} lamp={c[2]}"


def read_log(name):
    steps, chunks = [], []
    p = f"{HOME}/logs/{name}_log.jsonl"
    if not os.path.exists(p):
        return steps, chunks
    for line in open(p):
        r = json.loads(line)
        (chunks if r.get("chunk_end") else steps).append(r)
    return steps, chunks


def fig_training(names, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, col in names:
        steps, chunks = read_log(name)
        if not steps:
            continue
        ax[0].plot([r["train_seconds"] / 60 for r in steps], [r["loss"] for r in steps], color=col, label=f"{name} train loss")
        ax[1].plot([r["train_seconds"] / 60 for r in chunks], [r["eval_train"] for r in chunks], "o-", color=col, label=f"{name} train frames")
        ax[1].plot([r["train_seconds"] / 60 for r in chunks], [r["eval_test"] for r in chunks], "s--", color=col, label=f"{name} held-out trials")
        ax[2].plot([r["train_seconds"] / 60 for r in chunks], [r["mean_vdd_in_w"] for r in chunks], "o-", color=col, label=name)
    ax[0].set(yscale="log", xlabel="training wall time on Orin Nano (min)", ylabel="loss (200-step mean)", title="Training loss")
    ax[1].set(yscale="log", xlabel="training wall time (min)", ylabel="EMA loss at chunk end", title="Train vs held-out-trial loss")
    ax[2].set(xlabel="training wall time (min)", ylabel="mean VDD_IN (W)", title="Jetson power while training")
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_grid(pack, ckpts, out, cells=REAL_CELLS, k=6, with_real=True):
    rows = []
    for c in cells:
        if with_real:
            m = np.where(pack.mask("test", c[1], c[2], c[0]))[0]
            rows.append((f"REAL held-out | {label(c)}", pack.frames[m[np.linspace(0, len(m) - 1, k).astype(int)]]))
        for ck in ckpts:
            p = gen_path(ck, c, "B")
            if os.path.exists(p):
                rows.append((f"{ck} | {label(c)}", np.load(p)[:k]))
    fig, ax = plt.subplots(len(rows), k, figsize=(k * 2.1, len(rows) * 1.62), squeeze=False)
    for i, (name, fr) in enumerate(rows):
        for j in range(k):
            ax[i, j].imshow(np.clip(counts_to_celsius(fr[j]), 20, 38), cmap="inferno", vmin=20, vmax=38)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(name.replace(" | ", "\n"), fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Colour scale fixed 20-38 C (inferno); each row = one condition", fontsize=9)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98)); fig.savefig(out, dpi=120); plt.close(fig)


def fig_quality(ckpts, out):
    cv = {ck: np.load(f"{OUT}/{ck}_curves.npz") for ck, _ in ckpts if os.path.exists(f"{OUT}/{ck}_curves.npz")}
    if not cv:
        return
    first = next(iter(cv.values()))
    cells = [c for c in REAL_CELLS if f"{cell_name(c)}_gen_ht" in first]
    fig, ax = plt.subplots(4, len(cells), figsize=(3.3 * len(cells), 11), squeeze=False)
    tc = np.linspace(15, 60, 451); tc = 0.5 * (tc[1:] + tc[:-1])
    fr = (np.arange(30) + 0.5) / 60
    for j, c in enumerate(cells):
        n = cell_name(c)
        for key, col, lab in (("real_test", C_REAL, "real held-out"), ("real_train", C_TRAIN, "real train")):
            ax[0, j].plot(tc, first[f"{n}_{key}_hist"], color=col, label=lab, lw=1)
            ax[1, j].plot(fr, first[f"{n}_{key}_spec"], color=col, label=lab, lw=1)
        data, cols, labs = [first[f"{n}_real_test_ht"], first[f"{n}_real_train_ht"]], [C_REAL, C_TRAIN], ["real\nheld-out", "real\ntrain"]
        noise, = [[first[f"{n}_real_test_noise"], first[f"{n}_real_train_noise"]]]
        for ck, col in ckpts:
            if ck not in cv:
                continue
            ax[0, j].plot(tc, cv[ck][f"{n}_gen_hist"], color=col, label=ck, lw=1)
            ax[1, j].plot(fr, cv[ck][f"{n}_gen_spec"], color=col, label=ck, lw=1)
            data.append(cv[ck][f"{n}_gen_ht"]); cols.append(col); labs.append(ck.replace("_", "\n"))
            noise.append(cv[ck][f"{n}_gen_noise"])
        for row, d in ((2, data), (3, noise)):
            bp = ax[row, j].boxplot(d, labels=labs, patch_artist=True, showfliers=False)
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col); patch.set_alpha(0.6)
            ax[row, j].tick_params(axis="x", labelsize=6)
        ax[0, j].set(yscale="log", xlim=(18, 55), xlabel="pixel temperature (C)", title=label(c))
        ax[1, j].set(yscale="log", xlabel="spatial frequency (cycles/pixel)")
        ax[2, j].set(xlabel="HT (BIE) per frame")
        ax[3, j].set(xlabel="fine-scale spatial noise (mK)")
        for i in range(4):
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel("pixel fraction"); ax[1, 0].set_ylabel("radial power (K^2)")
    ax[2, 0].set_ylabel("HT"); ax[3, 0].set_ylabel("mK"); ax[0, 0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_memorisation(pack, ck, out, pair_cell=(0, 1, 1), k=6):
    p = f"{OUT}/{ck}_curves.npz"
    if not os.path.exists(p):
        return
    cv = np.load(p)
    cells = [c for c in REAL_CELLS if f"{cell_name(c)}_nn_gen" in cv]
    fig = plt.figure(figsize=(16, 7.5))
    for j, c in enumerate(cells):
        a = fig.add_subplot(2, len(cells), j + 1)
        n = cell_name(c)
        hi = max(np.percentile(cv[f"{n}_nn_test"], 99), np.percentile(cv[f"{n}_nn_gen"], 99))
        bins = np.linspace(0, hi, 30)
        a.hist(cv[f"{n}_nn_test"], bins, color=C_REAL, alpha=0.5, label="real held-out -> train")
        a.hist(cv[f"{n}_nn_train"], bins, color=C_TRAIN, alpha=0.5, label="train -> other train trials")
        a.hist(cv[f"{n}_nn_gen"], bins, color=C_DDPM, alpha=0.6, label="generated -> train")
        a.set(title=label(c), xlabel="RMSE to nearest training frame (C)")
        a.tick_params(labelsize=7)
        if j == 0:
            a.legend(fontsize=6); a.set_ylabel("frames")
    n = cell_name(pair_cell)
    if f"{n}_nn_gen" in cv:
        gen = np.load(gen_path(ck, pair_cell, "B"))
        order = np.argsort(cv[f"{n}_nn_gen"])
        pick = order[np.linspace(0, len(order) - 1, k).astype(int)]
        for j, i in enumerate(pick):
            a = fig.add_subplot(2, k, k + j + 1)
            near = pack.frames[cv[f"{n}_nn_gen_idx"][i]]
            both = np.vstack([counts_to_celsius(gen[i]), counts_to_celsius(near)])
            a.imshow(np.clip(both, 20, 38), cmap="inferno", vmin=20, vmax=38)
            a.set_title(f"top: generated, bottom: nearest train\nRMSE {cv[f'{n}_nn_gen'][i]:.2f} C", fontsize=7)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Memorisation check ({ck}): distance to the nearest training frame", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddpm", default="ddpm_main")
    ap.add_argument("--vae", default="vae_main")
    ap.add_argument("--extra", default="", help="comma-separated extra log names for the training figure")
    ap.add_argument("--fig", default=f"{HOME}/out/fig")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    pack = Pack(f"{HOME}/data/pack.npz")
    names = [(a.ddpm, C_DDPM), (a.vae, C_VAE)] + [(n, "#7570b3") for n in a.extra.split(",") if n]
    fig_training(names, f"{a.fig}/training_curves.png")
    fig_grid(pack, [a.ddpm, a.vae], f"{a.fig}/samples_real_vs_generated.png")
    fig_grid(pack, [a.ddpm], f"{a.fig}/samples_novel_combinations.png", cells=NOVEL_CELLS, k=8, with_real=False)
    fig_quality([(a.ddpm, C_DDPM), (a.vae, C_VAE)], f"{a.fig}/quality_metrics.png")
    fig_memorisation(pack, a.ddpm, f"{a.fig}/memorisation.png")
    print("figures written to", a.fig)
