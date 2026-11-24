"""Frame metrics: the project's exact HT (BIE) and the statistics used for validation.

`radiometric_bie_image`, `shannon_entropy` and `block_image_entropy` reproduce
scripts/experimentA.py of the parent project line for line (tests compare against the
original file when it is present), so HT here equals HT on the Jetson pipeline.
"""
from __future__ import annotations

import numpy as np

K_PER_COUNT = 0.01


def radiometric_bie_image(raw, temp_min_c=10.0, temp_max_c=50.0, k_per_count=K_PER_COUNT):
    temp_c = np.clip(raw.astype(np.float32) * k_per_count - 273.15, temp_min_c, temp_max_c)
    scaled = (temp_c - temp_min_c) * (255.0 / (temp_max_c - temp_min_c))
    return np.rint(scaled).astype(np.uint8)


def shannon_entropy(values, bins=256, value_range=(0, 256)):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    probs = hist.astype(np.float64)
    total = np.sum(probs)
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def block_image_entropy(image, block_size=15, alpha=1.0):
    values = np.clip(np.asarray(image), 0, 255).astype(np.float64)
    dynamic_range = 255.0
    half_range = dynamic_range / 2.0
    block_entropies, block_sds, modulation_entropies = [], [], []
    height, width = values.shape
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            block = values[row:min(row + block_size, height), column:min(column + block_size, width)]
            modulation = (float(np.max(block)) - float(np.min(block))) / dynamic_range
            if 0.0 < modulation < 1.0:
                modulation_entropy = -alpha * (modulation ** alpha) * np.log(modulation)
            else:
                modulation_entropy = 0.0
            modulation_entropies.append(float(modulation_entropy))
            block_entropies.append(shannon_entropy(block))
            block_sds.append(float(np.std(block)))
    mean_intensity = float(np.mean(values))
    adp = float(np.clip(1.0 - abs(mean_intensity - half_range) / half_range, 0.0, 1.0))
    mme = float(np.mean(modulation_entropies))
    mbe = float(np.mean(block_entropies))
    gsd = float(np.std(values))
    mbsd = float(np.mean(block_sds))
    return {"value": float(adp * (mme / (1.0 + mbe)) * (gsd / (1.0 + mbsd))), "adp": adp,
            "mean_modulation_entropy": mme, "mean_block_entropy": mbe, "global_sd": gsd,
            "mean_block_sd": mbsd}


def ht(raw):
    """HT (BIE) of one uint16 TLinear frame with the experiment's fixed 10-50 C mapping."""
    return block_image_entropy(radiometric_bie_image(raw))["value"]


def ht_stack(stack, step=1):
    return np.array([ht(f) for f in stack[::step]])


def radial_power_spectrum(stack_c, n_bins=40):
    """Mean radially averaged spatial power spectrum of mean-removed frames (K^2), vs cyc/px."""
    x = np.asarray(stack_c, np.float64)
    x = x - x.mean(axis=(1, 2), keepdims=True)
    p = np.mean(np.abs(np.fft.fft2(x, axes=(1, 2))) ** 2, axis=0) / (x.shape[1] * x.shape[2])
    fy = np.fft.fftfreq(x.shape[1])[:, None]
    fx = np.fft.fftfreq(x.shape[2])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2).ravel()
    edges = np.linspace(0, 0.5, n_bins + 1)
    which = np.digitize(r, edges) - 1
    out = np.array([p.ravel()[which == i].mean() if np.any(which == i) else np.nan
                    for i in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), out


def temporal_noise_stats(stack_counts):
    """Noise statistics of a (mostly) static clip, after removing the per-frame mean offset.

    Returns K-unit numbers: median per-pixel temporal SD, lag-1/2/5 autocorrelation,
    SD of row means and column means of the residual (stripe noise), frame-mean SD.
    """
    x = np.asarray(stack_counts, np.float64) * K_PER_COUNT
    fm = x.mean(axis=(1, 2))
    r = x - fm[:, None, None]
    r = r - r.mean(0, keepdims=True)
    var = np.mean(r * r)

    def ac(lag):
        return float(np.mean(r[lag:] * r[:-lag]) / var)

    return {"pixel_temporal_sd_k": float(np.median(r.std(0))),
            "lag1": ac(1), "lag2": ac(2), "lag5": ac(5),
            "row_mean_sd_k": float(r.mean(2).std()), "col_mean_sd_k": float(r.mean(1).std()),
            "h_grad_sd_k": float(np.diff(r, axis=2).std()), "v_grad_sd_k": float(np.diff(r, axis=1).std()),
            "frame_mean_sd_k": float(fm.std()),
            "frame_mean_step_sd_k": float(np.diff(fm).std())}
