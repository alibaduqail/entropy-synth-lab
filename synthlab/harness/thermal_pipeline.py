"""The real thermal processing steps, copied unchanged in behaviour from the original
project's scripts/experimentA.py (raw_to_celsius, radiometric_bie_image, shannon_entropy,
block_image_entropy, yolo_frame_from_thermal). Only the lazy-import helpers were removed.
Keep this file in step with the original: the harness must time the same arithmetic.
"""

import numpy as np

TLINEAR_K_PER_COUNT = 0.01
TEMP_MIN_C = 10.0
TEMP_MAX_C = 50.0
BIE_BLOCK_SIZE = 15
BIE_ALPHA = 1.0


def raw_to_celsius(raw, k_per_count=TLINEAR_K_PER_COUNT):
    return raw.astype(np.float32) * k_per_count - 273.15


def radiometric_bie_image(raw, temp_min_c=TEMP_MIN_C, temp_max_c=TEMP_MAX_C,
                          k_per_count=TLINEAR_K_PER_COUNT):
    """Fixed-range Celsius -> 8-bit mapping (no per-frame auto-normalisation)."""
    if temp_max_c <= temp_min_c:
        raise ValueError("temp_max_c must be greater than temp_min_c")
    temp_c = np.clip(raw_to_celsius(raw, k_per_count), temp_min_c, temp_max_c)
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


def block_image_entropy(image, block_size=BIE_BLOCK_SIZE, alpha=BIE_ALPHA):
    """Block image entropy (Ayunts et al. 2024, Eq. 5) with every component exposed."""
    if block_size < 1:
        raise ValueError("block_size must be 1 or greater")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("BIE requires a non-empty two-dimensional grayscale image")
    values = np.clip(values, 0, 255).astype(np.float64)
    dynamic_range = 255.0
    half_range = dynamic_range / 2.0
    block_entropies, block_sds, modulation_entropies = [], [], []
    height, width = values.shape
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            block = values[row:min(row + block_size, height),
                           column:min(column + block_size, width)]
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
    mean_modulation_entropy = float(np.mean(modulation_entropies))
    mean_block_entropy = float(np.mean(block_entropies))
    global_sd = float(np.std(values))
    mean_block_sd = float(np.mean(block_sds))
    bie = (adp * (mean_modulation_entropy / (1.0 + mean_block_entropy))
           * (global_sd / (1.0 + mean_block_sd)))
    return {"value": float(bie), "block_size": int(block_size),
            "block_count": len(block_entropies), "adp": adp,
            "mean_modulation_entropy": mean_modulation_entropy,
            "mean_block_entropy": mean_block_entropy,
            "global_sd": global_sd, "mean_block_sd": mean_block_sd}


def yolo_frame_from_thermal(gray):
    """Replicate the 8-bit thermal image into three channels for YOLO."""
    values = np.asarray(gray)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ValueError("Thermal YOLO input must be a two-dimensional uint8 image")
    return np.repeat(values[:, :, np.newaxis], 3, axis=2)


def validate_bank(bank):
    """A thermal bank is uint16 (N,120,160) TLinear counts."""
    if bank.ndim != 3 or bank.shape[1:] != (120, 160) or bank.dtype != np.uint16:
        raise ValueError(f"bank must be uint16 (N,120,160); got {bank.dtype} {bank.shape}")
    if bank.shape[0] < 1:
        raise ValueError("bank is empty")
    return bank
