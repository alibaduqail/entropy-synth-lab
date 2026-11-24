"""HT = block image entropy (BIE, Ayunts et al. 2024, doi 10.3390/e26050374).

The functions between the BEGIN/END markers are copied VERBATIM from the parent
project's `scripts/experimentA.py` (the real Experiment A pipeline) so that HT
numbers from synthetic frames are directly comparable with the real recordings.
`tests/test_thermal_physics_bie.py` checks them against the original file when
it is present on disk. Do not edit the copied block.
"""
import numpy as np

DEFAULT_TLINEAR_K_PER_COUNT = 0.01
DEFAULT_BIE_BLOCK_SIZE = 15
DEFAULT_TEMP_MIN_C = 10.0
DEFAULT_TEMP_MAX_C = 50.0


def require_numpy():
    return np


def tlinear_resolution_control_value(k_per_count: float):
    if abs(k_per_count - 0.1) < 1e-12:
        return 0
    if abs(k_per_count - 0.01) < 1e-12:
        return 1
    raise ValueError("TLinear resolution must be 0.1 or 0.01 K/count")


# ---- BEGIN verbatim copy from scripts/experimentA.py ----
def normalize_to_u8(image, low_percentile=1.0, high_percentile=99.0):
    numpy = require_numpy()
    values = numpy.asarray(image).astype(numpy.float32)
    finite = values[numpy.isfinite(values)]
    if finite.size == 0:
        return numpy.zeros(values.shape, dtype=numpy.uint8)

    low = float(numpy.percentile(finite, low_percentile))
    high = float(numpy.percentile(finite, high_percentile))
    if high <= low:
        low = float(numpy.min(finite))
        high = float(numpy.max(finite))
    if high <= low:
        return numpy.zeros(values.shape, dtype=numpy.uint8)

    scaled = (numpy.clip(values, low, high) - low) * (255.0 / (high - low))
    return scaled.astype(numpy.uint8)



def raw_to_celsius(raw, k_per_count=DEFAULT_TLINEAR_K_PER_COUNT):
    """
    Convert verified TLinear data to Celsius.

    Requires radiometry and TLinear enabled, AGC disabled on the raw stream,
    and a fixed resolution matching ``k_per_count``. This function does not
    query hardware; the runner verifies that state before capture by default.
    """
    numpy = require_numpy()
    tlinear_resolution_control_value(k_per_count)
    return raw.astype(numpy.float32) * k_per_count - 273.15


def shannon_entropy(values, bins=256, value_range=(0, 256)):
    """Shannon entropy helper used for each local BIE block."""
    numpy = require_numpy()
    hist, _ = numpy.histogram(values, bins=bins, range=value_range)
    probs = hist.astype(numpy.float64)
    total = numpy.sum(probs)
    if total <= 0:
        return 0.0

    probs = probs / total
    probs = probs[probs > 0]
    return float(-numpy.sum(probs * numpy.log2(probs)))


def radiometric_bie_image(
    raw,
    temp_min_c=DEFAULT_TEMP_MIN_C,
    temp_max_c=DEFAULT_TEMP_MAX_C,
    k_per_count=DEFAULT_TLINEAR_K_PER_COUNT,
):
    """
    Convert calibrated radiometric data to a reproducible 8-bit BIE image.

    The fixed Celsius range avoids per-frame auto-normalization, which would
    otherwise make entropy values incomparable between frames and trials.
    """
    if temp_max_c <= temp_min_c:
        raise ValueError("temp_max_c must be greater than temp_min_c")

    numpy = require_numpy()
    temp_c = numpy.clip(
        raw_to_celsius(raw, k_per_count),
        temp_min_c,
        temp_max_c,
    )
    scaled = (temp_c - temp_min_c) * (255.0 / (temp_max_c - temp_min_c))
    return numpy.rint(scaled).astype(numpy.uint8)


def block_image_entropy(image, block_size=DEFAULT_BIE_BLOCK_SIZE, alpha=1.0):
    """
    Compute Equation 5's block-wise image entropy (BIE) from Ayunts et al.

    BIE combines:
      * ADP, which rewards a balanced global mean;
      * modified block modulation, -alpha * M'^alpha * ln(M');
      * mean per-block Shannon entropy;
      * global and mean per-block standard deviation.

    The returned dictionary exposes every component so the recorded HT value
    can be audited later. Border blocks are retained instead of discarding
    thermal pixels when a dimension is not divisible by ``block_size``.
    """
    if block_size < 1:
        raise ValueError("block_size must be 1 or greater")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")

    numpy = require_numpy()
    values = numpy.asarray(image)
    if values.ndim != 2:
        raise ValueError("BIE requires a two-dimensional grayscale image")
    if values.size == 0:
        raise ValueError("BIE requires a non-empty image")

    values = numpy.clip(values, 0, 255).astype(numpy.float64)
    dynamic_range = 255.0
    half_range = dynamic_range / 2.0

    block_entropies = []
    block_standard_deviations = []
    modulation_entropies = []

    height, width = values.shape
    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            block = values[
                row : min(row + block_size, height),
                column : min(column + block_size, width),
            ]
            block_min = float(numpy.min(block))
            block_max = float(numpy.max(block))
            modulation = (block_max - block_min) / dynamic_range

            # The limit of -M ln(M) is 0 as M approaches 0; it is also 0 at M=1.
            if 0.0 < modulation < 1.0:
                modulation_entropy = -alpha * (modulation**alpha) * numpy.log(modulation)
            else:
                modulation_entropy = 0.0

            modulation_entropies.append(float(modulation_entropy))
            block_entropies.append(shannon_entropy(block))
            block_standard_deviations.append(float(numpy.std(block)))

    mean_intensity = float(numpy.mean(values))
    adp = 1.0 - abs(mean_intensity - half_range) / half_range
    adp = float(numpy.clip(adp, 0.0, 1.0))
    mean_modulation_entropy = float(numpy.mean(modulation_entropies))
    mean_block_entropy = float(numpy.mean(block_entropies))
    global_sd = float(numpy.std(values))
    mean_block_sd = float(numpy.mean(block_standard_deviations))

    bie = (
        adp
        * (mean_modulation_entropy / (1.0 + mean_block_entropy))
        * (global_sd / (1.0 + mean_block_sd))
    )

    return {
        "value": float(bie),
        "block_size": int(block_size),
        "block_count": len(block_entropies),
        "adp": adp,
        "mean_modulation_entropy": mean_modulation_entropy,
        "mean_block_entropy": mean_block_entropy,
        "global_sd": global_sd,
        "mean_block_sd": mean_block_sd,
    }
# ---- END verbatim copy ----


def ht_of_raw(raw, block_size=DEFAULT_BIE_BLOCK_SIZE):
    """HT of one TLinear uint16 frame, exactly as experimentA.py logs it."""
    return block_image_entropy(radiometric_bie_image(raw), block_size=block_size)["value"]


def ht_series(frames, block_size=DEFAULT_BIE_BLOCK_SIZE):
    return np.array([ht_of_raw(f, block_size) for f in frames])
