"""Fixed, invertible mapping between Lepton TLinear counts and model space, plus the
exact HT (block image entropy) and YOLO-input functions of the real pipeline.

TLinear: counts * 0.01 K - 273.15 = deg C.  Model space x:
    u = (T_C - CENTER_C) / SCALE_C                 (affine; 1 count = 0.002 in x)
    x = u                       for |u| <= KNEE
    x = sign(u) * (KNEE + log1p(|u| - KNEE))       beyond the knee (hot lamp pixels)
KNEE corresponds to 50 C, the top of the pipeline's fixed 10-50 C image range, so the
range that feeds BIE/YOLO is purely affine and only out-of-range pixels (the heat lamp,
up to ~123 C) are log-compressed.  The map is strictly monotonic, hence invertible;
round-tripping uint16 counts is exact after rounding (tested).
"""
import numpy as np

K_PER_COUNT = 0.01
CENTER_C = 25.0
SCALE_C = 5.0
KNEE = (50.0 - CENTER_C) / SCALE_C  # = 5.0
TEMP_MIN_C = 10.0
TEMP_MAX_C = 50.0
BIE_BLOCK = 15
H, W = 120, 160


def counts_to_celsius(raw):
    return np.asarray(raw).astype(np.float64) * K_PER_COUNT - 273.15


def celsius_to_counts(temp_c):
    c = np.rint((np.asarray(temp_c, dtype=np.float64) + 273.15) / K_PER_COUNT)
    return np.clip(c, 0, 65535).astype(np.uint16)


def counts_to_x(raw):
    u = (counts_to_celsius(raw) - CENTER_C) / SCALE_C
    a = np.abs(u)
    out = np.where(a <= KNEE, u, np.sign(u) * (KNEE + np.log1p(np.maximum(a - KNEE, 0.0))))
    return out.astype(np.float32)


def x_to_counts(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    u = np.where(a <= KNEE, x, np.sign(x) * (KNEE + np.expm1(np.maximum(a - KNEE, 0.0))))
    return celsius_to_counts(u * SCALE_C + CENTER_C)


def counts_to_x_torch(raw):
    """Same mapping for a torch tensor of counts (any integer/float dtype)."""
    import torch
    u = (raw.float() * K_PER_COUNT - 273.15 - CENTER_C) / SCALE_C
    a = u.abs()
    return torch.where(a <= KNEE, u, torch.sign(u) * (KNEE + torch.log1p((a - KNEE).clamp(min=0))))


def x_to_counts_torch(x):
    import torch
    x = x.double()
    a = x.abs()
    u = torch.where(a <= KNEE, x, torch.sign(x) * (KNEE + torch.expm1((a - KNEE).clamp(min=0))))
    c = torch.round((u * SCALE_C + CENTER_C + 273.15) / K_PER_COUNT)
    return c.clamp(0, 65535)


# ---- the real pipeline's image + entropy functions (ported from scripts/experimentA.py) ----

def bie_image(raw, temp_min_c=TEMP_MIN_C, temp_max_c=TEMP_MAX_C):
    """radiometric_bie_image(): fixed 10-50 C -> uint8. Also the YOLO input (3x repeated)."""
    temp_c = np.clip(np.asarray(raw).astype(np.float32) * K_PER_COUNT - 273.15, temp_min_c, temp_max_c)
    scaled = (temp_c - temp_min_c) * (255.0 / (temp_max_c - temp_min_c))
    return np.rint(scaled).astype(np.uint8)


def _shannon(values):
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    tot = p.sum()
    if tot <= 0:
        return 0.0
    p = p / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def block_image_entropy(image, block_size=BIE_BLOCK, alpha=1.0):
    """BIE of Ayunts et al. 2024, Eq. 5, identical to experimentA.block_image_entropy."""
    values = np.clip(np.asarray(image), 0, 255).astype(np.float64)
    half = 127.5
    ent, sds, mods = [], [], []
    h, w = values.shape
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            b = values[r:min(r + block_size, h), c:min(c + block_size, w)]
            m = (float(b.max()) - float(b.min())) / 255.0
            mods.append(-alpha * (m ** alpha) * np.log(m) if 0.0 < m < 1.0 else 0.0)
            ent.append(_shannon(b))
            sds.append(float(b.std()))
    adp = float(np.clip(1.0 - abs(float(values.mean()) - half) / half, 0.0, 1.0))
    return float(adp * (np.mean(mods) / (1.0 + np.mean(ent))) * (values.std() / (1.0 + np.mean(sds))))


def ht_of_counts(raw):
    """HT for one (120,160) uint16 TLinear frame, as the real pipeline logs it."""
    return block_image_entropy(bie_image(raw))


def yolo_frame(raw):
    g = bie_image(raw)
    return np.repeat(g[:, :, None], 3, axis=2)
