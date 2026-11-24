"""Vectorised re-implementation of the study's HT (block image entropy, Ayunts 2024) and
global Shannon entropy, for offline analysis. The Jetson rig uses the deployed
experimentA.py functions; tests check this file against the original when it is present."""
from __future__ import annotations

import numpy as np


def shannon_u8(img):
    hist = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def bie(img_u8, block=15, alpha=1.0):
    v = np.clip(np.asarray(img_u8), 0, 255).astype(np.float64)
    h, w = v.shape
    mod_e, blk_e, blk_sd = [], [], []
    for r in range(0, h, block):
        for c in range(0, w, block):
            b = v[r:r + block, c:c + block]
            m = (b.max() - b.min()) / 255.0
            mod_e.append(-alpha * m ** alpha * np.log(m) if 0.0 < m < 1.0 else 0.0)
            blk_e.append(shannon_u8(b))
            blk_sd.append(b.std())
    adp = float(np.clip(1.0 - abs(v.mean() - 127.5) / 127.5, 0.0, 1.0))
    return adp * (np.mean(mod_e) / (1.0 + np.mean(blk_e))) * (v.std() / (1.0 + np.mean(blk_sd)))
