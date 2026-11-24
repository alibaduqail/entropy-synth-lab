"""Reference implementations of the two entropy metrics, for validity checks.

`bie` reproduces the formula in the thermal project's scripts/experimentA.py
(block_image_entropy, Ayunts et al. 2024 Eq. 5) in vectorised numpy.
`eigen_entropies` reproduces the per-point formulas in the LiDAR track's
compute_hl.py (H_DS after dos Santos 2019, and normalised-eigenvalue H_eigen).
Both are re-written here from the published formulas; no teammate code is copied.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def to_u8(temp_c: np.ndarray, lo: float = 10.0, hi: float = 50.0) -> np.ndarray:
    """Fixed 10-50 C mapping to 8 bit used by the thermal pipeline."""
    t = np.clip(np.asarray(temp_c, float), lo, hi)
    return np.rint((t - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _block_entropy_bits(block: np.ndarray) -> float:
    hist = np.bincount(block.astype(np.int64).ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def bie(image_u8: np.ndarray, block: int = 15, alpha: float = 1.0) -> Dict[str, float]:
    """Block image entropy and its four factors.

    BIE = ADP * mean(-a M^a ln M) / (1 + mean block Shannon entropy)
              * global_sd / (1 + mean block sd)
    Note the two denominators: per-block entropy and per-block SD both DIVIDE.
    """
    v = np.clip(np.asarray(image_u8), 0, 255).astype(float)
    h, w = v.shape
    mod, ent, bsd = [], [], []
    for r in range(0, h, block):
        for c in range(0, w, block):
            b = v[r:r + block, c:c + block]
            m = (b.max() - b.min()) / 255.0
            mod.append(-alpha * m ** alpha * np.log(m) if 0.0 < m < 1.0 else 0.0)
            ent.append(_block_entropy_bits(b))
            bsd.append(float(b.std()))
    adp = float(np.clip(1.0 - abs(v.mean() - 127.5) / 127.5, 0.0, 1.0))
    out = {"adp": adp, "mean_modulation_entropy": float(np.mean(mod)),
           "mean_block_entropy": float(np.mean(ent)), "global_sd": float(v.std()),
           "mean_block_sd": float(np.mean(bsd))}
    out["value"] = (adp * out["mean_modulation_entropy"] / (1.0 + out["mean_block_entropy"])
                    * out["global_sd"] / (1.0 + out["mean_block_sd"]))
    return out


def global_shannon_bits(image_u8: np.ndarray) -> float:
    """Plain whole-image histogram entropy (bits), the textbook 'disorder'."""
    return _block_entropy_bits(np.asarray(image_u8))


def eigen_entropies(lam: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-point H_DS and H_eigen from eigenvalues (any order), shape (N,3)."""
    lam = np.sort(np.clip(np.asarray(lam, float), 0, None), axis=1)[:, ::-1]
    l1 = np.clip(lam[:, 0], 1e-12, None)
    L, P, S = (lam[:, 0] - lam[:, 1]) / l1, (lam[:, 1] - lam[:, 2]) / l1, lam[:, 2] / l1

    def plogp(p):
        return np.where(p > 1e-12, p * np.log(np.clip(p, 1e-12, None)), 0.0)

    e = lam / np.clip(lam.sum(1, keepdims=True), 1e-12, None)
    return {"H_DS": -(plogp(L) + plogp(P) + plogp(S)),
            "H_eigen": -(plogp(e[:, 0]) + plogp(e[:, 1]) + plogp(e[:, 2])),
            "L": L, "P": P, "S": S}


def synthetic_thermal_scene(n_people: int = 1, ambient_c: float = 24.0,
                            seed: int = 0, shape=(120, 160)) -> np.ndarray:
    """A deliberately simple noise-free scene (Celsius): flat room + warm blobs.

    Only used to probe how the metric responds to controlled changes; it is not
    a claim about real Lepton imagery.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full(shape, ambient_c, float) + 0.5 * (yy / h)   # mild vertical gradient
    for _ in range(n_people):
        cx, cy = rng.uniform(20, w - 20), rng.uniform(45, h - 35)
        body = (((xx - cx) / 11.0) ** 2 + ((yy - cy) / 30.0) ** 2) < 1.0
        head = (((xx - cx) / 6.0) ** 2 + ((yy - (cy - 36)) / 7.0) ** 2) < 1.0
        img[body] = 30.5
        img[head] = 33.5
    return img
