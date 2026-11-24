"""Radar "entropy" candidates and scene-stillness metrics.

Entropy candidates (symmetry with HL / HT):
- ``rd_entropy``: Shannon entropy (bits, also normalised 0..1) of the range-Doppler power
  map treated as a probability distribution. Computed on the full map (static scene
  included) and on the map after static clutter removal.
- ``point_entropy``: Shannon entropy of the CFAR point cloud binned on a range x Doppler
  x azimuth grid (0 for an empty cloud).

Stillness metrics (what the "Layer 4 witness" could actually compute):
- ``motion_energy_db``: energy outside zero Doppler relative to the noise floor (needs the
  range-Doppler heat map, available from the TI demos).
- torso phase track: displacement  d(t) = lambda/(4 pi) * unwrap(phase)  of the person's
  range bin after subtracting an empty-room background (needs complex range profiles:
  raw ADC via DCA1000 or the vital-signs lab output).
    ``excursion_mm``  = max-min of the breathing-filtered displacement (4 s moving mean)
    ``net_shift_mm``  = |mean(last 4 s) - mean(first 4 s)|  (4 s = one breath)
    ``lateral_shift_mm`` = same, for the cross-range position from the azimuth phase slope
    ``raw_p2p_mm``    = max-min of the unfiltered displacement (includes breathing)
    ``range_centroid_shift_mm`` = change of the power-weighted range centroid of the
    background-subtracted profile (coarse, but immune to phase wrapping on big moves)
- ``centroid_drift_mm``: largest distance between 1 s point-cloud centroids (needs only
  the default UART point cloud).
- ``scene_decorrelation``: 1 - |coherence| between the first-second and last-second mean
  complex range profiles over bins that do NOT contain the person (did furniture move?).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import RadarConfig


def shannon_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    tot = p.sum()
    if tot <= 0:
        return 0.0
    p = p[p > 0] / tot
    return float(-(p * np.log2(p)).sum())


def rd_entropy(rd_power: np.ndarray) -> Dict[str, float]:
    h = shannon_bits(rd_power)
    return {"bits": h, "norm": h / np.log2(rd_power.size)}


def point_entropy(points: np.ndarray, cfg: RadarConfig, range_bin_m: float = 0.25,
                  n_az: int = 8, n_dop: int = 8) -> float:
    """Entropy (bits) of the point distribution over (range, azimuth, Doppler) cells."""
    if len(points) == 0:
        return 0.0
    r = np.floor(points[:, 5] / range_bin_m).astype(int)
    az = np.arctan2(points[:, 0], points[:, 1])
    a = np.clip(((az + np.pi / 2) / np.pi * n_az).astype(int), 0, n_az - 1)
    d = np.clip(((points[:, 3] + cfg.v_max_mps) / (2 * cfg.v_max_mps) * n_dop).astype(int),
                0, n_dop - 1)
    key = (r * n_az + a) * n_dop + d
    _, counts = np.unique(key, return_counts=True)
    return shannon_bits(counts)


def frame_entropies(res, cfg: RadarConfig) -> Dict[str, float]:
    e_tot = rd_entropy(res.rd_total)
    e_dyn = rd_entropy(res.rd_dyn)
    return {"h_rd_total_bits": e_tot["bits"], "h_rd_total_norm": e_tot["norm"],
            "h_rd_dyn_bits": e_dyn["bits"], "h_rd_dyn_norm": e_dyn["norm"],
            "h_points_bits": point_entropy(res.points, cfg)}


def motion_energy_db(rd_dyn: np.ndarray) -> float:
    """Total clutter-removed energy over the median (noise) level, in dB."""
    noise = np.median(rd_dyn) * rd_dyn.size
    return float(10 * np.log10(rd_dyn.sum() / max(noise, 1e-300)))


class StillnessRecorder:
    """Accumulates per-frame outputs of a trial and computes the stillness metrics."""

    def __init__(self, cfg: RadarConfig, background: Optional[np.ndarray] = None,
                 az_flat: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.background = background        # empty-room zero-Doppler profile (range, n_virtual)
        self.az_flat = az_flat              # Processor.az_flat (enables the lateral metric)
        self.zd: List[np.ndarray] = []
        self.zd_sub: List[Optional[np.ndarray]] = []
        self.motion_db: List[float] = []
        self.centroids: List[np.ndarray] = []
        self.n_det: List[int] = []

    def add(self, res) -> None:
        self.zd.append(res.zero_doppler)
        self.zd_sub.append(res.zd_sub)
        self.motion_db.append(motion_energy_db(res.rd_dyn))
        self.n_det.append(res.n_det)
        self.centroids.append(res.points[:, :3].mean(axis=0) if len(res.points)
                              else np.full(3, np.nan))

    # ------------------------------------------------------------------
    def displacement_mm(self, range_gate=(0.4, 6.0)) -> Dict[str, np.ndarray]:
        """Radial displacement track of the person's range bin.

        Uses the per-frame block means (``zd_sub``, 8 blocks of 12 chirps) when available:
        the phase is unwrapped inside each frame (blocks are 3.2 ms apart) and the jump
        across the inter-frame gap is resolved with the velocity measured inside the two
        frames, so the track survives speeds far above the lambda/4-per-frame limit
        (~22 mm/s) of naive frame-to-frame unwrapping."""
        cfg = self.cfg
        zd = np.array(self.zd)                                   # (T, K, V)
        bgd = self.background if self.background is not None else zd.mean(axis=0)
        x = zd - bgd[None]
        rax = cfg.range_axis()
        gate = (rax >= range_gate[0]) & (rax <= range_gate[1])
        n1 = max(1, int(round(1.0 / cfg.frame_period_s)))
        power = (np.abs(x[:n1]) ** 2).sum(axis=2).mean(axis=0)   # first second: where is she?
        power[~gate] = 0
        k = int(np.argmax(power))
        w = np.conj(x[:n1, k, :].mean(axis=0))                   # spatial matched filter
        s = x[:, k, :] @ w
        if self.zd_sub and self.zd_sub[0] is not None:
            xs = np.array([z[:, k, :] for z in self.zd_sub]) - bgd[None, None, k, :]
            ss = xs @ w                                          # (T, B)
            B = ss.shape[1]
            dt_b = (cfg.n_loops // B) * cfg.tx_repeat_s
            gap = cfg.frame_period_s - (B - 1) * dt_b
            ph = np.unwrap(np.angle(ss), axis=1)                 # inside each frame
            slope = np.polyfit(np.arange(B) * dt_b, ph.T, 1)[0]  # rad/s per frame
            out = np.zeros_like(ph)
            out[0] = ph[0]
            for t in range(1, len(ph)):
                pred = 0.5 * (slope[t - 1] + slope[t]) * gap
                d = np.angle(ss[t, 0] * np.conj(ss[t - 1, -1]))
                d += 2 * np.pi * np.round((pred - d) / (2 * np.pi))
                out[t] = ph[t] - ph[t, 0] + out[t - 1, -1] + d
            phase = out.mean(axis=1)
        else:
            phase = np.unwrap(np.angle(s))
        d = cfg.wavelength_m / (4 * np.pi) * (phase - phase[0]) * 1e3
        near = np.abs(rax - rax[k]) <= 0.5          # +-0.5 m around the person only
        pw = (np.abs(x) ** 2).sum(axis=2)[:, near]
        cen = (pw * rax[near][None]).sum(axis=1) / np.maximum(pw.sum(axis=1), 1e-300)
        res = {"d_mm": d, "bin": k, "range_m": float(rax[k]), "person_power": power,
               "range_centroid_m": cen}
        if self.az_flat is not None:
            a = x[:, k, :][:, self.az_flat]                      # (T, 8) azimuth row
            ux = -np.angle((a[:, 1:] * np.conj(a[:, :-1])).sum(axis=1)) / np.pi
            res["lateral_mm"] = ux * rax[k] * 1e3
        return res

    def metrics(self, breath_window_s: float = 4.0, edge_s: float = 4.0) -> Dict[str, float]:
        cfg = self.cfg
        fps = 1.0 / cfg.frame_period_s
        out: Dict[str, float] = {}
        disp = self.displacement_mm()
        d = disp["d_mm"]
        n = max(1, int(round(breath_window_s * fps)))
        if len(d) >= 2 * n:
            filt = np.convolve(d, np.ones(n) / n, mode="valid")
        else:
            filt = d
        ne = max(1, int(round(edge_s * fps)))
        out["torso_range_m"] = disp["range_m"]
        out["raw_p2p_mm"] = float(d.max() - d.min())
        out["excursion_mm"] = float(filt.max() - filt.min())
        out["net_shift_mm"] = float(abs(np.mean(d[-ne:]) - np.mean(d[:ne])))
        if "lateral_mm" in disp:
            lat = disp["lateral_mm"]
            out["lateral_shift_mm"] = float(abs(np.mean(lat[-ne:]) - np.mean(lat[:ne])))
        else:
            out["lateral_shift_mm"] = float("nan")
        cen = disp["range_centroid_m"]
        out["range_centroid_shift_mm"] = float(abs(np.median(cen[-ne:]) - np.median(cen[:ne])) * 1e3)
        m = np.array(self.motion_db)
        out["motion_db_median"] = float(np.median(m))
        out["motion_db_max"] = float(m.max())
        out["n_det_mean"] = float(np.mean(self.n_det))
        # point-cloud centroid drift over 1 s windows
        c = np.array(self.centroids)
        nw = max(1, int(round(fps)))
        wins = [np.nanmean(c[i:i + nw], axis=0) for i in range(0, len(c) - nw + 1, nw)
                if np.isfinite(c[i:i + nw, 0]).any()]
        if len(wins) >= 2:
            w = np.array(wins)
            out["centroid_drift_mm"] = float(np.linalg.norm(w - w[0], axis=1).max() * 1e3)
        else:
            out["centroid_drift_mm"] = float("nan")
        out["frames_with_points_frac"] = float(np.isfinite(c[:, 0]).mean())
        # coherence of the non-person bins between the first and last second
        zd = np.array(self.zd)
        a, b = zd[:nw].mean(axis=0), zd[-nw:].mean(axis=0)
        rax = cfg.range_axis()
        keep = np.abs(rax - disp["range_m"]) > 0.35
        keep &= rax > 0.3
        num = np.abs((a[keep] * np.conj(b[keep])).sum())
        den = np.sqrt((np.abs(a[keep]) ** 2).sum() * (np.abs(b[keep]) ** 2).sum())
        out["scene_decorrelation"] = float(1.0 - num / max(den, 1e-300))
        return out
