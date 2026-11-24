"""Out-of-box-demo-style processing chain with per-stage timing.

range FFT -> static clutter removal -> Doppler FFT -> non-coherent detection matrix ->
CA-CFAR (range and Doppler) + peak grouping -> per-point angle estimation (TDM Doppler
compensation, azimuth FFT, elevation monopulse) -> point cloud -> DBSCAN -> tracker.

FIXED-cost stages (same arithmetic whatever the scene): range_fft, clutter_removal,
doppler_fft, detection_matrix, cfar (the threshold comparison runs over every cell).
DATA-DEPENDENT stages (work scales with the number of detections): angle, cluster, track.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import RadarConfig, virtual_array_index

FIXED_STAGES = ["range_fft", "clutter_removal", "doppler_fft", "detection_matrix", "cfar"]
DATA_STAGES = ["angle", "cluster", "track"]
STAGES = FIXED_STAGES + DATA_STAGES


def _side_sum(x: np.ndarray, lo: int, hi: int, axis: int, wrap: bool) -> np.ndarray:
    """Sum of cells at offsets lo..hi (inclusive, may be negative) along ``axis``."""
    x = np.moveaxis(x, axis, -1)
    pad = max(abs(lo), abs(hi))
    if wrap:
        xp = np.concatenate([x[..., -pad:], x, x[..., :pad]], axis=-1)
    else:
        xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    cs = np.cumsum(xp, axis=-1)
    cs = np.concatenate([np.zeros_like(cs[..., :1]), cs], axis=-1)
    n = x.shape[-1]
    out = cs[..., pad + hi + 1: pad + hi + 1 + n] - cs[..., pad + lo: pad + lo + n]
    return np.moveaxis(out, -1, axis)


def ca_cfar(power: np.ndarray, axis: int, guard: int, train: int, thr_db: float,
            wrap: bool, mode: str = "caso"):
    """Cell-averaging CFAR along one axis with ``train`` cells each side.
    mode "ca" averages both sides, "caso" takes the smaller side (less self-masking by
    extended targets), "cago" the greater. Cost is the same for every input."""
    left = _side_sum(power, -(guard + train), -(guard + 1), axis, wrap) / train
    right = _side_sum(power, guard + 1, guard + train, axis, wrap) / train
    if mode == "ca":
        noise = 0.5 * (left + right)
    elif mode == "cago":
        noise = np.maximum(left, right)
    else:
        noise = np.minimum(left, right)
    return power > noise * 10 ** (thr_db / 10.0), noise


def dbscan(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Plain O(n^2) DBSCAN; returns labels (-1 = noise)."""
    n = len(pts)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    neigh = d2 <= eps * eps
    core = neigh.sum(axis=1) >= min_pts
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.nonzero(neigh[j] & (labels == -1))[0]:
                labels[k] = cid
                stack.append(k)
        cid += 1
    return labels


@dataclass
class Track:
    tid: int
    pos: np.ndarray
    vel: np.ndarray
    hits: int = 1
    misses: int = 0


class Tracker:
    """Nearest-neighbour alpha-beta tracker on cluster centroids (GTRACK stand-in)."""

    def __init__(self, dt: float, gate_m: float = 0.6, alpha: float = 0.5, beta: float = 0.1,
                 max_misses: int = 10):
        self.dt, self.gate, self.alpha, self.beta = dt, gate_m, alpha, beta
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next = 0

    def update(self, centroids: np.ndarray) -> List[Track]:
        used = np.zeros(len(centroids), dtype=bool)
        for tr in self.tracks:
            pred = tr.pos + tr.vel * self.dt
            best, bd = -1, self.gate
            for j, c in enumerate(centroids):
                if used[j]:
                    continue
                d = float(np.linalg.norm(c - pred))
                if d < bd:
                    best, bd = j, d
            if best >= 0:
                res = centroids[best] - pred
                tr.pos = pred + self.alpha * res
                tr.vel = tr.vel + self.beta / self.dt * res
                tr.hits += 1
                tr.misses = 0
                used[best] = True
            else:
                tr.pos = pred
                tr.misses += 1
        for j, c in enumerate(centroids):
            if not used[j]:
                self.tracks.append(Track(self._next, np.array(c, dtype=float), np.zeros(3)))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks


@dataclass
class FrameResult:
    points: np.ndarray            # (n, 6): x, y, z (radar frame, m), radial velocity, snr_db, range
    rd_total: np.ndarray          # (doppler, range) power incl. static scene
    rd_dyn: np.ndarray            # (doppler, range) power after static clutter removal
    zero_doppler: np.ndarray      # (range, n_virtual) complex mean over chirps
    n_det: int
    n_clusters: int
    n_tracks: int
    timings_ms: Dict[str, float] = field(default_factory=dict)
    zd_sub: Optional[np.ndarray] = None   # (n_sub, range, n_virtual): block means over chirps


class Processor:
    def __init__(self, cfg: RadarConfig, cfar_thr_db: float = 15.0, cfar_guard: int = 2,
                 cfar_train: int = 8, doppler_thr_db: float = 12.0, n_az_fft: int = 64,
                 max_points: int = 1200, cluster_eps_m: float = 0.35, cluster_min_pts: int = 3,
                 min_range_m: float = 0.25, keep_maps: bool = True,
                 cfar_mode: str = "caso", group_doppler: bool = False,
                 n_sub: int = 0):
        self.cfg = cfg
        self.cfar_thr_db, self.cfar_guard, self.cfar_train = cfar_thr_db, cfar_guard, cfar_train
        self.doppler_thr_db = doppler_thr_db
        self.cfar_mode, self.group_doppler = cfar_mode, group_doppler
        self.n_sub = n_sub
        self.n_az_fft = n_az_fft
        self.max_points = max_points
        self.cluster_eps_m, self.cluster_min_pts = cluster_eps_m, cluster_min_pts
        self.keep_maps = keep_maps
        self.win_r = np.hanning(cfg.n_samples + 2)[1:-1].astype(np.float32)
        self.win_d = np.hanning(cfg.n_loops + 2)[1:-1].astype(np.float32)
        self.range_axis = cfg.range_axis()
        self.doppler_axis = cfg.doppler_axis()
        self.min_bin = int(np.ceil(min_range_m / (cfg.range_max_m / cfg.n_samples)))
        va = virtual_array_index(cfg)
        self.az_idx, self.el_idx = va["az"], va["el"]
        slot = np.arange(cfg.n_tx)[:, None] * np.ones((1, cfg.n_rx), dtype=int)
        row0 = self.el_idx == 0
        order = np.argsort(self.az_idx[row0])
        self.az_slots = slot[row0][order]            # slot of each azimuth-row element
        self.az_rx = (np.ones((cfg.n_tx, 1), dtype=int) * np.arange(cfg.n_rx))[row0][order]
        self.az_pos = self.az_idx[row0][order]
        row1 = self.el_idx == 1
        order1 = np.argsort(self.az_idx[row1])
        self.el_slots = slot[row1][order1]
        self.el_rx = (np.ones((cfg.n_tx, 1), dtype=int) * np.arange(cfg.n_rx))[row1][order1]
        self.el_pos = self.az_idx[row1][order1]
        self.tracker = Tracker(cfg.frame_period_s)
        # flattened (slot*n_rx + rx) indices of the 8-element azimuth row, in azimuth order
        self.az_flat = self.az_slots * cfg.n_rx + self.az_rx

    # ------------------------------------------------------------------
    def process(self, cube: np.ndarray) -> FrameResult:
        cfg = self.cfg
        tm: Dict[str, float] = {}
        t0 = time.perf_counter()
        rf = np.fft.fft(cube * self.win_r, axis=-1)                         # [L, M, N, K]
        t1 = time.perf_counter(); tm["range_fft"] = (t1 - t0) * 1e3
        mean = rf.mean(axis=0)
        dyn = rf - mean
        t2 = time.perf_counter(); tm["clutter_removal"] = (t2 - t1) * 1e3
        dop = np.fft.fftshift(np.fft.fft(dyn * self.win_d[:, None, None, None], axis=0), axes=0)
        t3 = time.perf_counter(); tm["doppler_fft"] = (t3 - t2) * 1e3
        det = (dop.real ** 2 + dop.imag ** 2).sum(axis=(1, 2))              # [D, K]
        t4 = time.perf_counter(); tm["detection_matrix"] = (t4 - t3) * 1e3
        m_r, noise = ca_cfar(det, 1, self.cfar_guard, self.cfar_train, self.cfar_thr_db,
                             False, self.cfar_mode)
        m_d, _ = ca_cfar(det, 0, self.cfar_guard, self.cfar_train, self.doppler_thr_db,
                         True, self.cfar_mode)
        mask = m_r & m_d
        # peak grouping along range (fixed cost); optional along Doppler
        mask &= det >= np.roll(det, 1, axis=1)
        mask &= det >= np.roll(det, -1, axis=1)
        if self.group_doppler:
            mask &= det >= np.roll(det, 1, axis=0)
            mask &= det >= np.roll(det, -1, axis=0)
        mask[:, :self.min_bin] = False
        mask[:, -2:] = False
        di, ki = np.nonzero(mask)
        if len(di) > self.max_points:
            keep = np.argsort(det[di, ki])[::-1][:self.max_points]
            di, ki = di[keep], ki[keep]
        t5 = time.perf_counter(); tm["cfar"] = (t5 - t4) * 1e3
        points = self._angles(dop, det, noise, di, ki)
        t6 = time.perf_counter(); tm["angle"] = (t6 - t5) * 1e3
        labels = dbscan(points[:, :3], self.cluster_eps_m, self.cluster_min_pts)
        ncl = int(labels.max() + 1) if len(labels) else 0
        cents = np.array([points[labels == c, :3].mean(axis=0) for c in range(ncl)]).reshape(-1, 3)
        t7 = time.perf_counter(); tm["cluster"] = (t7 - t6) * 1e3
        tracks = self.tracker.update(cents)
        t8 = time.perf_counter(); tm["track"] = (t8 - t7) * 1e3
        tm["total"] = (t8 - t0) * 1e3
        tm["fixed"] = sum(tm[s] for s in FIXED_STAGES)
        tm["data_dependent"] = sum(tm[s] for s in DATA_STAGES)
        if self.keep_maps:
            # diagnostics outside the timed chain: full map including the static scene
            full = np.fft.fftshift(np.fft.fft(rf * self.win_d[:, None, None, None], axis=0), axes=0)
            rd_total = (full.real ** 2 + full.imag ** 2).sum(axis=(1, 2))
        else:
            rd_total = det
        zd = mean.reshape(cfg.n_tx * cfg.n_rx, cfg.n_samples).T
        zd_sub = None
        if self.n_sub:
            # diagnostics for the phase-based stillness metric (outside the timed chain)
            blk = cfg.n_loops // self.n_sub
            zs = rf[:blk * self.n_sub].reshape(self.n_sub, blk, cfg.n_tx * cfg.n_rx,
                                               cfg.n_samples).mean(axis=1)
            zd_sub = zs.transpose(0, 2, 1)
        return FrameResult(points, rd_total, det, zd, len(di), ncl, len(tracks), tm, zd_sub)

    def _angles(self, dop, det, noise, di, ki) -> np.ndarray:
        cfg = self.cfg
        n = len(di)
        if n == 0:
            return np.zeros((0, 6))
        x = dop[di, :, :, ki]                                               # (n, M, N)
        dbin = di - cfg.n_loops // 2
        # TDM-MIMO Doppler compensation: slot m is sampled m*Tc later
        comp = np.exp(-2j * np.pi * dbin[:, None] * np.arange(cfg.n_tx)[None, :]
                      / (cfg.n_loops * cfg.n_tx))
        x = x * comp[:, :, None]
        az_row = x[:, self.az_slots, self.az_rx]                            # (n, 8)
        el_row = x[:, self.el_slots, self.el_rx]                            # (n, 4)
        spec = np.fft.fft(az_row, n=self.n_az_fft, axis=1)
        pk = np.argmax(np.abs(spec), axis=1)
        f = np.where(pk >= self.n_az_fft // 2, pk - self.n_az_fft, pk) / float(self.n_az_fft)
        ux = -2.0 * f                                    # phase across azimuth = -pi*idx*ux
        sel = (self.az_pos[None, :] >= self.el_pos.min()) & (self.az_pos[None, :] <= self.el_pos.max())
        az_sub = az_row[:, sel[0]]
        wz = np.angle((el_row * np.conj(az_sub)).sum(axis=1))
        uz = np.clip(-wz / np.pi, -1, 1)
        ux = np.clip(ux, -1, 1)
        r = self.range_axis[ki]
        uy = np.sqrt(np.clip(1.0 - ux ** 2 - uz ** 2, 0.0, 1.0))
        v = self.doppler_axis[di]
        snr = 10 * np.log10(np.maximum(det[di, ki].astype(float), 1e-300)
                            / np.maximum(noise[di, ki].astype(float), 1e-300))
        return np.stack([r * ux, r * uy, r * uz, v, snr, r], axis=1)
