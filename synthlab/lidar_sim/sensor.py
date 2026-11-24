"""Livox Mid-360 sensor model: non-repetitive scan pattern, noise, dropout, glue points, tags.

SCAN PATTERN IS AN APPROXIMATION (documented in docs/lidar-sim.md). The real unit spins an
optical head and steers elevation with a second rotating optic, giving near-vertical scan
lines whose azimuth positions never repeat. We model 4 emitters (the Livox `line` field is
0..3), each sweeping elevation sinusoidally at F_EL while the head turns at F_AZ; the two
frequencies are incommensurate so coverage keeps growing with integration time.
"""
import math
from dataclasses import dataclass, asdict

import numpy as np

from .fog import apply_fog

TAG_GLUE_SHIFT, TAG_PARTICLE_SHIFT = 0, 2      # Livox tag byte: bits 0-1 glue, bits 2-3 particle


@dataclass
class SensorParams:
    points_per_s: int = 200000
    frame_hz: float = 10.0
    el_min_deg: float = -7.0
    el_max_deg: float = 52.0
    n_lines: int = 4
    f_az_hz: float = 9.83            # head rotation (assumed, incommensurate with f_el)
    f_el_hz: float = 131.7           # elevation sweep per emitter (assumed)
    min_range: float = 0.10          # blind zone [m] (spec)
    sigma_r_far: float = 0.02        # range noise 1 sigma at >= ~10 m (spec <= 2 cm)
    sigma_r_near: float = 0.03       # at 0.2 m (spec <= 3 cm)
    sigma_ang_deg: float = 0.15      # total angular 1 sigma (spec < 0.15 deg)
    quant_m: float = 0.001           # packet quantisation 1 mm
    det_rho: float = 0.10            # detection limit: rho = 10 % ...
    det_range: float = 40.0          # ... at 40 m (spec)  -> P_th = det_rho / det_range^2
    det_noise_sigma: float = 0.3     # lognormal sigma on received power (assumed)
    p_drop: float = 0.02             # reflectivity-independent random dropout (assumed)
    glue_div_deg: float = 0.2        # beam footprint used for mixed "glue" points (assumed)
    glue_prob: float = 0.5           # P(mixed range | footprint straddles a depth edge) (assumed)
    p_false_particle_tag: float = 0.0046   # clear-air particle-tag rate (real lab M0: 0.46 %)
    scan_csv: str = ""               # optional recorded pattern (Livox mid360.csv: Time/s,
                                     # Azimuth/deg,Zenith/deg); replaces the analytic pattern

    @property
    def p_thresh(self):
        return self.det_rho / self.det_range ** 2

    @property
    def pts_per_frame(self):
        return int(round(self.points_per_s / self.frame_hz))


_CSV_CACHE = {}


def _csv_dirs(path, n0, n):
    if path not in _CSV_CACHE:
        a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2))
        az, el = np.radians(a[:, 0]), np.radians(90.0 - a[:, 1])
        ce = np.cos(el)
        _CSV_CACHE[path] = np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1)
    d = _CSV_CACHE[path]
    idx = np.arange(n0, n0 + n) % d.shape[0]
    return d[idx], (idx % 4).astype(np.uint8)


def scan_dirs(sp, n0, n):
    """Unit directions (sensor frame) of points n0 .. n0+n-1 of the endless scan."""
    if sp.scan_csv:
        return _csv_dirs(sp.scan_csv, n0, n)
    idx = np.arange(n0, n0 + n, dtype=np.float64)
    t = idx / sp.points_per_s
    line = (idx % sp.n_lines)
    az = 2 * np.pi * (sp.f_az_hz * t + line / sp.n_lines)
    ph = 2 * np.pi * (sp.f_el_hz * t + 0.37 * line)
    el0, el1 = math.radians(sp.el_min_deg), math.radians(sp.el_max_deg)
    el = el0 + (el1 - el0) * (0.5 - 0.5 * np.cos(ph))
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=1), line.astype(np.uint8)


def _jitter(d, sigma_rad, rng):
    """Small random rotation of unit vectors (isotropic, total 1-sigma = sigma_rad)."""
    e = rng.normal(0.0, sigma_rad / math.sqrt(2.0), d.shape)
    e -= np.einsum("ij,ij->i", e, d)[:, None] * d
    out = d + e
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def simulate_frame(scene, frame_idx, rng, sp=None, fog=None, backend="numpy", debug=False):
    """One 100 ms Mid-360 frame. Returns dict(xyz float32 (M,3) sensor frame, intensity uint8,
    tag uint8, and (debug) is_fog bool, n_emitted, n_fog, n_dropped)."""
    sp = sp or SensorParams()
    n = sp.pts_per_frame
    d_s, _ = scan_dirs(sp, frame_idx * n, n)
    d_w = scene.dirs_to_world(d_s)
    r0, refl, cosi, _ = scene.raycast(d_w)

    # glue / mixed points: a second ray inside the beam footprint sees a different depth
    tag = np.zeros(n, np.uint8)
    d2 = _jitter(d_w, math.radians(sp.glue_div_deg), rng)
    r2, _, _, _ = scene.raycast(d2)
    edge = (np.abs(r2 - r0) > 0.05) & (rng.random(n) < sp.glue_prob)
    r0 = np.where(edge, r0 + rng.uniform(0.1, 0.9, n) * (r2 - r0), r0)
    tag[edge] |= (rng.integers(1, 3, edge.sum()).astype(np.uint8) << TAG_GLUE_SHIFT)

    p_clear = refl * cosi / np.maximum(r0, 1e-3) ** 2
    if fog is not None and fog.active:
        f = apply_fog(fog, scene.sensor_pos, d_w, r0, p_clear, sp.p_thresh, rng, backend)
        rr, pw, is_fog = f["range"], f["power"], f["is_fog"]
    else:
        rr, pw, is_fog = r0, p_clear, np.zeros(n, bool)

    # detection: noisy power vs threshold, random dropout, blind zone
    pw_n = pw * rng.lognormal(0.0, sp.det_noise_sigma, n)
    keep = (pw_n >= sp.p_thresh) & (rng.random(n) >= sp.p_drop) & (rr >= sp.min_range)

    # measurement noise
    sig = sp.sigma_r_far + (sp.sigma_r_near - sp.sigma_r_far) * np.exp(-rr / 3.0)
    rr_m = rr + rng.normal(0.0, 1.0, n) * sig
    d_m = _jitter(d_s, math.radians(sp.sigma_ang_deg), rng)
    xyz = d_m * rr_m[:, None]
    xyz = np.round(xyz / sp.quant_m) * sp.quant_m

    # Livox reflectivity byte: 0..150 <-> Lambertian 0..100 %, estimated as P * R^2
    inten = np.clip(np.round(pw * rr ** 2 * 150.0), 0, 150).astype(np.uint8)
    p_tag = fog.p_tag if fog is not None else 0.0
    flag = (is_fog & (rng.random(n) < p_tag)) | (rng.random(n) < sp.p_false_particle_tag)
    tag[flag] |= (rng.integers(1, 3, flag.sum()).astype(np.uint8) << TAG_PARTICLE_SHIFT)

    out = dict(xyz=xyz[keep].astype(np.float32), intensity=inten[keep], tag=tag[keep])
    if debug:
        out.update(is_fog=is_fog[keep], n_emitted=n, n_fog=int((is_fog & keep).sum()),
                   n_dropped=int((~keep).sum()), n_fog_blind=int((is_fog & (rr < sp.min_range)).sum()))
    return out


def sensor_describe(sp):
    return asdict(sp) | {"p_thresh": sp.p_thresh}


def coverage_vs_time(sp, times_s=(0.1, 0.2, 0.5, 1.0, 2.0), cell_deg=0.5):
    """Fraction of az-el cells (cell_deg square) hit at least once vs integration time."""
    out = {}
    naz, nel = int(360 / cell_deg), int((sp.el_max_deg - sp.el_min_deg) / cell_deg)
    for T in times_s:
        d, _ = scan_dirs(sp, 0, int(T * sp.points_per_s))
        az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
        el = np.degrees(np.arcsin(d[:, 2]))
        ia = np.minimum((az / cell_deg).astype(int), naz - 1)
        ie = np.clip(((el - sp.el_min_deg) / cell_deg).astype(int), 0, nel - 1)
        out[float(T)] = float(np.unique(ia * nel + ie).size / (naz * nel))
    return out
