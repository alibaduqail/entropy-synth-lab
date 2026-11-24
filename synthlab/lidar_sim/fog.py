"""Physically-motivated fog model for a pulsed ToF LiDAR (Rasshofer 2011 / Hahner ICCV 2021).

Normalised power units: a Lambertian hard target of reflectivity rho at range R0 seen at
incidence cos_i returns  P_hard_clear = rho * cos_i / R0^2.
In fog with extinction alpha(r) [1/m] along the ray:
    T2(r)   = exp(-2 * int_0^r alpha ds)                       two-way transmission
    P_hard  = P_hard_clear * T2(R0)                            attenuated hard return
    S(r)    = pi * beta(r) * xi(r) * T2(r) / r^2 , r < R0      fog impulse response (per metre)
    W(R)    = int S(r) * sin^2(pi (R - r) / (c tau_H)) dr      soft return waveform (pulse conv.)
with beta = (0.046 / ln 20) * alpha  (Rasshofer: beta = 0.046 / MOR, alpha = ln(20) / MOR) and
xi the receiver overlap ("crossover") function, a linear ramp 0 -> 1 between r1 and r2.
The factor pi converts the Lambertian convention (beta_0 = rho / pi in Hahner) to ours.
Return choice per ray:
    "strongest" (Hahner): soft replaces hard if peak(W) > P_hard.
    "first": the earliest echo above threshold wins -> soft wins whenever it is detectable.
A soft echo is only "detectable" if it passes the far-field power threshold AND the near-field
gate rho_gate on apparent reflectivity P*R^2 (stands in for the unknown near-range sensitivity
control of the real receiver; without it any trace of haze would fill target-less rays).
Either way the surviving echo must exceed the detection threshold or the point is dropped
(Mid-360 is single-return: a fog point REPLACES the surface point).
Deviations from Hahner, all flagged in docs/lidar-sim.md: (1) the soft range is DRAWN from the
impulse response S(r) (pdf prop. to S) instead of argmax(W) + ad-hoc 2^U(-1,1) noise; (2) gamma speckle on the
soft power; (3) alpha may vary in space (plume) so T2 is integrated numerically per ray; (4) rho_gate.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

C_LIGHT = 299792458.0
BETA_OVER_ALPHA = 0.046 / math.log(20.0)      # 0.01536 sr^-1


def mor_to_alpha(mor_m):
    return math.log(20.0) / mor_m


def alpha_to_mor(alpha):
    return math.log(20.0) / alpha if alpha > 0 else float("inf")


@dataclass
class Fog:
    alpha: float = 0.0                 # uniform background extinction [1/m]
    plume_alpha: float = 0.0           # extra extinction at the plume core [1/m]
    plume_center: tuple = (1.35, 1.60, 0.30)   # world xyz of the mist source
    plume_sigma_xy: float = 0.45       # horizontal Gaussian sigma [m]
    plume_h: float = 0.6               # e-folding height of the rising plume above the source [m]
    tau_h: float = 5e-9                # pulse half-power duration [s] (ASSUMED for Mid-360)
    xi_r1: float = 0.10                # overlap ramp start [m]  (ASSUMED = the 0.1 m blind zone)
    xi_r2: float = 0.50                # overlap ramp end   [m]  (ASSUMED; Hahner/Velodyne: 0.9, 1.0)
    return_mode: str = "strongest"     # or "first"
    rho_gate: float = 0.0012           # near-field gate: a fog echo is accepted only if its apparent
                                       # reflectivity P*R^2 >= rho_gate (ASSUMED; set so clutter
                                       # starts near MOR 45 m, the clearest fog in which Bijelic
                                       # 2018 saw near-sensor clutter on Velodyne units)
    speckle_shape: float = 4.0         # gamma shape of soft-return power fluctuation (ASSUMED)
    p_tag: float = 0.6                 # P(firmware flags a fog point in tag bits 2-3) (ASSUMED)
    dr: float = 0.02                   # range grid [m]

    @property
    def active(self):
        return self.alpha > 0 or self.plume_alpha > 0

    def alpha_at(self, pts_world):
        """Extinction at world points (..., 3)."""
        a = np.full(pts_world.shape[:-1], self.alpha, dtype=np.float32)
        if self.plume_alpha > 0:
            c = np.asarray(self.plume_center, np.float32)
            d2 = (pts_world[..., 0] - c[0]) ** 2 + (pts_world[..., 1] - c[1]) ** 2
            dz = pts_world[..., 2] - c[2]
            vert = np.where(dz >= 0, np.exp(-np.maximum(dz, 0) / self.plume_h),
                            np.exp(-(dz / 0.15) ** 2))
            a = a + self.plume_alpha * np.exp(-d2 / (2 * self.plume_sigma_xy ** 2)) * vert
        return a

    def describe(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {
            "mor_m_background": alpha_to_mor(self.alpha), "beta_over_alpha": BETA_OVER_ALPHA}


def _pulse_kernel(fog):
    n = max(2, int(round(C_LIGHT * fog.tau_h / fog.dr)))      # pulse length in range bins
    u = (np.arange(n) + 0.5) / n
    return (np.sin(np.pi * u) ** 2).astype(np.float32)


def soft_waveform(fog, origin, d_world, r0, backend="numpy"):
    """Per-ray fog waveform. Returns (W (N, M), r_axis (M,) apparent range of each sample,
    T2_at_target (N,), impulse response S (N, nb), range bins (nb,))."""
    n = d_world.shape[0]
    nb = int(math.ceil((float(r0.max()) + fog.dr) / fog.dr))
    r = ((np.arange(nb) + 0.5) * fog.dr).astype(np.float32)
    if fog.plume_alpha > 0:
        pts = origin.astype(np.float32) + d_world[:, None, :].astype(np.float32) * r[None, :, None]
        a = fog.alpha_at(pts)                                  # (N, nb)
    else:
        a = np.full((1, nb), fog.alpha, np.float32)
    tau = np.cumsum(a, axis=1) * fog.dr - 0.5 * a * fog.dr      # optical depth at bin centre
    t2 = np.exp(-2.0 * tau)
    xi = np.clip((r - fog.xi_r1) / max(fog.xi_r2 - fog.xi_r1, 1e-6), 0.0, 1.0)
    s = (math.pi * BETA_OVER_ALPHA) * a * t2 * xi / (r * r) * fog.dr
    s = np.broadcast_to(s, (n, nb)) * (r[None, :] < r0[:, None])
    idx = np.minimum((r0 / fog.dr).astype(np.int64), nb - 1)
    ar = np.arange(n)
    a_b, tau_b = np.broadcast_to(a, (n, nb)), np.broadcast_to(tau, (n, nb))
    tau_t = tau_b[ar, idx] + a_b[ar, idx] * (r0 - r[idx])      # exact optical depth at the target
    t2_target = np.exp(-2.0 * tau_t)
    g = _pulse_kernel(fog)
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        st = torch.as_tensor(np.ascontiguousarray(s), device=dev)[:, None, :]
        gt = torch.as_tensor(g[::-1].copy(), device=dev)[None, None, :]
        w = torch.nn.functional.conv1d(st, gt, padding=g.size - 1)[:, 0, :].cpu().numpy()
    else:
        w = fftconvolve(s, g[None, :], mode="full", axes=1)
        w = np.maximum(w, 0.0)
    # sample m of the full convolution = waveform at R = r[0] + m*dr; the sensor reports the
    # range of a pulse PEAK, i.e. R - c*tau_H/2 (half the kernel length)
    r_axis = r[0] + np.arange(w.shape[1]) * fog.dr - 0.5 * g.size * fog.dr
    return w, r_axis, t2_target, s, r


def apply_fog(fog, origin, d_world, r0, p_hard_clear, p_thresh, rng, backend="numpy"):
    """Decide hard / soft / dropped per ray.
    Returns dict(range, power, is_fog, detected, p_hard, p_soft_peak)."""
    w, r_axis, t2, s_imp, r_bins = soft_waveform(fog, origin, d_world, r0, backend)
    n = r0.shape[0]
    p_hard = p_hard_clear * t2
    valid = r_axis > 0
    w = w * valid[None, :]
    peak = w.max(axis=1)
    speckle = rng.gamma(fog.speckle_shape, 1.0 / fog.speckle_shape, n)
    p_soft = peak * speckle
    # draw the soft range from the fog impulse response S(r) (pdf proportional to S, r < R0)
    cdf = np.cumsum(s_imp, axis=1)
    tot = cdf[:, -1]
    u = rng.random(n) * np.where(tot > 0, tot, 1.0)
    j = np.minimum((cdf < u[:, None]).sum(axis=1), s_imp.shape[1] - 1)
    r_soft = np.minimum(r_bins[j] + (rng.random(n) - 0.5) * fog.dr, r0)
    r_peak = np.maximum(r_axis[np.argmax(w, axis=1)], fog.xi_r1)       # range of the waveform peak
    soft_ok = (p_soft >= p_thresh) & (p_soft * r_peak ** 2 >= fog.rho_gate)
    if fog.return_mode == "first":
        soft_wins = soft_ok
    else:
        soft_wins = soft_ok & (p_soft > p_hard)
    rng_out = np.where(soft_wins, r_soft, r0)
    pw = np.where(soft_wins, p_soft, p_hard)
    return dict(range=rng_out, power=pw, is_fog=soft_wins, detected=pw >= p_thresh,
                p_hard=p_hard, p_soft_peak=peak)
