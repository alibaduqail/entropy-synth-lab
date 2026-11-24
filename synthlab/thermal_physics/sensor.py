"""FLIR Lepton 3.5 / PureThermal 3 sensor model.

radiance (supersampled) -> lens PSF -> pixel aperture -> TLinear counts ->
low-frequency non-uniformity -> fixed-pattern noise (redrawn at each FFC) ->
global offset (slow drift + post-FFC transient + flicker) -> temporally filtered
(AR(1)) pixel / row / column noise -> raw-count lattice with dead-band hold ->
uint16 TLinear (0.01 K/count).

Every default below is either MEASURED from the real recordings
(results/thermal-physics/real_calibration.json; see docs/thermal-physics.md for the
table) or marked ASSUMED. Units: TLinear counts (1 count = 10 mK).
"""
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage

from . import radiometry


@dataclass
class SensorParams:
    width: int = 160
    height: int = 120
    fps: float = 8.60                 # MEASURED effective capture rate (nominal 8.7)
    psf_sigma_px: float = 0.50        # MEASURED via edge estimator (datasheet-derived value is ~0.8)
    # temporal noise: stationary SD of AR(1) processes, counts. MEASURED frame-difference SDs
    # (pixel 1.95, row 0.70, col 1.20, flicker 0.57 counts) divided by sqrt(1-rho).
    ar1_rho: float = 0.89             # MEASURED lag-1..32 autocorrelation of pixel series
    pixel_sd: float = 5.9
    row_sd: float = 2.1
    col_sd: float = 3.6
    flicker_sd: float = 1.7
    noise_blur_px: float = 0.60       # MEASURED neighbour correlation of pixel noise ~0.3
    # fixed pattern (flat-wall mean image after removing 2nd-order polynomial), counts
    fpn_col_sd: float = 7.2
    fpn_row_sd: float = 1.7
    fpn_pixel_sd: float = 7.5
    fpn_ffc_keep: float = 0.3         # ASSUMED fraction of FPN surviving an FFC (most is redrawn; measured change ~= full FPN)
    # low-frequency non-uniformity a + b*u + c*v + d*u^2 + e*u*v + f*v^2 (u,v in [-1,1]); MEASURED
    # on the flat-wall runs (median of 5; p2p ~1.2 K). Could partly be the wall itself - see doc.
    lowfreq_coef: tuple = (0.0, 16.0, 25.0, 42.0, -31.0, 5.0)
    # global offset
    drift_counts_per_min_sd: float = 9.0     # MEASURED |slope| median 7-16 counts/min between FFCs
    ffc_interval_s: float = 180.0            # datasheet (FLIR Lepton eng. datasheet rev 200, p19); runs show 0.2-0.67 per min
    ffc_gap_s: float = 1.82                  # MEASURED capture gap
    ffc_peak_counts: float = 165.0           # MEASURED post-FFC offset bump (133-227)
    ffc_peak_sd: float = 25.0
    ffc_start_frac: float = 0.33             # MEASURED first frame after the gap is at ~1/3 of the peak
    ffc_rise_tau_s: float = 2.2              # MEASURED peak reached after ~6 s
    ffc_hold_s: float = 6.0
    ffc_decay_tau_s: float = 13.7            # MEASURED half-decay ~9.5 s after the peak
    # output lattice
    lattice_counts: float = 2.25             # MEASURED spacing of distinct values in a frame
    deadband_counts: float = 3.5             # TUNED so ~63 % of pixels repeat frame to frame (measured 63 %)
    calibration_offset_k: float = 0.0        # absolute radiometric error (datasheet +-5 C); 0 = trust scene temps
    seed: int = 0
    notes: dict = field(default_factory=dict)


def lowfreq_map(coef, h=120, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    a, b, c, d, e, f = coef
    m = a + b * u + c * v + d * u * u + e * u * v + f * v * v
    return m - m.mean()


def ffc_bump(tau, p, peak):
    """Global offset (counts) `tau` seconds after an FFC ends."""
    tau = np.asarray(tau, float)
    rise = 1.0 - (1.0 - p.ffc_start_frac) * np.exp(-tau / p.ffc_rise_tau_s)
    decay = np.exp(-np.clip(tau - p.ffc_hold_s, 0, None) / p.ffc_decay_tau_s)
    return np.where(tau >= 0, peak * rise * decay, 0.0)


class Lepton35:
    def __init__(self, params=None, supersample=2, first_ffc_s=None, noise=True, lut=None):
        self.p = params or SensorParams()
        self.ss = int(supersample)
        self.noise = bool(noise)
        self.lut = lut or radiometry.default_lut()
        self.rng = np.random.default_rng(self.p.seed)
        p, h, w = self.p, self.p.height, self.p.width
        self.low = lowfreq_map(p.lowfreq_coef, h, w)
        self.fpn = self._draw_fpn()
        self.drift_slope = self.rng.normal(0, p.drift_counts_per_min_sd) / 60.0
        self.next_ffc = self.rng.uniform(20.0, p.ffc_interval_s) if first_ffc_s is None else float(first_ffc_s)
        self.ffc_events = []            # (time the FFC ended, peak)
        self.t = 0.0
        self.frame_index = 0
        s = np.sqrt(1.0)
        self.a_pix = self._blurred_noise() * p.pixel_sd * s
        self.a_row = self.rng.normal(0, p.row_sd, (h, 1))
        self.a_col = self.rng.normal(0, p.col_sd, (1, w))
        self.a_flk = self.rng.normal(0, p.flicker_sd)
        self.held = None

    # -- noise pieces
    def _blurred_noise(self):
        p = self.p
        n = self.rng.normal(0, 1, (p.height, p.width))
        if p.noise_blur_px > 0:
            n = ndimage.gaussian_filter(n, p.noise_blur_px, mode="wrap")
            n /= n.std()
        return n

    def _draw_fpn(self):
        p = self.p
        return (self.rng.normal(0, p.fpn_col_sd, (1, p.width)) + self.rng.normal(0, p.fpn_row_sd, (p.height, 1))
                + self.rng.normal(0, p.fpn_pixel_sd, (p.height, p.width)))

    def _step_noise(self):
        p = self.p
        r, q = p.ar1_rho, np.sqrt(1.0 - p.ar1_rho ** 2)
        self.a_pix = r * self.a_pix + q * p.pixel_sd * self._blurred_noise()
        self.a_row = r * self.a_row + q * self.rng.normal(0, p.row_sd, self.a_row.shape)
        self.a_col = r * self.a_col + q * self.rng.normal(0, p.col_sd, self.a_col.shape)
        self.a_flk = r * self.a_flk + q * self.rng.normal(0, p.flicker_sd)

    # -- optics
    def optics(self, radiance_hi):
        """PSF blur (radiance domain) + pixel-aperture box average -> (H, W) radiance."""
        l = np.asarray(radiance_hi, np.float64)
        ss = self.ss
        if self.p.psf_sigma_px > 0:
            l = ndimage.gaussian_filter(l, self.p.psf_sigma_px * ss, mode="nearest")
        h, w = self.p.height, self.p.width
        return l.reshape(h, ss, w, ss).mean((1, 3))

    def ideal_counts(self, radiance_hi):
        t_k = self.lut.temperature(self.optics(radiance_hi)) + self.p.calibration_offset_k
        return t_k * 100.0

    # -- main entry
    def capture(self, radiance_hi=None, counts_ideal=None):
        """Advance one frame period and return (uint16 frame, info dict).
        Pass either supersampled radiance or precomputed ideal counts (H, W)."""
        p = self.p
        c = self.ideal_counts(radiance_hi) if counts_ideal is None else np.asarray(counts_ideal, np.float64)
        info = dict(t=self.t, ffc=False)
        if not self.noise:
            out = radiometry.counts_from_kelvin(c / 100.0)
            self._advance()
            return out, info
        if self.t >= self.next_ffc:                      # shutter event: frames lost, FPN redrawn
            self.t += p.ffc_gap_s
            keep = p.fpn_ffc_keep
            self.fpn = keep * self.fpn + np.sqrt(1 - keep ** 2) * self._draw_fpn()
            self.ffc_events.append((self.t, self.rng.normal(p.ffc_peak_counts, p.ffc_peak_sd)))
            self.next_ffc = self.t + p.ffc_interval_s
            self.drift_slope = self.rng.normal(0, p.drift_counts_per_min_sd) / 60.0
            info.update(ffc=True, t=self.t)
        self._step_noise()
        offset = self.drift_slope * self.t + self.a_flk
        for t_end, peak in self.ffc_events[-2:]:
            offset += float(ffc_bump(self.t - t_end, p, peak))
        x = c + self.low + self.fpn + offset + self.a_pix + self.a_row + self.a_col
        if p.lattice_counts > 0:
            q = np.rint(x / p.lattice_counts) * p.lattice_counts
            if self.held is not None and p.deadband_counts > 0:
                q = np.where(np.abs(x - self.held) < p.deadband_counts, self.held, q)
            self.held = q
            x = q
        out = np.clip(np.rint(x), 0, 65535).astype(np.uint16)
        info["offset_counts"] = float(offset)
        self._advance()
        return out, info

    def _advance(self):
        self.t += 1.0 / self.p.fps
        self.frame_index += 1

    def describe(self):
        d = asdict(self.p)
        d["supersample"] = self.ss
        return d


def simulate_static(counts_ideal, n_frames, params=None, first_ffc_s=None):
    """Convenience: N noisy frames of a static ideal-count image. Returns (frames, times)."""
    cam = Lepton35(params, first_ffc_s=first_ffc_s)
    frames, times = [], []
    for _ in range(n_frames):
        f, info = cam.capture(counts_ideal=counts_ideal)
        frames.append(f)
        times.append(info["t"])
    return np.stack(frames), np.array(times)
