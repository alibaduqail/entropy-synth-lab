"""TI IWR6843ISK radar parameters: chirp presets, antenna geometry, link budget.

Sources (see docs/radar-sim.md for the full list and what is assumed):
- ``people_counting_6m``: TI 3D People Counting ``ISK_6m_default.cfg``
  ``profileCfg 0 60.75 30.00 25.00 59.10 657930 0 54.71 1 96 2950.00 2 1 36``,
  ``frameCfg 0 2 96 0 55.00 1 0`` (3 TX in TDM, 96 loops, 55 ms frame).
- ``oob_3d``: mmWave SDK out-of-box demo ``profile_3d.cfg`` for xwr68xx
  (``profileCfg 0 60 359 7 57.14 0 0 70 1 256 5209 0 0 158``, ``frameCfg 0 2 16 0 100 1 0``).
  Recalled from the SDK, NOT re-verified tonight.
- RF numbers (12 dBm TX, 12 dB noise figure, ~7 dBi antenna) are datasheet-typical values
  recalled from the IWR6843 datasheet / ISK EVM guide: ASSUMED, check before quoting.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np

C0 = 299_792_458.0
K_BOLTZ = 1.380649e-23


@dataclass
class RadarConfig:
    name: str = "people_counting_6m"
    f_start_hz: float = 60.75e9
    idle_s: float = 30e-6
    adc_start_s: float = 25e-6
    ramp_end_s: float = 59.1e-6
    slope_hz_per_s: float = 54.71e12
    n_samples: int = 96
    fs_hz: float = 2.95e6
    n_loops: int = 96          # chirps per TX per frame (Doppler FFT size)
    frame_period_s: float = 55e-3
    # TDM order, as indices into the TX list (TX1, TX2, TX3) -> ISK cfg uses TX1, TX3, TX2
    tdm_order: Tuple[int, ...] = (0, 2, 1)
    n_rx: int = 4
    # link budget (assumed datasheet-typical)
    tx_power_dbm: float = 12.0
    ant_gain_dbi: float = 7.0
    noise_figure_db: float = 12.0
    temp_k: float = 295.0
    # one-way antenna power pattern ~ cos^p; exponents chosen so the one-way -6 dB points
    # fall at +-60 deg azimuth and +-15 deg elevation... the ISK guide quotes a
    # 120 x 30 deg field of view (ASSUMED pattern shape)
    az_cos_power: float = 2.0
    el_cos_power: float = 40.0
    back_lobe_db: float = -30.0

    # ---- derived quantities -------------------------------------------------
    @property
    def n_tx(self) -> int:
        return len(self.tdm_order)

    @property
    def chirp_time_s(self) -> float:
        return self.idle_s + self.ramp_end_s

    @property
    def tx_repeat_s(self) -> float:
        """Time between two chirps of the same TX (Doppler sampling interval)."""
        return self.chirp_time_s * self.n_tx

    @property
    def adc_time_s(self) -> float:
        return self.n_samples / self.fs_hz

    @property
    def bandwidth_hz(self) -> float:
        return self.slope_hz_per_s * self.adc_time_s

    @property
    def f_center_hz(self) -> float:
        return self.f_start_hz + self.slope_hz_per_s * (self.adc_start_s + 0.5 * self.adc_time_s)

    @property
    def wavelength_m(self) -> float:
        return C0 / self.f_center_hz

    @property
    def range_res_m(self) -> float:
        return C0 / (2.0 * self.bandwidth_hz)

    @property
    def range_max_m(self) -> float:
        # complex baseband: unambiguous beat frequency = fs
        return self.fs_hz * C0 / (2.0 * self.slope_hz_per_s)

    @property
    def v_max_mps(self) -> float:
        return self.wavelength_m / (4.0 * self.tx_repeat_s)

    @property
    def v_res_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.n_loops * self.tx_repeat_s)

    @property
    def frame_active_s(self) -> float:
        return self.n_loops * self.tx_repeat_s

    @property
    def noise_power_w(self) -> float:
        """Thermal noise power per complex ADC sample (bandwidth = fs)."""
        return K_BOLTZ * self.temp_k * self.fs_hz * 10 ** (self.noise_figure_db / 10.0)

    @property
    def tx_power_w(self) -> float:
        return 1e-3 * 10 ** (self.tx_power_dbm / 10.0)

    def range_axis(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.range_max_m / self.n_samples

    def doppler_axis(self) -> np.ndarray:
        """Velocity of each fftshift-ed Doppler bin (m/s, + = receding)."""
        return (np.arange(self.n_loops) - self.n_loops // 2) * self.v_res_mps

    def summary(self) -> Dict[str, float]:
        d = asdict(self)
        d["tdm_order"] = list(self.tdm_order)
        for k in ("n_tx", "chirp_time_s", "tx_repeat_s", "adc_time_s", "bandwidth_hz",
                  "f_center_hz", "wavelength_m", "range_res_m", "range_max_m", "v_max_mps",
                  "v_res_mps", "frame_active_s", "noise_power_w", "tx_power_w"):
            d[k] = float(getattr(self, k))
        d["frame_rate_hz"] = 1.0 / self.frame_period_s
        d["adc_cube_shape"] = [self.n_loops, self.n_tx, self.n_rx, self.n_samples]
        return d


def preset(name: str = "people_counting_6m") -> RadarConfig:
    if name == "people_counting_6m":
        return RadarConfig()
    if name == "oob_3d":
        return RadarConfig(name="oob_3d", f_start_hz=60.0e9, idle_s=359e-6, adc_start_s=7e-6,
                           ramp_end_s=57.14e-6, slope_hz_per_s=70e12, n_samples=256,
                           fs_hz=5.209e6, n_loops=16, frame_period_s=100e-3)
    if name == "tiny_test":  # small cube for unit tests only
        return RadarConfig(name="tiny_test", n_samples=64, n_loops=16)
    raise ValueError("unknown radar preset %r" % name)


PRESETS: List[str] = ["people_counting_6m", "oob_3d"]


# ---- IWR6843ISK antenna geometry -------------------------------------------------
# Units of half-wavelength; radar frame: x = azimuth (to the right when looking along
# boresight), z = up, y = boresight. RX: 4-element lambda/2 azimuth ULA. TX1 and TX3 are
# 2 lambda apart in azimuth; TX2 sits midway and lambda/2 higher. This gives the usual
# ISK virtual array: 8 azimuth elements (TX1+TX3) plus 4 elements one row up (TX2)
# above azimuth positions 2..5.
RX_POS_HALF_LAMBDA = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)   # (x, z)
TX_POS_HALF_LAMBDA = np.array([[0, 0], [2, 1], [4, 0]], dtype=float)           # TX1, TX2, TX3


def antenna_positions(cfg: RadarConfig) -> Tuple[np.ndarray, np.ndarray]:
    """TX (n_tx_phys,3) and RX (n_rx,3) element positions in the radar frame, metres,
    centred on the array centroid."""
    half = cfg.wavelength_m / 2.0
    tx = np.zeros((3, 3))
    rx = np.zeros((4, 3))
    tx[:, 0] = TX_POS_HALF_LAMBDA[:, 0] * half
    tx[:, 2] = TX_POS_HALF_LAMBDA[:, 1] * half
    rx[:, 0] = RX_POS_HALF_LAMBDA[:, 0] * half
    rx[:, 2] = RX_POS_HALF_LAMBDA[:, 1] * half
    ctr = 0.5 * (tx.mean(axis=0) + rx.mean(axis=0))
    return tx - ctr, rx - ctr


def virtual_array_index(cfg: RadarConfig) -> Dict[str, np.ndarray]:
    """Where each (tdm_slot, rx) channel sits in the virtual array, in half-wavelengths.

    Returns ``az`` and ``el`` integer arrays of shape (n_tx, n_rx) indexed by TDM slot.
    """
    az = np.zeros((cfg.n_tx, cfg.n_rx), dtype=int)
    el = np.zeros((cfg.n_tx, cfg.n_rx), dtype=int)
    for slot, tx in enumerate(cfg.tdm_order):
        az[slot] = (TX_POS_HALF_LAMBDA[tx, 0] + RX_POS_HALF_LAMBDA[:, 0]).astype(int)
        el[slot] = (TX_POS_HALF_LAMBDA[tx, 1] + RX_POS_HALF_LAMBDA[:, 1]).astype(int)
    return {"az": az, "el": el}


def antenna_gain(cfg: RadarConfig, az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """One-way linear power gain toward azimuth/elevation (radians)."""
    g0 = 10 ** (cfg.ant_gain_dbi / 10.0)
    front = np.cos(az) > 0
    pat = (np.abs(np.cos(az)) ** cfg.az_cos_power) * (np.abs(np.cos(el)) ** cfg.el_cos_power)
    back = 10 ** (cfg.back_lobe_db / 10.0)
    return g0 * np.where(front, np.maximum(pat, back), back)
