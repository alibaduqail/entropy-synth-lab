"""Signal-level FMCW simulation: raw complex ADC cubes [loops, tdm_slot, rx, samples].

Each point scatterer contributes, for TX m / RX n and fast-time sample k of a chirp,
    a * exp(j 2 pi (f0 tau + S tau t_k)),   tau = (|p - tx_m| + |p - rx_n|) n_air / c
with the amplitude from the radar equation (watts^0.5), "stop-and-hop" per chirp (the
scatterer is frozen during one 59 us chirp and moves between chirps). The default fast
path factorises the fast-time tone (using the mean delay over the 4 RX) from the exact
per-antenna carrier phase; ``exact=True`` evaluates the full expression (used in tests).
Thermal noise is complex white Gaussian with power k T fs F per sample.
Not modelled: phase noise, TX-RX leakage, ADC quantisation/saturation, antenna coupling,
multipath ghosts, range-dependent IF filter response.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .config import C0, RadarConfig, antenna_gain, antenna_positions
from .fog import FogState
from .person import Person
from .scene import Scatterers, radar_pose, shadow_factors, static_scatterers


def link_budget(cfg: RadarConfig, rcs_m2: float, range_m: float) -> Dict[str, float]:
    """Boresight radar-equation numbers for one scatterer."""
    lam = cfg.wavelength_m
    g = 10 ** (cfg.ant_gain_dbi / 10.0)
    pr = cfg.tx_power_w * g * g * lam ** 2 * rcs_m2 / ((4 * np.pi) ** 3 * range_m ** 4)
    pn = cfg.noise_power_w
    gain = cfg.n_samples * cfg.n_loops * cfg.n_tx * cfg.n_rx
    snr1 = pr / pn
    return {
        "rcs_m2": rcs_m2, "range_m": range_m,
        "rx_power_dbm": 10 * np.log10(pr / 1e-3), "noise_dbm_per_sample": 10 * np.log10(pn / 1e-3),
        "snr_per_sample_db": 10 * np.log10(snr1),
        "processing_gain_db": 10 * np.log10(gain),
        "snr_after_processing_db": 10 * np.log10(snr1 * gain),
        "noise_equivalent_rcs_dbsm": 10 * np.log10(rcs_m2 / (snr1 * gain)),
    }


class RadarSimulator:
    def __init__(self, cfg: RadarConfig, scene: dict, people: Sequence[Person] = (),
                 extra_objects: Sequence[dict] = (), fog: Optional[FogState] = None,
                 seed: int = 0, noise: bool = True, exact: bool = False,
                 clutter_jitter_m: float = 0.0, n_fog_scatterers: int = 24,
                 fog_ramp: Optional[Callable[[float], float]] = None,
                 include_static: bool = True, shadowing: bool = True):
        self.cfg = cfg
        self.scene = scene
        self.people = list(people)
        self.fog = fog if fog is not None else FogState()
        self.fog_ramp = fog_ramp
        self.noise = noise
        self.exact = exact
        self.clutter_jitter_m = clutter_jitter_m
        self.rng = np.random.default_rng(seed)
        self.pose = radar_pose(scene)
        self.tx_pos, self.rx_pos = antenna_positions(cfg)
        self.slot_tx = np.array(cfg.tdm_order)
        k = np.arange(cfg.n_samples)
        self.t_fast = cfg.adc_start_s + k / cfg.fs_hz
        self.t_mid = float(self.t_fast.mean())
        if include_static:
            st = static_scatterers(scene, cfg.wavelength_m, extra_objects, seed=seed)
        else:
            st = Scatterers(np.zeros((0, 3)), np.zeros(0), [])
        self.static = st
        self.static_fac = (shadow_factors(st, self.pose.position, [p.xy0 for p in self.people])
                           if shadowing else np.ones(len(st)))
        self.static_radar = self.pose.to_radar(st.pos) if len(st) else np.zeros((0, 3))
        # fog droplets: a few drifting weak scatterers, each standing for a parcel of mist
        self._init_fog_scatterers(n_fog_scatterers, seed)
        self._static_cache = None

    # ------------------------------------------------------------------
    def _init_fog_scatterers(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed + 77)
        sx, sy, sz = self.scene["room"]["size_xyz"]
        self.fog_pos0 = rng.uniform([0, 0, 0], [sx, sy, sz], size=(n, 3))
        v = rng.standard_normal((n, 3))
        self.fog_vel = v / np.linalg.norm(v, axis=1, keepdims=True) * self.fog.air_speed_mps
        self.fog_parcel_m3 = sx * sy * sz / max(n, 1)

    def _fog_scale(self, t: float) -> float:
        return 1.0 if self.fog_ramp is None else float(self.fog_ramp(t))

    def _medium(self, t: float):
        """(one-way dB/m, refractive index, complex film factor, fog scale) at time t."""
        f_ghz = self.cfg.f_center_hz / 1e9
        s = self._fog_scale(t)
        base = FogState().atten_db_per_m_one_way(f_ghz)
        att = base + s * (self.fog.atten_db_per_m_one_way(f_ghz) - base)
        n_air = 1.0 + s * self.fog.delta_refractivity_ppm(f_ghz) * 1e-6
        # conj: in this model a delay is a POSITIVE phase (exp(+j 2 pi f0 tau))
        film = (1.0 + s * (np.conj(self.fog.film_two_way(f_ghz)) - 1.0)
                if self.fog.film_m > 0 else 1.0)
        return att, n_air, film, s

    # ------------------------------------------------------------------
    def _synth(self, pos: np.ndarray, rcs: np.ndarray, slots: np.ndarray, att_db_m: float,
               n_air: float, fac: Optional[np.ndarray] = None) -> np.ndarray:
        """pos (C,P,3) radar frame, rcs (P,), slots (C,) -> cube (C, n_rx, n_samples)."""
        cfg = self.cfg
        C, P = pos.shape[0], pos.shape[1]
        if P == 0:
            return np.zeros((C, cfg.n_rx, cfg.n_samples), dtype=np.complex128)
        txp = self.tx_pos[self.slot_tx[slots]]                             # (C,3)
        d_tx = np.linalg.norm(pos - txp[:, None, :], axis=2)                # (C,P)
        d_rx = np.linalg.norm(pos[:, :, None, :] - self.rx_pos[None, None, :, :], axis=3)
        rng_c = np.linalg.norm(pos, axis=2)
        az = np.arctan2(pos[:, :, 0], pos[:, :, 1])
        el = np.arcsin(np.clip(pos[:, :, 2] / np.maximum(rng_c, 1e-6), -1, 1))
        g = antenna_gain(cfg, az, el)                                      # one-way
        lam = cfg.wavelength_m
        amp = np.sqrt(cfg.tx_power_w * lam ** 2 * rcs[None, :] / (4 * np.pi) ** 3) * g \
            / np.maximum(rng_c, 0.05) ** 2
        amp = amp * 10 ** (-att_db_m * 2.0 * rng_c / 20.0)
        if fac is not None:
            amp = amp * fac[None, :]
        tau = (d_tx[:, :, None] + d_rx) * (n_air / C0)                      # (C,P,N)
        S, f0 = cfg.slope_hz_per_s, cfg.f_start_hz
        if self.exact:
            ph = (f0 * tau)[..., None] + S * tau[..., None] * self.t_fast    # cycles (C,P,N,K)
            sig = amp[:, :, None, None] * np.exp(2j * np.pi * (ph % 1.0))
            return sig.sum(axis=1)
        tau_m = tau.mean(axis=2)                                            # (C,P)
        slow_ph = (f0 + S * self.t_mid) * tau
        slow = amp[:, :, None] * np.exp(2j * np.pi * (slow_ph % 1.0))       # (C,P,N)
        fast = np.exp(2j * np.pi * S * tau_m[:, :, None] * (self.t_fast - self.t_mid))
        return np.matmul(slow.transpose(0, 2, 1), fast)                     # (C,N,K)

    def _static_cube(self, t: float) -> np.ndarray:
        """Static-scene cube for one loop, tiled over loops (cached when nothing changes)."""
        cfg = self.cfg
        varying = self.fog_ramp is not None or self.clutter_jitter_m > 0
        if self._static_cache is not None and not varying:
            return self._static_cache
        att, n_air, film, _ = self._medium(t)
        pos = self.static_radar
        if self.clutter_jitter_m > 0 and len(pos):
            pos = pos + self.rng.standard_normal(pos.shape) * self.clutter_jitter_m
        slots = np.arange(cfg.n_tx)
        one = self._synth(np.repeat(pos[None], cfg.n_tx, axis=0), self.static.rcs, slots,
                          att, n_air, self.static_fac) * film
        cube = np.broadcast_to(one[None], (cfg.n_loops,) + one.shape).reshape(
            cfg.n_loops * cfg.n_tx, cfg.n_rx, cfg.n_samples)
        if not varying:
            self._static_cache = cube
        return cube

    def chirp_times(self, frame: int) -> np.ndarray:
        cfg = self.cfg
        c = np.arange(cfg.n_loops * cfg.n_tx)
        return frame * cfg.frame_period_s + c * cfg.chirp_time_s + cfg.idle_s + self.t_mid

    def frame(self, i: int) -> np.ndarray:
        """Raw ADC cube of frame ``i``: complex64 [n_loops, n_tx(slot), n_rx, n_samples]."""
        cfg = self.cfg
        t = self.chirp_times(i)
        slots = np.arange(len(t)) % cfg.n_tx
        att, n_air, film, fscale = self._medium(float(t[0]))
        cube = np.array(self._static_cube(float(t[0])), dtype=np.complex128)
        for person in self.people:
            pos = self.pose.to_radar(person.positions(t))
            cube += self._synth(pos, person.rcs, slots, att, n_air) * film
        eta = self.fog.eta(cfg.f_center_hz / 1e9) * fscale
        if eta > 0 and len(self.fog_pos0):
            p = self.fog_pos0[None] + self.fog_vel[None] * t[:, None, None]
            size = np.array(self.scene["room"]["size_xyz"])
            p = np.abs(((p + size) % (2 * size)) - size)                    # bounce in the room
            rcs = np.full(len(self.fog_pos0), eta * self.fog_parcel_m3)
            cube += self._synth(self.pose.to_radar(p), rcs, slots, att, n_air) * film
        if self.noise:
            sd = np.sqrt(cfg.noise_power_w / 2.0)
            nz = self.rng.standard_normal(cube.shape + (2,), dtype=np.float32)
            cube += sd * (nz[..., 0] + 1j * nz[..., 1])
        return cube.reshape(cfg.n_loops, cfg.n_tx, cfg.n_rx, cfg.n_samples).astype(np.complex64)
