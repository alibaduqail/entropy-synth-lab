"""60 GHz propagation through fog / ultrasonic mist.

- Liquid-water (cloud/fog) attenuation: ITU-R P.840 Rayleigh model with the double-Debye
  permittivity of water.  gamma = K_l(f, T) * M   [dB/km],  M = liquid water content g/m^3.
- Droplet backscatter: Rayleigh volume reflectivity  eta = pi^5 |K|^2 / lambda^4 * sum(N D^6).
- Refractivity change (humidity + liquid water) -> apparent range shift (ITU-R P.453 form).
- Water film on the radome: plane-slab transmission.
- Gas absorption (O2 ~15 dB/km at 60 GHz, water vapour) is the same in every condition
  and entered as ASSUMED constants; it is not computed line-by-line.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

C0 = 299_792_458.0
RHO_WATER = 1.0e6            # g/m^3
O2_DB_PER_KM_60GHZ = 15.0    # ASSUMED (ITU-R P.676 figure, sea level, near the 60 GHz peak)
VAPOUR_DB_PER_KM_PER_GM3 = 0.025   # ASSUMED order of magnitude at 60 GHz


def water_permittivity(f_ghz, temp_c: float = 22.5):
    """Complex relative permittivity eps' - j eps'' of liquid water (P.840 double Debye)."""
    f = np.asarray(f_ghz, dtype=float)
    theta = 300.0 / (temp_c + 273.15)
    e0 = 77.66 + 103.3 * (theta - 1.0)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146.0 * (theta - 1.0) + 316.0 * (theta - 1.0) ** 2
    fs = 39.8 * fp
    er = (e0 - e1) / (1 + (f / fp) ** 2) + (e1 - e2) / (1 + (f / fs) ** 2) + e2
    ei = f * (e0 - e1) / (fp * (1 + (f / fp) ** 2)) + f * (e1 - e2) / (fs * (1 + (f / fs) ** 2))
    return er, ei


def kl_db_per_km_per_gm3(f_ghz, temp_c: float = 22.5):
    """ITU-R P.840 specific attenuation coefficient K_l, (dB/km)/(g/m^3)."""
    er, ei = water_permittivity(f_ghz, temp_c)
    eta = (2.0 + er) / ei
    return 0.819 * np.asarray(f_ghz, dtype=float) / (ei * (1.0 + eta ** 2))


def fog_attenuation_db(f_ghz, lwc_gm3, path_m, temp_c: float = 22.5, two_way: bool = True):
    """Liquid-water attenuation over a target at distance ``path_m`` (two-way by default)."""
    length_km = np.asarray(path_m, dtype=float) * (2.0 if two_way else 1.0) / 1000.0
    return kl_db_per_km_per_gm3(f_ghz, temp_c) * np.asarray(lwc_gm3, dtype=float) * length_km


def rayleigh_k2(f_ghz, temp_c: float = 22.5):
    er, ei = water_permittivity(f_ghz, temp_c)
    eps = er - 1j * ei
    return np.abs((eps - 1) / (eps + 2)) ** 2


def volume_reflectivity(f_ghz, lwc_gm3, drop_diam_m, temp_c: float = 22.5):
    """Radar backscatter cross-section per unit volume (m^2/m^3) of a monodisperse mist."""
    lam = C0 / (np.asarray(f_ghz, dtype=float) * 1e9)
    n_per_m3 = np.asarray(lwc_gm3, dtype=float) / (RHO_WATER * np.pi * drop_diam_m ** 3 / 6.0)
    return np.pi ** 5 * rayleigh_k2(f_ghz, temp_c) / lam ** 4 * n_per_m3 * drop_diam_m ** 6


def optical_extinction_per_m(lwc_gm3, drop_diam_m, q_ext: float = 2.0):
    """Visible/near-IR extinction coefficient (1/m) of the same mist (geometric optics,
    Q_ext = 2). This is what the LiDAR sees; used only to give the fog levels a meaning."""
    return 3.0 * q_ext * np.asarray(lwc_gm3, dtype=float) / (2.0 * RHO_WATER * drop_diam_m)


def saturation_vapour_hpa(temp_c: float) -> float:
    return 6.1121 * np.exp(17.502 * temp_c / (240.97 + temp_c))


def vapour_density_gm3(temp_c: float, rh: float) -> float:
    e = rh * saturation_vapour_hpa(temp_c)
    return 216.7 * e / (temp_c + 273.15)


def refractivity_ppm(temp_c: float, rh: float, lwc_gm3: float = 0.0, p_hpa: float = 1013.0,
                     f_ghz: float = 62.0) -> float:
    """Radio refractivity N = (n-1)*1e6 of moist air plus suspended liquid water."""
    t = temp_c + 273.15
    e = rh * saturation_vapour_hpa(temp_c)
    n_air = 77.6 * p_hpa / t - 5.6 * e / t + 3.75e5 * e / t ** 2
    er, ei = water_permittivity(f_ghz, temp_c)
    eps = er - 1j * ei
    n_liq = 1.5 * (lwc_gm3 / RHO_WATER) * np.real((eps - 1) / (eps + 2)) * 1e6
    return float(n_air + n_liq)


def water_film_transmission(f_ghz, thickness_m, temp_c: float = 22.5):
    """Complex one-way field transmission of a free-standing water film (normal incidence)."""
    er, ei = water_permittivity(f_ghz, temp_c)
    n = np.sqrt(er - 1j * ei)
    k0 = 2 * np.pi * np.asarray(f_ghz, dtype=float) * 1e9 / C0
    d = np.asarray(thickness_m, dtype=float)
    r = (1 - n) / (1 + n)
    ph = np.exp(-1j * k0 * n * d)
    t = (1 - r ** 2) * ph / (1 - r ** 2 * ph ** 2)
    # remove the phase of the same thickness of air so the phase is the EXTRA delay
    return t * np.exp(1j * k0 * d)


@dataclass
class FogState:
    """Uniform fog filling the room. ``level`` 0 = clear air."""
    name: str = "clear"
    lwc_gm3: float = 0.0
    drop_diam_m: float = 4e-6        # ultrasonic mist makers: a few micrometres (ASSUMED)
    rh: float = 0.45
    temp_c: float = 22.5
    rh_clear: float = 0.45           # reference humidity that defines "no extra delay"
    air_speed_mps: float = 0.10      # droplets drift with room air currents (ASSUMED)
    film_m: float = 0.0              # water film on the radome (0 = dry)

    def atten_db_per_m_one_way(self, f_ghz: float) -> float:
        liquid = float(kl_db_per_km_per_gm3(f_ghz, self.temp_c)) * self.lwc_gm3
        vap = VAPOUR_DB_PER_KM_PER_GM3 * vapour_density_gm3(self.temp_c, self.rh)
        return (liquid + vap + O2_DB_PER_KM_60GHZ) / 1000.0

    def delta_refractivity_ppm(self, f_ghz: float = 62.0) -> float:
        return (refractivity_ppm(self.temp_c, self.rh, self.lwc_gm3, f_ghz=f_ghz)
                - refractivity_ppm(self.temp_c, self.rh_clear, 0.0, f_ghz=f_ghz))

    def film_two_way(self, f_ghz: float) -> complex:
        if self.film_m <= 0:
            return 1.0 + 0j
        return complex(water_film_transmission(f_ghz, self.film_m, self.temp_c)) ** 2

    def eta(self, f_ghz: float) -> float:
        if self.lwc_gm3 <= 0:
            return 0.0
        return float(volume_reflectivity(f_ghz, self.lwc_gm3, self.drop_diam_m, self.temp_c))

    def summary(self, f_ghz: float = 62.0) -> Dict[str, float]:
        d = asdict(self)
        d.update(
            kl_db_per_km_per_gm3=float(kl_db_per_km_per_gm3(f_ghz, self.temp_c)),
            atten_db_per_m_one_way=self.atten_db_per_m_one_way(f_ghz),
            liquid_two_way_db_at_2m=float(fog_attenuation_db(f_ghz, self.lwc_gm3, 2.0, self.temp_c)),
            delta_refractivity_ppm=self.delta_refractivity_ppm(f_ghz),
            apparent_range_shift_mm_at_2m=self.delta_refractivity_ppm(f_ghz) * 1e-6 * 2.0 * 1e3,
            volume_reflectivity_m2_per_m3=self.eta(f_ghz),
            optical_extinction_per_m=float(optical_extinction_per_m(self.lwc_gm3, self.drop_diam_m)),
            optical_transmission_over_3m=float(np.exp(-3.0 * optical_extinction_per_m(
                self.lwc_gm3, self.drop_diam_m))),
        )
        return d


# Fog levels used for B0-B3. LWC values are ASSUMED: chosen so the *optical* transmission
# over 3 m spans "light haze" to "cannot see the far wall" for ~4 um mist droplets.
FOG_LEVELS = {
    "B0": FogState("B0_clear", 0.0, rh=0.45),
    "B1": FogState("B1_light", 0.1, rh=0.80),
    "B2": FogState("B2_medium", 0.5, rh=0.95),
    "B3": FogState("B3_heavy", 2.0, rh=1.00),
}
