"""Band radiometry for the Lepton 3.5 (8-14 um): Planck integration, LUTs, fog.

All radiances are in-band radiance L [W m^-2 sr^-1] integrated over 8-14 um with a
flat spectral response (ASSUMPTION: the real Lepton response is not flat, but the
camera is itself calibrated to report black-body-equivalent temperature, so only
the T<->L mapping shape matters and it is nearly identical for any 8-14 um weighting).
"""
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
BAND_UM = (8.0, 14.0)
T_MIN_K = 223.15   # -50 C
T_MAX_K = 773.15   # 500 C (heat-lamp bulb face)
N_LUT = 4096


def planck_spectral(lam_m, t_k):
    """Spectral radiance B(lambda, T) [W m^-3 sr^-1]."""
    lam_m = np.asarray(lam_m, dtype=np.float64)
    t_k = np.asarray(t_k, dtype=np.float64)
    x = H * C / (lam_m * KB * t_k)
    return 2.0 * H * C**2 / lam_m**5 / np.expm1(x)


def band_radiance_exact(t_k, band_um=BAND_UM, n=600):
    """Numerically integrate Planck over the band (trapezoid, n samples)."""
    t_k = np.atleast_1d(np.asarray(t_k, dtype=np.float64))
    lam = np.linspace(band_um[0], band_um[1], n) * 1e-6
    b = planck_spectral(lam[None, :], t_k[:, None])
    return np.trapz(b, lam, axis=1) if hasattr(np, "trapz") else np.trapezoid(b, lam, axis=1)


class BandLUT:
    """Temperature <-> band-radiance lookup, usable with numpy or torch arrays.

    T grid is uniform so T->L is a direct linear interpolation; L->T uses a
    second table on a uniform L grid (radiance is monotonic in T).
    """

    def __init__(self, band_um=BAND_UM, t_min_k=T_MIN_K, t_max_k=T_MAX_K, n=N_LUT):
        self.band_um = tuple(band_um)
        self.t_min, self.t_max, self.n = float(t_min_k), float(t_max_k), int(n)
        self.t_grid = np.linspace(self.t_min, self.t_max, self.n)
        self.l_grid = band_radiance_exact(self.t_grid, band_um)
        self.l_min, self.l_max = float(self.l_grid[0]), float(self.l_grid[-1])
        self.l_uniform = np.linspace(self.l_min, self.l_max, self.n * 4)
        self.t_of_l = np.interp(self.l_uniform, self.l_grid, self.t_grid)
        self._torch = {}

    # -- numpy --
    def radiance(self, t_k):
        return np.interp(t_k, self.t_grid, self.l_grid)

    def temperature(self, l):
        return np.interp(l, self.l_uniform, self.t_of_l)

    # -- torch (same maths, tables moved to the tensor's device) --
    def _tables(self, like):
        import torch
        key = (str(like.device), str(like.dtype))
        if key not in self._torch:
            self._torch[key] = (
                torch.as_tensor(self.l_grid, dtype=like.dtype, device=like.device),
                torch.as_tensor(self.t_of_l, dtype=like.dtype, device=like.device),
            )
        return self._torch[key]

    @staticmethod
    def _lerp_table(x, x0, x1, table):
        import torch
        n = table.shape[0]
        u = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0) * (n - 1)
        i = u.floor().long().clamp(max=n - 2)
        f = u - i
        return table[i] * (1 - f) + table[i + 1] * f

    def radiance_t(self, t_k):
        lg, _ = self._tables(t_k)
        return self._lerp_table(t_k, self.t_min, self.t_max, lg)

    def temperature_t(self, l):
        _, tl = self._tables(l)
        return self._lerp_table(l, self.l_min, self.l_max, tl)


_DEFAULT = None


def default_lut():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = BandLUT()
    return _DEFAULT


def c_to_k(t_c):
    return t_c + 273.15


def counts_from_kelvin(t_k):
    """Lepton TLinear, 0.01 K per count."""
    return np.clip(np.rint(np.asarray(t_k) * 100.0), 0, 65535).astype(np.uint16)


def fog_transmission(path_m, beta_per_m):
    """Beer-Lambert in-band transmission for a homogeneous medium."""
    return np.exp(-np.asarray(beta_per_m) * np.asarray(path_m))


def apply_fog(l_surface, tau, t_mist_k, lut=None):
    """Apparent radiance through an isothermal absorbing/emitting medium.

    L = tau * L_surface + (1 - tau) * B(T_mist). Scattering of ambient radiation
    into the path is folded into the emission term (the room is close to a
    black-body cavity at air temperature, so in-scattered radiance ~ B(T_air)).
    """
    lut = lut or default_lut()
    return tau * l_surface + (1.0 - tau) * lut.radiance(t_mist_k)
