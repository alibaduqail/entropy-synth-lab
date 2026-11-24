"""Measured Lepton background / drift / noise model, fitted from empty-room (A0) recordings.

Decomposition of an A0 stack (counts, 0.01 K):
    frame_t = B + sum_k a_k(t) M_k + sigma * n_t
B        temporal median of the session's build trials (static scene + fixed pattern).
M_k      low-rank drift modes (PCA of frame - trial median); mode 0 is ~uniform offset.
         a_k(t) holds slow drift and the FFC (shutter) transients; real coefficient series
         are stored and replayed, because FFC steps are not a stationary process.
sigma    per-pixel temporal SD map of what is left.
n_t      unit-variance field with measured 2-D spatial power spectrum P(k).  Per spatial
         frequency it is a slow AR(1) part (fraction f(k), coefficient rho(k)) plus a white
         part (1-f(k)); f and rho are solved from the measured lag-1 and lag-4 temporal
         correlations.  Row/column-correlated noise lives on the ky=0 / kx=0 lines of P(k),
         so it is reproduced without a separate term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

H, W = 120, 160
N_MODES = 3


@dataclass
class SessionModel:
    name: str
    bg: np.ndarray            # (H,W) float32 counts
    modes: np.ndarray         # (K,H,W) float32, unit L2 norm * sqrt(H*W)  (so coef is in counts)
    coefs: list               # list of (N_i,K) float32 real coefficient series, one per build trial
    sigma: np.ndarray         # (H,W) float32 counts
    psd: np.ndarray           # (H,W) float32, mean |FFT(n)|^2 / (H*W), unit mean
    rho: np.ndarray           # (H,W) float32 AR(1) coefficient of the slow part, per frequency
    frac: np.ndarray          # (H,W) float32 variance fraction of the slow part, per frequency
    stats: dict = field(default_factory=dict)

    def save(self, path):
        np.savez_compressed(
            path, name=self.name, bg=self.bg, modes=self.modes, sigma=self.sigma, psd=self.psd,
            rho=self.rho, frac=self.frac, coef_cat=np.concatenate(self.coefs).astype(np.float32),
            coef_len=np.array([len(c) for c in self.coefs]), stats=np.array(repr(self.stats)),
        )

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        lens = z["coef_len"]
        cat = z["coef_cat"]
        idx = np.cumsum(lens)[:-1]
        import ast
        return SessionModel(str(z["name"]), z["bg"], z["modes"], np.split(cat, idx), z["sigma"],
                            z["psd"], z["rho"], z["frac"], ast.literal_eval(str(z["stats"])))

    # ---------- sampling ----------
    def sample_noise(self, n, rng, mult=1.0, burn=8, round_counts=False):
        """Return (n,H,W) float32 noise in counts matching sigma map, P(k) and rho(k)."""
        amp = np.sqrt(self.psd)                    # unit-variance shaping
        rho, frac = self.rho, self.frac
        innov = np.sqrt(1.0 - rho**2)
        a_slow, a_white = np.sqrt(frac) * amp, np.sqrt(1.0 - frac) * amp

        def white():
            return np.fft.fft2(rng.standard_normal((H, W)))

        state = white()
        out = np.empty((n, H, W), np.float32)
        for t in range(-burn, n):
            state = rho * state + innov * white()
            if t >= 0:
                out[t] = np.real(np.fft.ifft2(state * a_slow + white() * a_white))
        out *= (self.sigma * mult)[None]
        return np.rint(out) if round_counts else out

    def sample_drift(self, n, rng, mult=1.0, trial=None, start=None):
        """Replay a real drift-coefficient segment (slow drift + any FFC transient): (n,H,W)."""
        if mult == 0:
            return np.zeros((n, H, W), np.float32), {"drift_trial": None}
        ok = [i for i, c in enumerate(self.coefs) if len(c) >= n]
        if not ok:
            raise ValueError(f"no stored drift series with >= {n} frames")
        i = int(rng.choice(ok)) if trial is None else trial
        c = self.coefs[i]
        s = int(rng.integers(0, len(c) - n + 1)) if start is None else start
        seg = c[s:s + n] * mult
        field_ = np.tensordot(seg, self.modes, axes=(1, 0)).astype(np.float32)
        return field_, {"drift_trial": i, "drift_start": s}


def _unit_fields(resid, sigma):
    return resid / sigma[None]


def spectrum_and_rho(n_unit):
    """2-D power spectrum and lag-1 / lag-4 temporal correlations per spatial frequency."""
    F = np.fft.fft2(n_unit, axes=(1, 2))
    p = np.mean(np.abs(F) ** 2, axis=0)
    c1 = np.mean(np.real(F[1:] * np.conj(F[:-1])), axis=0) / np.maximum(p, 1e-12)
    c4 = np.mean(np.real(F[4:] * np.conj(F[:-4])), axis=0) / np.maximum(p, 1e-12)
    return p, c1, c4


def slow_white_split(c1, c4):
    """Solve corr(lag) = f * rho**lag from lags 1 and 4; returns (rho, f) clipped to valid."""
    c1c = np.clip(c1, 1e-3, 0.999)
    c4c = np.clip(c4, 1e-4, None)
    rho = np.clip((np.minimum(c4c, c1c) / c1c) ** (1.0 / 3.0), 0.0, 0.995)
    frac = np.clip(c1c / np.maximum(rho, 1e-3), 0.0, 1.0)
    frac = np.where(c1 <= 1e-3, 0.0, frac)
    return rho.astype(np.float32), frac.astype(np.float32)


def fit_session(name, stacks, n_modes=N_MODES, max_pca_frames=1500, seed=0):
    """Fit a SessionModel from a list of uint16 (N,H,W) empty-room stacks of ONE session."""
    rng = np.random.default_rng(seed)
    stacks = [np.asarray(s, np.float32) for s in stacks]
    meds = [np.median(s, axis=0) for s in stacks]
    bg = np.median(np.stack(meds), axis=0).astype(np.float32)
    # residuals about the SESSION background so that between-trial drift is in the modes too
    resid = [s - bg[None] for s in stacks]
    pool = np.concatenate(resid)
    if len(pool) > max_pca_frames:
        pool = pool[rng.choice(len(pool), max_pca_frames, replace=False)]
    X = pool.reshape(len(pool), -1)
    # mode 0 forced to the uniform offset; remaining modes by PCA of what is left
    m0 = np.ones(H * W, np.float32) / np.sqrt(H * W)
    Xr = X - np.outer(X @ m0, m0)
    # PCA through the (frames x frames) Gram matrix: cheap, same leading modes as an SVD
    ev, eu = np.linalg.eigh((Xr @ Xr.T).astype(np.float64))
    order = np.argsort(ev)[::-1][: max(8, n_modes)]
    sv = np.sqrt(np.maximum(ev[order], 1e-12))
    vt = ((eu[:, order].T @ Xr) / sv[:, None]).astype(np.float32)
    total_var = float(np.sum(Xr.astype(np.float64) ** 2))
    basis = np.vstack([m0[None], vt[: n_modes - 1]])
    modes = (basis * np.sqrt(H * W)).reshape(n_modes, H, W).astype(np.float32)
    coefs, noise = [], []
    for r in resid:
        a = (r.reshape(len(r), -1) @ basis.T) / np.sqrt(H * W)
        coefs.append(a.astype(np.float32))
        n = r - np.tensordot(a, modes, axes=(1, 0))
        n -= n.mean(0, keepdims=True)       # residual static offset of this trial vs session bg
        noise.append(n)
    var = np.mean([np.var(n, axis=0) for n in noise], axis=0)
    sigma = np.sqrt(var).astype(np.float32)
    ps, c1s, c4s, wts = [], [], [], []
    for n in noise:
        p, c1, c4 = spectrum_and_rho(_unit_fields(n, sigma))
        ps.append(p); c1s.append(c1); c4s.append(c4); wts.append(len(n))
    psd = np.average(ps, axis=0, weights=wts) / (H * W)
    psd = (psd / psd.mean()).astype(np.float32)
    rho, frac = slow_white_split(np.average(c1s, axis=0, weights=wts),
                                 np.average(c4s, axis=0, weights=wts))
    allc = np.concatenate(coefs)
    cat = np.concatenate([n[:: max(1, len(n) // 100)] for n in noise])
    u = (cat / sigma[None]).ravel()
    stats = {
        "n_trials": len(stacks), "n_frames": int(sum(len(s) for s in stacks)),
        "bg_mean_c": float(bg.mean() * 0.01 - 273.15),
        "sigma_median_counts": float(np.median(sigma)),
        "sigma_p05_p95_counts": [float(np.percentile(sigma, 5)), float(np.percentile(sigma, 95))],
        "pca_sv_fraction_top8": [float(v) for v in (sv[:8] ** 2 / total_var)],
        "coef_sd_counts": [float(v) for v in allc.std(0)],
        "noise_excess_kurtosis": float(np.mean(u**4) / np.mean(u**2) ** 2 - 3.0),
        "rho_median": float(np.median(rho)), "slow_fraction_median": float(np.median(frac)),
        "row_noise_var_fraction": float(psd[0, 1:].sum() / psd.sum()),   # ky=0 line: column stripes
        "col_noise_var_fraction": float(psd[1:, 0].sum() / psd.sum()),   # kx=0 line: row stripes
        "seed": seed,
    }
    return SessionModel(name, bg, modes, coefs, sigma, psd, rho, frac, stats)


def detect_ffc_events(frame_mean_counts, thresh=40.0):
    """Indices where the frame mean jumps by more than `thresh` counts in one frame (FFC)."""
    d = np.diff(np.asarray(frame_mean_counts, np.float64))
    return [int(i) + 1 for i in np.nonzero(np.abs(d) > thresh)[0]]
