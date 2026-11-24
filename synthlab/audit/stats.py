"""Small, dependency-light statistics helpers for the experiment audit.

Everything here is numpy/scipy only so it can be unit-tested and reused on the
Jetson (scipy 1.8).  Functions return plain floats / dicts.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# regression / partial correlation
# ---------------------------------------------------------------------------
def _design(covars: Sequence[np.ndarray], n: int) -> np.ndarray:
    cols = [np.ones(n)]
    for c in covars:
        c = np.asarray(c, dtype=float)
        cols.extend([c] if c.ndim == 1 else list(c.T))
    return np.column_stack(cols)


def residualize(y: np.ndarray, covars: Sequence[np.ndarray]) -> np.ndarray:
    """Residuals of y after least-squares regression on covars (+ intercept)."""
    y = np.asarray(y, dtype=float)
    X = _design(covars, y.size)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(x, y, covars: Sequence[np.ndarray]) -> Dict[str, float]:
    """Pearson correlation of x and y after removing covars from both.

    Returns r, two-sided p (t-test with n - 2 - k dof) and dof.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = _design(covars, x.size)
    k = np.linalg.matrix_rank(X) - 1
    rx, ry = residualize(x, covars), residualize(y, covars)
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    r = float(rx @ ry) / den if den > 0 else float("nan")
    dof = x.size - 2 - k
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return {"r": r, "p": float("nan"), "dof": dof}
    t = r * math.sqrt(dof / (1 - r * r))
    return {"r": r, "p": float(2 * sps.t.sf(abs(t), dof)), "dof": dof}


def ols(y, X: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Ordinary least squares with classical standard errors.

    X must NOT include the intercept; one is added.  Returns per-term
    beta / se / t / p plus '_model' with r2, n, dof, sigma.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = y.size - np.linalg.matrix_rank(A)
    sigma2 = float(res @ res) / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {}
    for i, name in enumerate(["intercept", *names]):
        t = beta[i] / se[i] if se[i] > 0 else float("nan")
        out[name] = {"beta": float(beta[i]), "se": float(se[i]), "t": float(t),
                     "p": float(2 * sps.t.sf(abs(t), dof)) if np.isfinite(t) else float("nan")}
    sst = float(((y - y.mean()) ** 2).sum())
    out["_model"] = {"r2": 1 - float(res @ res) / sst if sst > 0 else float("nan"),
                     "n": int(y.size), "dof": int(dof), "sigma": math.sqrt(sigma2)}
    return out


def one_hot(labels: Iterable, drop_first: bool = True) -> np.ndarray:
    """Dummy-code a label vector (for block / round fixed effects)."""
    labels = np.asarray(list(labels))
    levels = sorted(set(labels.tolist()), key=str)
    if drop_first:
        levels = levels[1:]
    return np.column_stack([(labels == lv).astype(float) for lv in levels]) \
        if levels else np.zeros((labels.size, 0))


# ---------------------------------------------------------------------------
# time-order structure
# ---------------------------------------------------------------------------
def lag1_autocorr(x) -> float:
    """Lag-1 autocorrelation of a series taken in acquisition order."""
    x = np.asarray(x, float)
    x = x - x.mean()
    den = float(x @ x)
    return float(x[:-1] @ x[1:]) / den if den > 0 else float("nan")


def effective_n(n: int, rho1: float) -> float:
    """AR(1) effective sample size for a mean: n (1-rho)/(1+rho)."""
    rho1 = min(max(rho1, -0.99), 0.99)
    return n * (1 - rho1) / (1 + rho1)


def icc_oneway(values, groups) -> Dict[str, float]:
    """One-way random-effects ICC(1) and the design effect for cluster means.

    values: observations (e.g. frames); groups: cluster label (e.g. trial).
    deff = 1 + (m - 1) * icc, m = mean cluster size.
    """
    values, groups = np.asarray(values, float), np.asarray(groups)
    labs, inv = np.unique(groups, return_inverse=True)
    k = labs.size
    n_i = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=values) / n_i
    grand = values.mean()
    ssb = float((n_i * (means - grand) ** 2).sum())
    ssw = float(((values - means[inv]) ** 2).sum())
    N = values.size
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - float((n_i ** 2).sum()) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    m = float(n_i.mean())
    return {"icc": float(icc), "mean_cluster_size": m, "n_clusters": int(k),
            "design_effect": float(1 + (m - 1) * max(icc, 0.0)),
            "effective_n": float(N / (1 + (m - 1) * max(icc, 0.0)))}


# ---------------------------------------------------------------------------
# blocked permutation test
# ---------------------------------------------------------------------------
def blocked_permutation_slope(x, y, blocks, n_perm: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Permutation test of the within-block association between x and y.

    Statistic: OLS slope of y on x after centring both within each block.
    Null: x labels exchangeable WITHIN a block (the randomisation actually
    used when scenario order is shuffled inside each round).  Slow drift
    between blocks cannot create a false positive under this null.
    """
    x, y, blocks = np.asarray(x, float), np.asarray(y, float), np.asarray(blocks)
    _, inv = np.unique(blocks, return_inverse=True)
    cnt = np.bincount(inv).astype(float)

    def centre(v):
        return v - (np.bincount(inv, weights=v) / cnt)[inv]

    yc = centre(y)

    def slope(xv):
        xc = centre(xv)
        den = float(xc @ xc)
        return float(xc @ yc) / den if den > 0 else 0.0

    obs = slope(x)
    rng = np.random.default_rng(seed)
    idx_by_block = [np.flatnonzero(inv == b) for b in range(cnt.size)]
    hits = 0
    for _ in range(n_perm):
        xp = x.copy()
        for idx in idx_by_block:
            xp[idx] = x[rng.permutation(idx)]
        if abs(slope(xp)) >= abs(obs) - 1e-15:
            hits += 1
    return {"slope": obs, "p": (hits + 1) / (n_perm + 1), "n_perm": n_perm,
            "n_blocks": int(cnt.size)}


# ---------------------------------------------------------------------------
# effect sizes and power analysis
# ---------------------------------------------------------------------------
def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                   / (a.size + b.size - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else float("nan")


def welch(a, b) -> Dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = sps.ttest_ind(b, a, equal_var=False)
    return {"diff": float(b.mean() - a.mean()), "t": float(t), "p": float(p),
            "d": cohens_d(a, b)}


def power_two_sample(delta: float, sd: float, n: int, alpha: float = 0.05) -> float:
    """Power of a two-sided two-sample t-test, n per group, common sd.

    Uses the noncentral t distribution (exact for equal n, equal variance).
    """
    if n < 2 or sd <= 0:
        return float("nan")
    dof = 2 * n - 2
    ncp = abs(delta) / (sd * math.sqrt(2.0 / n))
    tcrit = sps.t.ppf(1 - alpha / 2, dof)
    if ncp > 30:            # nct is numerically unstable far out; power is 1 there
        return 1.0
    p = float(sps.nct.sf(tcrit, dof, ncp) + sps.nct.cdf(-tcrit, dof, ncp))
    return p if np.isfinite(p) else 1.0


def n_for_power(delta: float, sd: float, power: float = 0.8, alpha: float = 0.05,
                n_max: int = 100000) -> Optional[int]:
    """Smallest n per group reaching the requested power (None if > n_max)."""
    if delta == 0 or sd <= 0:
        return None
    lo, hi = 2, 2
    while power_two_sample(delta, sd, hi, alpha) < power:
        hi *= 2
        if hi > n_max:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power_two_sample(delta, sd, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def min_detectable_effect(sd: float, n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest |delta| detectable with n per group (bisection on power)."""
    lo, hi = 0.0, 50.0 * sd
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_two_sample(mid, sd, n, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


def two_level_sd(sd_trial: float, sd_session: float, trials_per_session: int) -> float:
    """SD of one session's condition mean when sessions add their own offset."""
    return math.sqrt(sd_session ** 2 + sd_trial ** 2 / trials_per_session)
