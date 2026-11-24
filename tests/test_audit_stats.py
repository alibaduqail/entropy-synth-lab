import math

import numpy as np
import pytest

from synthlab.audit import metrics, stats


def test_partial_corr_removes_common_cause():
    rng = np.random.default_rng(1)
    z = rng.normal(size=4000)
    x = z + 0.5 * rng.normal(size=4000)
    y = z + 0.5 * rng.normal(size=4000)
    assert np.corrcoef(x, y)[0, 1] > 0.7
    pc = stats.partial_corr(x, y, [z])
    assert abs(pc["r"]) < 0.05
    assert pc["dof"] == 4000 - 3


def test_partial_corr_keeps_direct_effect():
    rng = np.random.default_rng(2)
    z = rng.normal(size=2000)
    x = rng.normal(size=2000)
    y = 2 * x + z + 0.1 * rng.normal(size=2000)
    pc = stats.partial_corr(x, y, [z])
    assert pc["r"] > 0.95 and pc["p"] < 1e-10


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 2))
    y = 1.5 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.01 * rng.normal(size=500)
    fit = stats.ols(y, X, ["a", "b"])
    assert fit["a"]["beta"] == pytest.approx(2.0, abs=0.01)
    assert fit["b"]["beta"] == pytest.approx(-0.5, abs=0.01)
    assert fit["intercept"]["beta"] == pytest.approx(1.5, abs=0.01)
    assert fit["_model"]["r2"] > 0.999


def test_one_hot_shapes():
    d = stats.one_hot(["a", "b", "a", "c"])
    assert d.shape == (4, 2)
    assert stats.one_hot(["a", "a"]).shape == (2, 0)


def test_lag1_and_effective_n():
    rng = np.random.default_rng(4)
    e = rng.normal(size=20000)
    x = np.zeros_like(e)
    for i in range(1, e.size):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert stats.lag1_autocorr(x) == pytest.approx(0.8, abs=0.03)
    assert abs(stats.lag1_autocorr(e)) < 0.03
    assert stats.effective_n(100, 0.0) == 100
    assert stats.effective_n(100, 0.5) == pytest.approx(100 / 3)


def test_icc_detects_clusters():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(50), 40)
    clustered = rng.normal(size=50)[groups] + 0.1 * rng.normal(size=groups.size)
    independent = rng.normal(size=groups.size)
    hi = stats.icc_oneway(clustered, groups)
    lo = stats.icc_oneway(independent, groups)
    assert hi["icc"] > 0.95 and hi["effective_n"] < 60
    assert abs(lo["icc"]) < 0.05


def test_blocked_permutation_ignores_between_block_drift():
    rng = np.random.default_rng(6)
    blocks = np.repeat(np.arange(20), 10)
    # x and y both drift with the block, but are unrelated inside a block
    x = blocks + rng.normal(size=blocks.size)
    y = 3.0 * blocks + rng.normal(size=blocks.size)
    assert np.corrcoef(x, y)[0, 1] > 0.9
    res = stats.blocked_permutation_slope(x, y, blocks, n_perm=500, seed=1)
    assert res["p"] > 0.05
    y2 = y + 2.0 * x
    res2 = stats.blocked_permutation_slope(x, y2, blocks, n_perm=500, seed=1)
    assert res2["p"] < 0.01 and res2["slope"] == pytest.approx(2.0, abs=0.3)


def test_power_matches_textbook_values():
    # d = 0.5 needs 64 per group for 80 % power (Cohen); d = 1.0 needs 17
    assert stats.n_for_power(0.5, 1.0) == 64
    assert stats.n_for_power(1.0, 1.0) == 17
    assert stats.power_two_sample(0.5, 1.0, 64) == pytest.approx(0.80, abs=0.01)
    assert stats.power_two_sample(0.0, 1.0, 64) == pytest.approx(0.05, abs=0.005)
    mde = stats.min_detectable_effect(1.0, 64)
    assert mde == pytest.approx(0.5, abs=0.01)


def test_min_detectable_effect_with_three_repeats_is_large():
    # the Guide's "at least 3 repeats": need > 3 SD separation
    assert stats.min_detectable_effect(1.0, 3) > 3.0


def test_two_level_sd():
    assert stats.two_level_sd(1.0, 0.0, 4) == pytest.approx(0.5)
    assert stats.two_level_sd(0.0, 2.0, 100) == pytest.approx(2.0)


def test_welch_and_cohens_d():
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7])
    w = stats.welch(a, b)
    assert w["diff"] == pytest.approx(2.0)
    assert w["d"] == pytest.approx(2.0 / math.sqrt(2.5))


# ---- metric reference implementations ------------------------------------
def test_bie_is_zero_for_flat_image_and_falls_with_noise():
    scene = metrics.synthetic_thermal_scene(n_people=2, seed=0)
    clean = metrics.bie(metrics.to_u8(scene))["value"]
    rng = np.random.default_rng(0)
    noisy = metrics.bie(metrics.to_u8(scene + rng.normal(0, 1.0, scene.shape)))["value"]
    flat = metrics.bie(metrics.to_u8(np.full((120, 160), 24.0)))["value"]
    assert flat == 0.0
    assert clean > 0
    assert noisy < clean          # "entropy" goes DOWN when disorder goes up


def test_global_shannon_rises_with_noise():
    scene = metrics.synthetic_thermal_scene(n_people=1, seed=0)
    rng = np.random.default_rng(0)
    a = metrics.global_shannon_bits(metrics.to_u8(scene))
    b = metrics.global_shannon_bits(metrics.to_u8(scene + rng.normal(0, 1.0, scene.shape)))
    assert b > a


def test_eigen_entropy_limits():
    lam = np.array([[1.0, 0.0, 0.0],      # line
                    [1.0, 1.0, 0.0],      # plane
                    [1.0, 1.0, 1.0],      # isotropic scatter
                    [1.0, 2 / 3, 1 / 3]])  # equal L/P/S mix
    e = metrics.eigen_entropies(lam)
    assert np.allclose(e["H_DS"][:3], 0.0, atol=1e-9)        # H_DS blind to isotropy
    assert e["H_DS"][3] == pytest.approx(math.log(3))
    assert e["H_eigen"][0] == pytest.approx(0.0, abs=1e-9)
    assert e["H_eigen"][1] == pytest.approx(math.log(2))
    assert e["H_eigen"][2] == pytest.approx(math.log(3))
