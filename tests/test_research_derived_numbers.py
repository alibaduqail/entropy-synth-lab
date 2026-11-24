import numpy as np

from synthlab.research.derived_numbers import bie, p840_kl


def test_p840_kl_60ghz_range():
    assert 1.6 < p840_kl(60, 293.15) < 1.9
    assert p840_kl(60, 273.15) > p840_kl(60, 293.15)


def test_bie_falls_and_shannon_rises_with_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:120, 0:160]
    base = 70 + 100 * np.exp(-(((xx - 80) / 12.0) ** 2 + ((yy - 60) / 25.0) ** 2))
    clean = bie(np.rint(base))
    noisy = bie(np.rint(np.clip(base + rng.normal(0, 10, base.shape), 0, 255)))
    assert noisy["bie"] < clean["bie"]
    assert noisy["global_shannon"] > clean["global_shannon"]
