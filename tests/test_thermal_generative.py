"""Tests for the thermal-generative track: data mapping (numpy only) and model shapes (torch).
Runs under pytest, or directly: python3 tests/test_thermal_generative.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthlab.thermal_generative import mapping as M  # noqa: E402


def test_counts_celsius_anchor():
    assert abs(M.counts_to_celsius(np.array([29515]))[0] - 22.0) < 1e-9
    assert M.celsius_to_counts(22.0) == 29515


def test_mapping_roundtrip_exact_all_plausible_counts():
    raw = np.arange(20000, 48000, dtype=np.uint16)  # -73 C .. 207 C
    back = M.x_to_counts(M.counts_to_x(raw))
    assert back.dtype == np.uint16 and np.array_equal(back, raw)


def test_mapping_monotonic_and_affine_in_pipeline_range():
    raw = np.arange(20000, 48000, dtype=np.uint16)
    x = M.counts_to_x(raw).astype(np.float64)
    assert np.all(np.diff(x) > 0)
    lo, hi = M.celsius_to_counts(10.0), M.celsius_to_counts(50.0)
    seg = M.counts_to_x(np.arange(lo, hi + 1, dtype=np.uint16)).astype(np.float64)
    assert np.allclose(np.diff(seg), 0.01 / M.SCALE_C, atol=1e-6)
    assert abs(float(M.counts_to_x(np.array([M.celsius_to_counts(25.0)]))[0])) < 1e-6


def test_bie_image_matches_pipeline_definition():
    raw = M.celsius_to_counts(np.array([[5.0, 10.0, 30.0, 50.0, 90.0]]))
    assert M.bie_image(raw).tolist() == [[0, 0, 128, 255, 255]]


def test_bie_flat_is_zero_and_structure_raises_it():
    flat = np.full((120, 160), M.celsius_to_counts(24.0), np.uint16)
    assert M.ht_of_counts(flat) == 0.0
    rng = np.random.default_rng(0)
    t = 24.0 + rng.normal(0, 0.05, (120, 160))
    t[40:100, 60:90] += 8.0
    assert M.ht_of_counts(M.celsius_to_counts(t)) > M.ht_of_counts(M.celsius_to_counts(t - (t > 30) * 8.0))


def test_bie_reference_value():
    # hand-checkable case: left half 0, right half 255 on a 30x30 image with 15-px blocks:
    # every block is constant -> modulation term 0 -> BIE 0.
    img = np.zeros((30, 30), np.uint8)
    img[:, 15:] = 255
    assert M.block_image_entropy(img) == 0.0
    # two-level blocks: M=100/255, entropy 1 bit, block sd 50, global sd 50, mean 50
    img = np.zeros((30, 30), np.uint8)
    img[::2] = 100
    m = 100 / 255.0
    adp = 1 - abs(50 - 127.5) / 127.5
    sh = M._shannon(img[:15, :15])
    sd = float(img[:15, :15].std())
    exp = adp * (-m * np.log(m) / (1 + sh)) * (float(img.std()) / (1 + sd))
    assert abs(M.block_image_entropy(img) - exp) < 1e-12


def test_split_holds_out_whole_trials():
    from synthlab.thermal_generative.data import split_of
    runs = list(range(1, 27))
    s = [split_of(0, 0, r, runs) for r in runs]
    assert s.count("test") == 5 and s[4] == "test"
    assert [split_of(1, 1, r, [1, 2, 3]) for r in (1, 2, 3)] == ["train", "train", "test"]


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_torch_mapping_matches_numpy():
    torch = _torch()
    if torch is None:
        return
    raw = np.arange(27000, 45000, 7, dtype=np.uint16)
    x = M.counts_to_x_torch(torch.from_numpy(raw.astype(np.int32)))
    assert np.allclose(x.numpy(), M.counts_to_x(raw), atol=1e-5)
    back = M.x_to_counts_torch(torch.from_numpy(M.counts_to_x(raw).astype(np.float64)))
    assert np.array_equal(back.numpy().astype(np.uint16), raw)


def test_model_shapes_and_sampler():
    torch = _torch()
    if torch is None:
        return
    from synthlab.thermal_generative import diffusion as D
    from synthlab.thermal_generative.models import CVAE, UNet, null_cond
    torch.manual_seed(0)
    c = torch.tensor([[1, 1, 0], [0, 0, 2]])
    u = UNet(base=8)
    assert u(torch.randn(2, 1, 120, 160), torch.tensor([10.0, 900.0]), c).shape == (2, 1, 120, 160)
    assert u(torch.randn(2, 1, 120, 160), torch.tensor([10.0, 900.0]), null_cond(2, "cpu")).shape == (2, 1, 120, 160)
    v = CVAE(zdim=16, base=8)
    rec, mu, logvar = v(torch.randn(2, 1, 120, 160), c)
    assert rec.shape == (2, 1, 120, 160) and mu.shape == (2, 16) and logvar.shape == (2, 16)
    z = D.ddim_sample(u, c, steps=3, guidance=2.0, temperature=0.9)
    assert z.shape == (2, 1, 120, 160) and float(z.min()) >= D.Z_MIN and float(z.max()) <= D.Z_MAX
    assert torch.isfinite(D.loss_fn(u, torch.randn(2, 1, 120, 160), c))


def test_schedule_and_cond_dropout():
    torch = _torch()
    if torch is None:
        return
    from synthlab.thermal_generative import diffusion as D
    t = torch.linspace(0, 1, 50)
    ab = D.alpha_bar(t)
    assert abs(float(ab[0]) - 1.0) < 1e-6 and float(ab[-1]) < 1e-3 and bool(torch.all(ab[1:] <= ab[:-1]))
    z0, n = torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
    tt = torch.tensor([0.1, 0.4, 0.7, 0.9])
    zt, v = D.q_sample(z0, tt, n), D.v_target(z0, tt, n)
    a = D.alpha_bar(tt)[:, None, None, None]
    assert torch.allclose(a.sqrt() * zt - (1 - a).sqrt() * v, z0, atol=1e-5)  # v-param inverts to x0
    c = torch.zeros(4000, 3, dtype=torch.long)
    d = D.drop_cond(c, generator=torch.Generator().manual_seed(0))
    frac = (d != 0).float().mean(0)
    assert bool(torch.all((frac > 0.18) & (frac < 0.30)))  # 1-(0.85*0.9)=0.235


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(len(fns), "tests passed", "(torch tests skipped)" if _torch() is None else "")
