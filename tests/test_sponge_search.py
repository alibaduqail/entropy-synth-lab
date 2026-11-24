import importlib.util
import os

import numpy as np
import pytest

from synthlab.sponge_search import entropy, scenes
from synthlab.sponge_search.search import HT_BAND, dim_of, render, score

ORIG = "/Users/alibaduqail/Documents/thermal camera research work/scripts/experimentA.py"


def test_tlinear_roundtrip():
    assert scenes.celsius_to_raw(22.0) == 29515
    assert abs(float(scenes.raw_to_celsius(np.uint16(29515))) - 22.0) < 1e-3


def test_layout_frames_format_and_plausible_range():
    rng = np.random.default_rng(0)
    for i in range(20):
        frames, scene = scenes.render_layout(rng.random(scenes.DIM_LAYOUT), 4, seed=i)
        assert frames.shape == (4, 120, 160) and frames.dtype == np.uint16
        t = scenes.raw_to_celsius(frames)
        assert t.min() > 16.0 and t.max() < scenes.T_OBJ_MAX + 1.0
        assert scene["blur"] >= scenes.PSF_SIGMA_PX
        assert not np.array_equal(frames[0], frames[1])  # temporal noise differs


def test_render_is_deterministic():
    x = np.random.default_rng(3).random(scenes.DIM_LAYOUT)
    a, _ = scenes.render_layout(x, 2, seed=7)
    b, _ = scenes.render_layout(x, 2, seed=7)
    assert np.array_equal(a, b)


def test_person_is_warmer_than_background():
    x = np.zeros(scenes.DIM_LAYOUT)
    x[0] = 0.5
    x[18:25] = [1, 0.0, 0.5, 0.5, 0.8, 0.5, 0.6]
    clean = scenes.render_clean(scenes.decode_layout(x))
    assert clean.max() > clean[2, 2] + 5.0


def test_panel_and_blocks_tiers():
    rng = np.random.default_rng(1)
    for tier in ("panel", "blocks", "layout"):
        frames, _ = render(tier, rng.random(dim_of(tier)), 3, 0)
        assert frames.shape == (3, 120, 160) and frames.dtype == np.uint16
    f, s = scenes.render_panel(rng.random(scenes.DIM_PANEL))
    assert scenes.raw_to_celsius(f).max() < scenes.T_OBJ_MAX + 1.0


def test_u8_roundtrip_through_pipeline_mapping():
    img = np.random.default_rng(2).integers(0, 256, (120, 160)).astype(np.uint8)
    assert np.array_equal(scenes.pipeline_u8(scenes.u8_to_raw(img)), img)


def test_real_background_backend():
    bg = np.full((120, 160), 27.0, np.float32)
    x = np.full(scenes.DIM_LAYOUT, 0.5)
    frames, _ = scenes.render_layout(x, 2, 0, background=bg)
    assert frames.shape == (2, 120, 160)


def test_score_modes():
    base = {"frame_ms": 40.0, "energy_mj": 300.0}
    m = {"frame_ms": 45.0, "energy_mj": 340.0, "ht": 0.015, "n_cand": 10, "n_cand10": 20}
    assert score("latency", m, base) == pytest.approx(5.0)
    assert score("energy", m, base) == pytest.approx(40.0)
    assert score("stealth", m, base) == pytest.approx(40.0)
    loud = dict(m, ht=HT_BAND[1] + 0.01)
    assert score("stealth", loud, base) < score("stealth", m, base)
    assert score("decouple", dict(m, frame_ms=40.1), base) > score("decouple", m, base)


def test_bie_matches_original_pipeline():
    if not os.path.exists(ORIG):
        pytest.skip("original experimentA.py not present")
    spec = importlib.util.spec_from_file_location("expa", ORIG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(4)
    for i in range(5):
        frames, _ = scenes.render_layout(rng.random(scenes.DIM_LAYOUT), 1, i)
        img = mod.radiometric_bie_image(frames[0])
        assert np.array_equal(img, scenes.pipeline_u8(frames[0]))
        assert entropy.bie(img) == pytest.approx(mod.block_image_entropy(img)["value"], rel=1e-9)
        assert entropy.shannon_u8(img) == pytest.approx(mod.shannon_entropy(img), rel=1e-9)
