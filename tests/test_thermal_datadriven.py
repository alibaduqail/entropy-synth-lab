"""Tests for the real-data-driven thermal generator.  They use small synthetic stand-in data,
so they run without the (private, 7 GB) recordings."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from synthlab.thermal_datadriven import bank, compose, layers, library, metrics, sensor_model

H, W = 120, 160
ORIG = Path("/Users/alibaduqail/Documents/thermal camera research work/scripts/experimentA.py")


def fake_empty_stack(n=160, seed=0, sd=8.0, rho=0.8):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 29800 + 0.5 * xx + 0.3 * yy
    x = np.zeros((n, H, W))
    state = rng.standard_normal((H, W))
    for t in range(n):
        state = rho * state + np.sqrt(1 - rho ** 2) * rng.standard_normal((H, W))
        col = rng.standard_normal((1, W)) * 0.5
        x[t] = bg + sd * (state + col) / np.sqrt(1.25) + 20 * np.sin(t / 30.0)
    return np.rint(x).astype(np.uint16), bg


def test_bie_matches_original_pipeline():
    if not ORIG.exists():
        pytest.skip("parent project not present")
    spec = importlib.util.spec_from_file_location("experimentA_orig", ORIG)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["experimentA_orig"] = mod
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(3)
    raw = (29500 + 300 * rng.random((H, W)) + 900 * (rng.random((H, W)) > 0.9)).astype(np.uint16)
    a = mod.block_image_entropy(mod.radiometric_bie_image(raw), 15, 1.0)["value"]
    assert metrics.ht(raw) == pytest.approx(a, rel=1e-12)


def test_bie_flat_image_is_zero():
    assert metrics.ht(np.full((H, W), 29515, np.uint16)) == 0.0


def test_fit_and_sample_reproduce_noise_statistics(tmp_path):
    stacks = [fake_empty_stack(seed=s)[0] for s in range(3)]
    m = sensor_model.fit_session("fake", stacks, max_pca_frames=300)
    assert m.bg.shape == (H, W) and m.modes.shape == (3, H, W)
    assert 6.0 < float(np.median(m.sigma)) < 10.0
    rng = np.random.default_rng(1)
    noise = m.sample_noise(200, rng)
    assert noise.shape == (200, H, W)
    assert np.std(noise) == pytest.approx(float(np.sqrt(np.mean(m.sigma ** 2))), rel=0.1)
    lag1 = np.mean(noise[1:] * noise[:-1]) / np.mean(noise ** 2)
    assert 0.45 < lag1 < 0.8                      # 0.8 * (1/1.25) = 0.64 expected (white stripes)
    # column-stripe component survives: column means vary more than white noise would allow
    assert noise.mean(1).std() > 2.5 * np.std(noise) / np.sqrt(H)
    m.save(tmp_path / "m.npz")
    m2 = sensor_model.SessionModel.load(tmp_path / "m.npz")
    assert np.allclose(m2.psd, m.psd) and len(m2.coefs) == 3
    d, info = m2.sample_drift(64, rng)
    assert d.shape == (64, H, W) and "drift_start" in info
    assert np.all(m2.sample_drift(64, rng, mult=0.0)[0] == 0)


def test_noise_multiplier_scales_sd():
    m = sensor_model.fit_session("fake", [fake_empty_stack()[0]], max_pca_frames=160)
    a = m.sample_noise(40, np.random.default_rng(0), 1.0)
    b = m.sample_noise(40, np.random.default_rng(0), 3.0)
    assert np.allclose(b, 3 * a, atol=1e-3)


def person_frame(bg, x0=60, x1=90, y0=30, t=30800.0):
    f = bg.copy()
    f[y0:, x0:x1] = t
    return f


def test_extract_frame_recovers_person_matte():
    _, bg = fake_empty_stack(4)
    bg = bg.astype(np.float32)
    frame = person_frame(bg) + 37.0            # uniform drift must be absorbed by the offset
    r = layers.extract_frame(frame, bg)
    assert abs(r["offset"] - 37.0) < 5
    truth = np.zeros((H, W), bool)
    truth[30:, 60:90] = True
    hard = r["alpha"] >= 0.5
    iou = (hard & truth).sum() / (hard | truth).sum()
    assert iou > 0.9
    assert np.allclose(r["fg"][truth & (r["alpha"] == 1)], 30837.0, atol=1.0)
    meta = layers.layer_meta(np.rint(r["alpha"] * 255).astype(np.uint8), r["label"], r["lamp"], 1)
    assert meta["cut_bottom"] and abs(meta["body_width_px"] - 30) <= 3


def test_split_two_touching_people():
    _, bg = fake_empty_stack(4)
    bg = bg.astype(np.float32)
    f = bg.copy()
    f[40:, 20:70] = 31500
    f[40:, 90:140] = 31500
    f[100:, 70:90] = 31500                     # shoulders touch
    r = layers.extract_frame(f, bg, max_people=2)
    assert r["label"].max() == 2
    assert np.median(np.nonzero(r["label"] == 1)[1]) < np.median(np.nonzero(r["label"] == 2)[1])


def build_fake_world(tmp_path, n=40):
    stack, bg = fake_empty_stack(n + 60)
    m = sensor_model.fit_session("hl", [stack], max_pca_frames=100)
    mdir, ldir = tmp_path / "models", tmp_path / "layers"
    mdir.mkdir(); ldir.mkdir()
    m.save(mdir / "session_hl.npz")
    np.savez(mdir / "lamp_surface_hl.npz", surface=np.zeros((H, W), np.float32), hl_on_bg=bg)
    a1 = np.stack([person_frame(bg.astype(np.float32), 60 + t // 8, 90 + t // 8) for t in range(n)])
    a1[:, 35:45, 70:80] = 33500.0              # a "lamp": hotter than 38 C
    ex = layers.extract_trial(a1, m.bg, 1)
    metas = [[layers.layer_meta(ex["alpha"][i], ex["label"][i], ex["lamp"][i], 1)] for i in range(n)]
    np.savez(ldir / "hl_on__fake.npz", alpha=ex["alpha"], fg=ex["fg"], label=ex["label"],
             lamp=ex["lamp"], offset=ex["offset"], src_bg=m.bg, meta=np.array(json.dumps(metas)))
    a2 = np.stack([person_frame(bg.astype(np.float32), 30, 55, 20, 30600.0) for _ in range(n)])
    ex = layers.extract_trial(a2, m.bg, 1)
    metas = [[layers.layer_meta(ex["alpha"][i], ex["label"][i], ex["lamp"][i], 1)] for i in range(n)]
    np.savez(ldir / "rj_A1__fake.npz", alpha=ex["alpha"], fg=ex["fg"], label=ex["label"],
             lamp=ex["lamp"], offset=ex["offset"], src_bg=m.bg, meta=np.array(json.dumps(metas)))
    return compose.Generator(model_dir=mdir, layer_dir=ldir)


def test_generator_controls_and_labels(tmp_path):
    g = build_fake_world(tmp_path)
    f0, m0 = g.generate(compose.SceneSpec(n_frames=16, n_people=0, seed=1))
    assert f0.dtype == np.uint16 and f0.shape == (16, H, W)
    assert all(l["n_people"] == 0 for l in m0["labels"])
    f2, m2 = g.generate(compose.SceneSpec(n_frames=16, n_people=2, seed=1))
    assert max(l["n_people"] for l in m2["labels"]) >= 1
    assert all(not p["source"].startswith("hl_on") for p in m2["people"])   # lamp-free clips only
    for l in m2["labels"]:
        for x0, y0, x1, y1 in l["boxes"]:
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
    assert f2.mean() > f0.mean()                       # people add heat
    # the labelled box really contains the warm person
    x0, y0, x1, y1 = m2["labels"][0]["boxes"][0]
    assert f2[0, y0:y1, x0:x1].mean() > f2[0].mean()
    # determinism and seed sensitivity
    f2b, _ = g.generate(compose.SceneSpec(n_frames=16, n_people=2, seed=1))
    assert np.array_equal(f2, f2b)
    f2c, _ = g.generate(compose.SceneSpec(n_frames=16, n_people=2, seed=2))
    assert not np.array_equal(f2, f2c)
    # lamp: person-holding-lamp clips are used and the frame gets hotter than skin
    fl, ml = g.generate(compose.SceneSpec(n_frames=16, n_people=1, lamp=1.0, people_hold_lamp=True, seed=3))
    assert ml["people"][0]["source"].startswith("hl_on") and ml["lamp_on"]
    assert fl.max() > compose.c2k(40.0)
    # noise multiplier raises temporal SD of an empty scene; background is fixed across frames
    lo, _ = g.generate(compose.SceneSpec(n_frames=32, noise_mult=1.0, drift_mult=0.0, seed=5))
    hi, _ = g.generate(compose.SceneSpec(n_frames=32, noise_mult=4.0, drift_mult=0.0, seed=5))
    assert np.median(hi.std(0)) > 3.0 * np.median(lo.std(0))
    # warm objects are labelled
    fw, mw = g.generate(compose.SceneSpec(n_frames=8, n_warm_objects=2, seed=6))
    assert len(mw["objects"]) == 2 and fw.max() > f0.max()


def test_temporal_coherence_people_move_smoothly(tmp_path):
    g = build_fake_world(tmp_path)
    _, m = g.generate(compose.SceneSpec(n_frames=32, n_people=1, seed=4, motion_px=10))
    cx = [0.5 * (l["boxes"][0][0] + l["boxes"][0][2]) for l in m["labels"] if l["boxes"]]
    assert len(cx) >= 30
    assert np.max(np.abs(np.diff(cx))) <= 4.0          # no teleporting between frames
    assert np.ptp(cx) >= 2.0                           # but the person does move


def test_bank_format(tmp_path):
    g = build_fake_world(tmp_path)
    frames, man = g.generate(compose.SceneSpec(n_frames=8, n_people=1, seed=0))
    out = tmp_path / "bank"
    bank.write_bank(out, frames, man)
    back = np.load(out / "frames.npy")
    man2 = json.loads((out / "manifest.json").read_text())
    assert back.dtype == np.uint16 and back.shape == (8, H, W)
    assert man2["n_frames"] == 8 and len(man2["labels"]) == 8 and "seed" in man2
    assert 15 < man2["temp_mean_c"] < 40
    with pytest.raises(ValueError):
        bank.write_bank(tmp_path / "bad", frames.astype(np.float32), man)


def test_library_clip_ping_pong(tmp_path):
    g = build_fake_world(tmp_path)
    tr = g.library().tracks[0]
    idx = tr.clip_indices(tr.n - 3, 10)
    assert idx.max() == tr.n - 1 and np.all(np.abs(np.diff(idx)) <= 1)
