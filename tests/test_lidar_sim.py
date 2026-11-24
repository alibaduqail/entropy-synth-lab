import json
import math
import os

import numpy as np
import pytest

from synthlab.lidar_sim import bank, entropy, fog as fogmod
from synthlab.lidar_sim.scene import Scene, person
from synthlab.lidar_sim.sensor import SensorParams, coverage_vs_time, scan_dirs, simulate_frame


V0 = os.path.join(os.path.dirname(__file__), "..", "scene", "room_v0.json")


def test_scene_loads_and_raycast_hits_room():
    s = Scene.load(V0)
    d = np.array([[0, -1.0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    t, refl, cosi, pid = s.raycast(d)
    assert t[0] == pytest.approx(s.sensor_pos[1], abs=1e-9)             # south wall, nothing between
    assert t[2] == pytest.approx(s.size[2] - s.sensor_pos[2])           # ceiling
    assert t[3] == pytest.approx(s.size[0] - s.sensor_pos[0])           # east wall
    assert np.all(cosi > 0.99) and np.all(t > 0)
    assert t[1] < s.size[1] - s.sensor_pos[1] + 1e-9                    # north: wall or TV in front


def test_default_scene_loads_and_apertures_escape():
    s = Scene.load()
    d, _ = scan_dirs(SensorParams(), 0, 20000)
    t, refl, cosi, pid = s.raycast(s.dirs_to_world(d))
    assert np.all(np.isfinite(t)) and t.max() < np.linalg.norm(s.size) + 1e-6
    if s.apertures:
        assert 0.0 < (refl == 0).mean() < 0.5                           # some rays leave the room


def test_person_occludes_wall():
    s = Scene.load(V0)
    mark = np.array(s.slots["person_mark"][:2])
    v = np.append(mark, 1.2) - s.sensor_pos
    d = (v / np.linalg.norm(v))[None]
    t0 = s.raycast(d)[0][0]
    s2 = s.copy().add(person(mark))
    t1, _, _, pid = s2.raycast(d)
    assert t1[0] < t0 and t1[0] == pytest.approx(np.linalg.norm(v[:2]) - 0.17, abs=0.03)
    assert s2.prim_names()[pid[0]] == "person"


def test_scan_pattern_fov_rate_and_nonrepetition():
    sp = SensorParams()
    d, _ = scan_dirs(sp, 0, sp.pts_per_frame)
    assert d.shape == (20000, 3)
    el = np.degrees(np.arcsin(d[:, 2]))
    assert el.min() >= -7.0 - 1e-6 and el.max() <= 52.0 + 1e-6
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    assert np.histogram(az, bins=36, range=(0, 360))[0].min() > 0      # full 360 deg each frame
    cov = coverage_vs_time(sp, (0.1, 0.5, 1.0))
    assert cov[0.1] < cov[0.5] < cov[1.0]                                # coverage grows with time
    d2, _ = scan_dirs(sp, sp.pts_per_frame, sp.pts_per_frame)
    assert np.abs(d - d2).max() > 1e-3                                   # next frame differs


def test_clear_frame_format_and_ranges():
    s = Scene.load(V0)
    fr = simulate_frame(s, 0, np.random.default_rng(0), debug=True)
    assert fr["xyz"].dtype == np.float32 and fr["intensity"].dtype == np.uint8
    assert fr["tag"].dtype == np.uint8 and fr["xyz"].shape[0] == fr["tag"].shape[0]
    r = np.linalg.norm(fr["xyz"], axis=1)
    assert r.min() >= 0.05 and r.max() < 5.5
    assert 0.9 * 20000 < fr["xyz"].shape[0] <= 20000                     # single return, never more
    assert fr["n_fog"] == 0
    # plane fit residual of the south wall ~ range noise (2-3 cm)
    w = s.to_sensor(np.zeros((1, 3)))  # dummy use of to_sensor
    south = fr["xyz"][(fr["xyz"][:, 0] > s.sensor_pos[1] - 0.1) & (np.abs(fr["xyz"][:, 2]) < 0.1)
                      & (np.abs(fr["xyz"][:, 1]) < 0.3)]
    assert 0.01 < south[:, 0].std() < 0.05


def test_fog_attenuation_and_monotonic_soft_fraction():
    s = Scene.load(V0)
    d = s.dirs_to_world(scan_dirs(SensorParams(), 0, 4000)[0])
    r0, refl, cosi, _ = s.raycast(d)
    p = refl * cosi / r0 ** 2
    fr = []
    for a in (0.05, 0.3, 1.0, 3.0):
        f = fogmod.apply_fog(fogmod.Fog(alpha=a), s.sensor_pos, d, r0, p, 6.25e-5,
                             np.random.default_rng(0))
        np.testing.assert_allclose(f["p_hard"], p * np.exp(-2 * a * r0), rtol=0.05)   # Beer-Lambert
        assert np.all(f["range"][f["is_fog"]] < r0[f["is_fog"]] + 0.05)              # fog is in front
        fr.append(f["is_fog"].mean())
    assert fr == sorted(fr) and fr[0] < 0.05 and fr[-1] > 0.9
    assert fogmod.mor_to_alpha(50.0) == pytest.approx(math.log(20) / 50)


def test_fog_points_never_exceed_emitted_and_get_tagged():
    s = Scene.load()
    fr = simulate_frame(s, 0, np.random.default_rng(1), fog=fogmod.Fog(alpha=1.0), debug=True)
    assert fr["xyz"].shape[0] <= 20000 and fr["n_fog"] > 5000
    particle = (fr["tag"] >> 2) & 3
    assert particle[fr["is_fog"]].astype(bool).mean() == pytest.approx(0.6, abs=0.05)
    assert particle[~fr["is_fog"]].astype(bool).mean() < 0.02


def test_plume_is_local():
    f = fogmod.Fog(alpha=0.0, plume_alpha=3.0)
    src = np.array(f.plume_center)
    assert f.alpha_at(src[None] + [0, 0, 0.3])[0] > 1.0
    assert f.alpha_at(src[None] + [2.0, 0, 0.3])[0] < 0.01


def _plane(n, rng, noise=0.0):
    p = np.zeros((n, 3))
    p[:, :2] = rng.uniform(-1, 1, (n, 2))
    p[:, 2] = rng.normal(0, noise, n) if noise else 0.0
    return p


def test_entropy_limits_line_plane_volume():
    rng = np.random.default_rng(0)
    hp = entropy.HLParams(roi=(-5, 5, -5, 5, -5, 5), min_voxels=10)
    line = np.zeros((4000, 3)); line[:, 0] = np.linspace(-4.5, 4.5, 4000)
    plane = _plane(60000, rng)
    vol = rng.uniform(-0.6, 0.6, (300000, 3))
    rp, rv = (entropy.compute_hl(x, params=hp) for x in (plane, vol))
    rl = entropy.compute_hl(line, params=entropy.HLParams(roi=(-5, 5, -5, 5, -5, 5), min_voxels=10, leaf=0.02))
    assert rl["H_eigen"] < 0.05 and rl["H_DS"] < 0.05
    assert rp["H_eigen"] == pytest.approx(math.log(2), abs=0.06)
    assert rv["H_eigen"] > 1.0 and rv["mean_S"] > 0.5
    # the docstring warning: H_DS does not rank plane < volume the way H_eigen does
    assert rv["H_eigen"] - rp["H_eigen"] > 0.3
    assert rl["H_DS"] < 1e-3
    lam = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 2 / 3, 1 / 3]], float)
    h_ds, h_e, *_ = entropy.shape_entropies(lam)
    np.testing.assert_allclose(h_ds, [0, 0, 0, math.log(3)], atol=1e-9)
    np.testing.assert_allclose(h_e[:3], [0, math.log(2), math.log(3)], atol=1e-9)


def test_voxel_downsample_and_match():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (50000, 3))
    v = entropy.voxel_downsample(p, 0.25)
    assert v.shape[0] == 64
    r = entropy.compute_hl(_plane(60000, rng, 0.01), params=entropy.HLParams(roi=(-5, 5, -5, 5, -5, 5)),
                           match_nvoxel=800)
    assert r["n_voxel"] == 800 and r["n_voxel_full"] > 800


def test_bank_roundtrip(tmp_path):
    man = bank.write_bank(str(tmp_path / "b"), "B2", 10, seed=3)
    assert man["n_frames"] == 10 and man["fog"]["alpha"] == 0.30
    j = json.load(open(tmp_path / "b" / "manifest.json"))
    assert len(j["frames"]) == 10 and j["frames"][0]["n_people"] == 1
    fr = bank.load_frames(str(tmp_path / "b"))
    w = bank.windows(fr)
    assert len(w) == 1 and w[0]["xyz"].shape[0] == sum(f["xyz"].shape[0] for f in fr)
    r = entropy.compute_hl(w[0]["xyz"], w[0]["tag"])
    assert r["valid"] and 0.5 < r["H_eigen"] < math.log(3)
    # determinism
    man2 = bank.write_bank(str(tmp_path / "c"), "B2", 2, seed=3)
    a = np.load(tmp_path / "b" / "frame_00000.npz")["xyz"]
    b = np.load(tmp_path / "c" / "frame_00000.npz")["xyz"]
    np.testing.assert_array_equal(a, b)


def test_conditions_people_counts():
    base = Scene.load()
    assert bank.build_scene(base, "A0")[1] == 0
    assert bank.build_scene(base, "A1")[1] == 1
    assert bank.build_scene(base, "A2")[1] == 4
    s3, n3 = bank.build_scene(base, "A3")
    assert n3 == 4 and len(s3.boxes) > len(base.boxes)
    assert bank.build_scene(base, "B3")[1] == 1
