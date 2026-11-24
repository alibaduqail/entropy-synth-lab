"""Tests for synthlab.thermal_physics (run: python -m pytest tests/test_thermal_physics.py)."""
import importlib.util
import os

import numpy as np
import pytest

from synthlab.thermal_physics import bie, measure, radiometry, raycast, render, scenarios, scene, sensor

HERE = os.path.dirname(__file__)
ROOM_V0 = os.path.join(HERE, "..", "scene", "room_v0.json")
ORIGINAL = "/Users/alibaduqail/Documents/thermal camera research work/scripts/experimentA.py"


# ---------------------------------------------------------------- radiometry
def test_planck_band_radiance_matches_reference():
    # 300 K black body, 8-14 um: ~ 55 W m^-2 sr^-1 (37.6 % of sigma T^4 / pi = 146.2)
    l300 = radiometry.band_radiance_exact(300.0)[0]
    assert 52.0 < l300 < 58.0
    total = 5.670374419e-8 * 300.0 ** 4 / np.pi
    assert 0.35 < l300 / total < 0.40


def test_lut_roundtrip_and_monotonic():
    lut = radiometry.BandLUT()
    t = np.linspace(250.0, 700.0, 500)
    l = lut.radiance(t)
    assert np.all(np.diff(l) > 0)
    assert np.max(np.abs(lut.temperature(l) - t)) < 0.01           # < 10 mK = 1 count
    assert np.max(np.abs(l - radiometry.band_radiance_exact(t)) / l) < 1e-4


def test_fog_limits():
    lut = radiometry.default_lut()
    l_hot, t_mist = lut.radiance(307.0), 295.0
    assert radiometry.apply_fog(l_hot, 1.0, t_mist) == pytest.approx(l_hot)
    assert radiometry.apply_fog(l_hot, 0.0, t_mist) == pytest.approx(lut.radiance(t_mist))
    mid = radiometry.apply_fog(l_hot, radiometry.fog_transmission(2.0, 0.3), t_mist)
    assert lut.radiance(t_mist) < mid < l_hot


# ---------------------------------------------------------------- BIE
def test_bie_matches_original_pipeline():
    if not os.path.exists(ORIGINAL):
        pytest.skip("parent project not on this machine")
    spec = importlib.util.spec_from_file_location("experimentA_orig", ORIGINAL)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    rng = np.random.default_rng(0)
    raw = (29500 + rng.normal(0, 150, (120, 160))).astype(np.uint16)
    raw[30:80, 60:100] += 900
    a = orig.block_image_entropy(orig.radiometric_bie_image(raw))
    b = bie.block_image_entropy(bie.radiometric_bie_image(raw))
    assert a == b


def test_bie_flat_is_zero_and_structure_raises_it():
    flat = np.full((120, 160), 29515, np.uint16)
    assert bie.ht_of_raw(flat) == 0.0
    img = flat.copy()
    img[40:90, 50:110] += 1000
    assert bie.ht_of_raw(img) > 0.01


# ---------------------------------------------------------------- ray caster
def _bk():
    return raycast.Backend("numpy", dtype="float64")


def test_room_hit_distances():
    bk = _bk()
    o = bk.arr([1.0, 1.0, 1.0])
    d = bk.arr([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    t, surf, n = raycast.trace_room(bk, o, d, bk.arr([3.0, 4.0, 2.5]))
    assert np.allclose(t, [2.0, 1.0, 1.5])
    assert list(surf) == [1, 2, 5]                       # east, south, ceiling
    assert np.allclose(n, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def test_primitive_intersections_analytic():
    bk = _bk()
    ps = scene.PrimSet()
    ps.add_box([0, 5, 0], [2, 2, 2], 30.0)
    ps.add_ellipsoid([10, 0, 0], [1, 2, 3], 30.0)
    ps.add_capsule([0, 0, 10], [0, 0, 12], 0.5, 30.0)
    prims = ps.to_backend(bk)
    o = bk.arr([0.0, 0.0, 0.0])
    d = bk.arr([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    t, idx, n = raycast.trace(bk, o, d, prims, None)
    assert np.allclose(t[:3], [4.0, 9.0, 9.5], atol=1e-6)
    assert list(idx[:3]) == [6, 7, 8] and idx[3] == -1
    assert np.allclose(n[:3], [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], atol=1e-6)


def test_yawed_box_and_per_ray_origins():
    bk = _bk()
    ps = scene.PrimSet()
    ps.add_box([0, 3, 0], [2, 2, 2], 30.0, yaw=np.pi / 4)        # corner toward origin
    prims = ps.to_backend(bk)
    o = bk.arr([[0.0, 0, 0], [0.0, 1.0, 0]])
    d = bk.arr([[0, 1.0, 0], [0, 1.0, 0]])
    t, idx, _ = raycast.trace(bk, o, d, prims, None)
    assert np.allclose(t, [3 - np.sqrt(2), 2 - np.sqrt(2)], atol=1e-6)


# ---------------------------------------------------------------- renderer
@pytest.fixture(scope="module")
def room():
    return scene.load_room(ROOM_V0)


def _temp_c(r, out):
    cam = sensor.Lepton35(supersample=r.ss, noise=False)
    return cam.ideal_counts(r.bk.to_numpy(out["radiance"])) / 100.0 - 273.15


def test_empty_room_is_near_wall_temperature(room):
    cam = scene.camera_from_room(room)
    r = render.Renderer(room, cam, supersample=1)
    r.set_static(render.build_static(room, include_objects=False, cam=cam))
    t = _temp_c(r, r.render(None))
    assert abs(np.median(t) - 22.5) < 0.6 and t.max() - t.min() < 2.0


def test_person_is_warm_labelled_and_boxed(room):
    cam = scene.camera_from_room(room)
    r = render.Renderer(room, cam, supersample=2)
    r.set_static(render.build_static(room, include_objects=False, cam=cam))
    ps = scene.PrimSet()
    scene.add_person(ps, (1.75, 0.5), yaw=0.0, pose="standing", inst=0, air_c=22.5)
    out = r.render(ps)
    t = _temp_c(r, out)
    assert 31.0 < t.max() < 35.5                         # face
    boxes = r.person_boxes(out)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]["xyxy"]
    assert 60 < 0.5 * (x0 + x1) < 100 and y1 - y0 > 60
    assert set(np.unique(out["label"])) >= {raycast.ROOM, raycast.SKIN, raycast.CLOTH}


def test_all_poses_build(room):
    for pose in scene.POSES:
        ps = scene.PrimSet()
        j = scene.add_person(ps, (1.0, 1.0), pose=pose, phase=0.3, lamp_in_hand=(pose == "arms_raised"))
        assert len(ps) >= 17 and j["head"][2] > 1.0
    with pytest.raises(ValueError):
        scene.add_person(scene.PrimSet(), (1, 1), pose="flying")


def test_fog_compresses_contrast_toward_mist_temperature(room):
    f0, _ = scenarios.generate_bank(dict(n_people=1, person_xy=[(1.75, 0.6)], seed=1), 1, room=room, noise=False)
    f1, _ = scenarios.generate_bank(dict(n_people=1, person_xy=[(1.75, 0.6)], seed=1, fog_beta_per_m=1.0,
                                         fog_turbulence=0.0), 1, room=room, noise=False)
    f2, _ = scenarios.generate_bank(dict(n_people=1, person_xy=[(1.75, 0.6)], seed=1, fog_beta_per_m=50.0,
                                         fog_turbulence=0.0, mist_offset_k=-1.0), 1, room=room, noise=False)
    assert f1.max() < f0.max() and f1.std() < f0.std()
    assert abs(f2.astype(float).mean() / 100 - 273.15 - 21.5) < 0.2 and f2.std() < 15     # opaque: mist temp


def test_lamp_warms_static_surfaces(room):
    off, _ = scenarios.generate_bank(dict(seed=0), 1, room=room, noise=False)
    on, _ = scenarios.generate_bank(dict(seed=0, lamp_on=True, lamp_position=[1.75, 1.0, 1.0]), 1, room=room, noise=False)
    assert on.max() > off.max() + 300 and on.astype(float).mean() > off.astype(float).mean()


def test_determinism_and_format(room):
    a, ma = scenarios.generate_bank(dict(scenarios.CONDITIONS["A2_three_people"], seed=5), 3, room=room)
    b, _ = scenarios.generate_bank(dict(scenarios.CONDITIONS["A2_three_people"], seed=5), 3, room=room)
    assert a.dtype == np.uint16 and a.shape == (3, 120, 160) and np.array_equal(a, b)
    assert len(ma["frames"]) == 3 and "boxes" in ma["frames"][0]
    assert 29000 < np.median(a) < 30000                  # ~ 17-27 C


# ---------------------------------------------------------------- sensor
def test_sensor_noise_statistics_match_calibration_targets():
    flat = np.full((120, 160), 29460.0)
    x, t = sensor.simulate_static(flat, 400, sensor.SensorParams(seed=2), first_ffc_s=1e9)
    tn = measure.temporal_noise(x, t)
    assert 1.9 < tn["total_sd"] < 3.1                    # real: 2.3-2.6 counts
    assert 0.8 < tn["ar1_rho"] < 0.95                    # real: 0.89
    assert 0.45 < (np.diff(x.astype(int), axis=0) == 0).mean() < 0.8     # real: 0.63
    fpn, _, _ = measure.spatial_fpn(x.astype(float), t)
    assert 4.0 < fpn["col_sd"] < 11.0 and 80 < fpn["lowfreq_p2p"] < 160  # real: 7.2, 119


def test_sensor_ffc_event_and_transient():
    flat = np.full((120, 160), 29460.0)
    x, t = sensor.simulate_static(flat, 500, sensor.SensorParams(seed=3), first_ffc_s=10.0)
    ffc = measure.find_ffc(t)
    assert len(ffc) == 1
    tr = measure.ffc_transient(t, x.astype(float).mean((1, 2)), ffc[0])
    assert 1.7 < tr["gap_s"] < 2.0 and 90 < tr["peak"] < 260 and 3 < tr["t_peak_s"] < 10


def test_noise_free_sensor_reports_scene_temperature():
    cam = sensor.Lepton35(noise=False, supersample=1)
    lut = radiometry.default_lut()
    rad = np.full((120, 160), lut.radiance(np.array([300.0]))[0])
    f, _ = cam.capture(rad)
    assert np.all(np.abs(f.astype(int) - 30000) <= 1)


def test_psf_blurs_an_edge():
    lut = radiometry.default_lut()
    rad = np.full((240, 320), lut.radiance(np.array([295.0]))[0])
    rad[:, 161:] = lut.radiance(np.array([305.0]))[0]
    sharp = sensor.Lepton35(sensor.SensorParams(psf_sigma_px=0.0), noise=False).ideal_counts(rad)
    soft = sensor.Lepton35(sensor.SensorParams(psf_sigma_px=1.0), noise=False).ideal_counts(rad)
    assert np.abs(np.diff(soft[60])).max() < np.abs(np.diff(sharp[60])).max()
    e = measure.edge_psf_sigma(soft)
    assert e is not None and 0.7 < e["sigma_px_median"] < 1.4
