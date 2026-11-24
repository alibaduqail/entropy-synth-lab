"""Tests for synthlab.radar_sim. Run with pytest, or directly: python3 tests/test_radar_sim.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthlab.radar_sim import conditions, config, fog, metrics, person, process, simulate  # noqa: E402
from synthlab.radar_sim import scene as sc  # noqa: E402

CFG = config.preset()
SCENE = sc.load_scene()
POSE = sc.radar_pose(SCENE)


def _point_sim(target_radar, rcs=1.0, exact=False, noise=False, cfg=CFG, seed=0):
    sim = simulate.RadarSimulator(cfg, SCENE, include_static=False, exact=exact, noise=noise,
                                  seed=seed)
    room = POSE.to_room(np.atleast_2d(target_radar))
    sim.static = sc.Scatterers(room, np.full(len(room), rcs), ["pt"] * len(room))
    sim.static_radar = POSE.to_radar(room)
    sim.static_fac = np.ones(len(room))
    return sim


def test_derived_parameters():
    assert abs(CFG.range_res_m - 0.0842) < 1e-3
    assert abs(CFG.range_max_m - 8.08) < 0.02
    assert abs(CFG.bandwidth_hz - 1.78e9) < 2e7
    assert 4.3 < CFG.v_max_mps < 4.7
    va = config.virtual_array_index(CFG)
    assert sorted(va["az"][va["el"] == 0].tolist()) == list(range(8))
    assert sorted(va["az"][va["el"] == 1].tolist()) == [2, 3, 4, 5]


def test_scene_loads_and_pose():
    assert os.path.basename(sc.default_scene_path()).startswith("room_v")
    # a point 1 m in front of the radar lies 1 m along the boresight in the room
    p = POSE.to_room(np.array([0.0, 1.0, 0.0]))
    b = np.array(SCENE["sensors"]["iwr6843isk"]["boresight"], dtype=float)
    assert np.allclose(p - POSE.position, b / np.linalg.norm(b), atol=1e-9)
    v0 = sc.load_scene(os.path.join(os.path.dirname(sc.default_scene_path()), "room_v0.json"))
    assert len(sc.static_scatterers(v0, CFG.wavelength_m)) > 50
    assert np.allclose(POSE.to_radar(p), [0, 1, 0])
    st = sc.static_scatterers(SCENE, CFG.wavelength_m)
    assert len(st) > 50 and np.all(st.rcs > 0)


def test_fast_model_matches_exact():
    tgt = np.array([[0.5, 2.0, 0.2], [-0.8, 3.0, -0.1]])
    a = _point_sim(tgt, exact=False).frame(0)
    b = _point_sim(tgt, exact=True).frame(0)
    assert np.abs(a - b).max() / np.abs(b).max() < 0.05


def test_point_target_localised():
    tgt = np.array([0.5, 2.0, 0.2])
    sim = _point_sim(tgt, noise=True)
    proc = process.Processor(CFG)
    cube = sim.frame(0)
    rf = np.fft.fft(cube * proc.win_r, axis=-1)
    k = int(np.argmax(np.abs(rf).sum(axis=(0, 1, 2))))
    assert abs(CFG.range_axis()[k] - np.linalg.norm(tgt)) < CFG.range_res_m
    # a static target is removed by clutter removal -> no detections
    assert proc.process(cube).n_det == 0


def test_moving_target_velocity_and_angle():
    # one scatterer receding at 1 m/s along its line of sight, via a Person-free path
    cfg = CFG
    sim = simulate.RadarSimulator(cfg, SCENE, include_static=False, noise=True, seed=1)
    p0 = np.array([0.6, 2.0, 0.15])
    u = p0 / np.linalg.norm(p0)
    t = sim.chirp_times(0)
    pos = p0[None, None, :] + (1.0 * t)[:, None, None] * u[None, None, :]
    slots = np.arange(len(t)) % cfg.n_tx
    cube = sim._synth(pos, np.array([0.5]), slots, 0.0, 1.0)
    rng = np.random.default_rng(5)
    cube = cube + np.sqrt(cfg.noise_power_w / 2) * (rng.standard_normal(cube.shape)
                                                    + 1j * rng.standard_normal(cube.shape))
    cube = cube.reshape(cfg.n_loops, cfg.n_tx, cfg.n_rx, cfg.n_samples).astype(np.complex64)
    res = process.Processor(cfg).process(cube)
    assert res.n_det >= 1
    best = res.points[np.argmax(res.points[:, 4])]
    assert abs(best[3] - 1.0) < 2 * cfg.v_res_mps          # + = receding
    assert abs(best[0] - p0[0]) < 0.2                      # azimuth (x) within 20 cm at 2 m
    assert abs(best[1] - p0[1]) < 0.2
    assert abs(best[2] - p0[2]) < 0.35                     # elevation is coarse (2 rows)


def test_noise_power_matches_link_budget():
    sim = simulate.RadarSimulator(CFG, SCENE, include_static=False, noise=True, seed=3)
    cube = sim.frame(0)
    assert abs(np.mean(np.abs(cube) ** 2) / CFG.noise_power_w - 1.0) < 0.05
    lb = simulate.link_budget(CFG, 0.5, 1.44)
    assert 30 < lb["snr_per_sample_db"] < 40


def test_empty_room_has_no_detections_and_person_does():
    c0 = conditions.build_condition("A0", SCENE)
    c1 = conditions.build_condition("M1", SCENE)
    n = []
    for c in (c0, c1):
        sim = simulate.RadarSimulator(CFG, SCENE, c["people"], c["extra_objects"], c["fog"], seed=0)
        proc = process.Processor(CFG)
        n.append(np.mean([proc.process(sim.frame(i)).n_det for i in range(4)]))
    assert n[0] == 0 and n[1] > 3


def test_cfar_cost_structure_and_dbscan():
    rng = np.random.default_rng(0)
    x = rng.exponential(size=(64, 64))
    x[20, 30] = 500.0
    m, _ = process.ca_cfar(x, 1, 2, 8, 15.0, False)
    assert m[20, 30] and m.sum() <= 3
    pts = np.concatenate([rng.normal(0, 0.05, (10, 3)), rng.normal(3, 0.05, (8, 3)), [[10, 10, 10]]])
    lab = process.dbscan(pts, 0.3, 3)
    assert lab.max() == 1 and lab[-1] == -1


def test_fog_attenuation_negligible_at_60ghz():
    kl = float(fog.kl_db_per_km_per_gm3(60.0, 20.0))
    assert 1.5 < kl < 2.0                                   # ITU-R P.840: ~1.75 at 20 C
    a = float(fog.fog_attenuation_db(62.0, 2.0, 4.0))       # heavy mist, 4 m target, two-way
    assert a < 0.05
    assert fog.FOG_LEVELS["B3"].delta_refractivity_ppm() > 30
    t = abs(fog.water_film_transmission(62.0, 100e-6)) ** 2
    assert 0.05 < t < 0.9                                   # a wet radome is NOT negligible


def test_breathing_displacement_recovered():
    p = person.Person(SCENE["dynamic_slots"]["person_mark"][:2], face_to=POSE.position,
                      sway_rms_m=0.0, seed=4)
    bg_sim = simulate.RadarSimulator(CFG, SCENE, seed=0)
    proc = process.Processor(CFG, keep_maps=False)
    bg = proc.process(bg_sim.frame(0)).zero_doppler
    sim = simulate.RadarSimulator(CFG, SCENE, [p], seed=0, shadowing=False)
    rec = metrics.StillnessRecorder(CFG, bg)
    for i in range(int(10 / CFG.frame_period_s)):
        rec.add(proc.process(sim.frame(i)))
    m = rec.metrics()
    # chest moves 4 mm p-p; static body parts in the same range bin bias the scale
    assert 1.5 < m["raw_p2p_mm"] < 9.0
    assert m["excursion_mm"] < 0.5 * m["raw_p2p_mm"]


def test_entropy_helpers():
    assert abs(metrics.shannon_bits(np.ones(16)) - 4.0) < 1e-9
    assert metrics.shannon_bits(np.array([1.0, 0, 0])) == 0.0
    assert metrics.point_entropy(np.zeros((0, 6)), CFG) == 0.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if fails else 0)
