"""Quantify what fog / ultrasonic mist does to a 60 GHz radar over 1-4 m.

Writes results/radar-sim/fog_summary.json (all numbers quoted in docs/radar-sim.md) and
fog_curves.json (arrays for the figures). Analytic part: ITU-R P.840 attenuation, Rayleigh
backscatter, refractivity, radome water film. Simulated part: the same still person in
B0-B3, comparing person-bin power, phase and detections.

Usage: python3 -m synthlab.radar_sim.fog_study --out results/radar-sim
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import conditions as cond
from . import config, fog, metrics, process, simulate
from . import scene as sc


def analytic(cfg: config.RadarConfig) -> dict:
    f0 = cfg.f_center_hz / 1e9
    out = {"f_center_ghz": f0}
    out["kl"] = {"%g_GHz_%gC" % (f, t): float(fog.kl_db_per_km_per_gm3(f, t))
                 for f in (60.0, 62.0, 64.0, 77.0) for t in (0.0, 20.0, 22.5)}
    lwcs = {"natural_dense_fog_0.5": 0.5, "B1_0.1": 0.1, "B2_0.5": 0.5, "B3_2.0": 2.0,
            "plume_near_mist_maker_20": 20.0}
    out["liquid_two_way_db"] = {name: {"%gm" % r: float(fog.fog_attenuation_db(f0, m, r))
                                       for r in (1.0, 2.0, 3.0, 4.0)} for name, m in lwcs.items()}
    out["o2_two_way_db"] = {"%gm" % r: fog.O2_DB_PER_KM_60GHZ * 2 * r / 1000 for r in (1.0, 2.0, 4.0)}
    # mist maker budget: how much water can an ultrasonic mist maker hang in the room?
    out["mist_maker_budget"] = {
        "assumed_output_ml_per_h": 350.0, "assumed_room_volume_m3": 18.0,
        "lwc_after_10min_if_nothing_evaporates_or_settles_gm3": 350.0 / 6.0 / 18.0,
        "note": "upper bound; in practice most of it evaporates until the air saturates"}
    out["saturation_vapour_density_gm3_22.5C"] = fog.vapour_density_gm3(22.5, 1.0)
    out["vapour_density_gm3_45pct"] = fog.vapour_density_gm3(22.5, 0.45)
    out["levels"] = {k: v.summary(f0) for k, v in fog.FOG_LEVELS.items()}
    # backscatter vs noise floor and torso
    lb = simulate.link_budget(cfg, 0.5, 1.33)
    out["link_budget_torso_0.5m2_at_1.33m"] = lb
    out["link_budget_torso_0.5m2_at_4m"] = simulate.link_budget(cfg, 0.5, 4.0)
    cell = {}
    for r in (1.0, 2.0, 4.0):
        vol = cfg.range_res_m * (r * np.radians(15.0)) * (r * np.radians(30.0))
        ner = simulate.link_budget(cfg, 1.0, r)["noise_equivalent_rcs_dbsm"]
        cell["%gm" % r] = {"resolution_cell_m3": vol, "noise_equivalent_rcs_dbsm": ner,
                           "fog_rcs_dbsm": {}}
        for name, m in lwcs.items():
            for d_um in (4.0, 20.0):
                eta = float(fog.volume_reflectivity(f0, m, d_um * 1e-6))
                cell["%gm" % r]["fog_rcs_dbsm"]["%s_%gum" % (name, d_um)] = \
                    float(10 * np.log10(eta * vol))
    out["fog_backscatter_per_cell"] = cell
    out["rayleigh_k2"] = float(fog.rayleigh_k2(f0))
    out["water_film"] = {}
    for um in (1, 5, 10, 25, 50, 100, 200):
        t = complex(fog.water_film_transmission(f0, um * 1e-6))
        out["water_film"]["%dum" % um] = {
            "two_way_loss_db": float(-40 * np.log10(abs(t))),
            "apparent_range_shift_mm": float(-np.angle(t) * cfg.wavelength_m / (2 * np.pi) * 1e3)}
    return out


def curves(cfg: config.RadarConfig) -> dict:
    f0 = cfg.f_center_hz / 1e9
    f = np.linspace(1, 100, 199)
    lwc = np.logspace(-2, np.log10(30), 60)
    th = np.logspace(0, np.log10(300), 60)
    t = np.array([complex(fog.water_film_transmission(f0, x * 1e-6)) for x in th])
    return {
        "f_ghz": f.tolist(),
        "kl": {"%gC" % tc: fog.kl_db_per_km_per_gm3(f, tc).tolist() for tc in (0.0, 10.0, 22.5)},
        "lwc": lwc.tolist(),
        "two_way_db": {"%gm" % r: fog.fog_attenuation_db(f0, lwc, r).tolist() for r in (1, 2, 4)},
        "o2_two_way_db": {"%gm" % r: fog.O2_DB_PER_KM_60GHZ * 2 * r / 1000 for r in (1, 2, 4)},
        "optical_trans_3m_4um": np.exp(-3 * fog.optical_extinction_per_m(lwc, 4e-6)).tolist(),
        "fog_rcs_dbsm_2m": {"%gum" % d: (10 * np.log10(fog.volume_reflectivity(f0, lwc, d * 1e-6)
                            * cfg.range_res_m * (2 * np.radians(15.0)) * (2 * np.radians(30.0)))).tolist()
                            for d in (4.0, 20.0)},
        "noise_equivalent_rcs_dbsm_2m": simulate.link_budget(cfg, 1.0, 2.0)["noise_equivalent_rcs_dbsm"],
        "film_um": th.tolist(), "film_two_way_db": (-40 * np.log10(np.abs(t))).tolist(),
        "film_shift_mm": (-np.angle(t) * cfg.wavelength_m / (2 * np.pi) * 1e3).tolist(),
        "levels": {k: v.lwc_gm3 for k, v in fog.FOG_LEVELS.items()},
    }


def simulated(cfg: config.RadarConfig, scene: dict, n_frames: int = 36, seed: int = 0) -> dict:
    """Same person, same seeds, B0..B3 (+ a wet-radome case): what changes in the output?"""
    proc0 = process.Processor(cfg, keep_maps=False)

    def empty_room(fs):
        bsim = simulate.RadarSimulator(cfg, scene, fog=fs, seed=999)
        return np.mean([proc0.process(bsim.frame(i)).zero_doppler for i in range(4)], axis=0)

    bg_clear = empty_room(fog.FogState())
    out, ref = {}, None
    levels = dict(fog.FOG_LEVELS)
    b3 = fog.FOG_LEVELS["B3"]
    levels["B3_wet_radome_50um"] = fog.FogState("B3_film", b3.lwc_gm3, rh=b3.rh, film_m=50e-6)
    for name, fs in levels.items():
        c = cond.build_condition("B0", scene, seed=seed)
        sim = simulate.RadarSimulator(cfg, scene, c["people"], fog=fs, seed=seed)
        proc = process.Processor(cfg, keep_maps=True, n_sub=8)
        bg = empty_room(fs)                 # empty-room background taken in the SAME medium
        rec = metrics.StillnessRecorder(cfg, bg, proc.az_flat)
        ents = []
        for i in range(n_frames):
            res = proc.process(sim.frame(i))
            rec.add(res)
            ents.append(metrics.frame_entropies(res, cfg))
        disp = rec.displacement_mm()
        k = disp["bin"]
        zd = np.array(rec.zd)
        sig = (zd[:, k, :] - bg[k, :]).mean(axis=0)
        wall = zd[:, :, :].mean(axis=0)
        e = {"person_bin_power_db": float(10 * np.log10((np.abs(sig) ** 2).sum())),
             "static_scene_power_db": float(10 * np.log10((np.abs(wall) ** 2).sum())),
             "person_bin_phase_rad": float(np.angle(sig[0])),
             "n_det_mean": float(np.mean(rec.n_det)),
             "raw_p2p_mm": float(disp["d_mm"].max() - disp["d_mm"].min())}
        e.update({kk: float(np.mean([x[kk] for x in ents])) for kk in ents[0]})
        # what a background recorded in clear, dry air leaves behind in the person's bin
        resid = bg[k, :] - bg_clear[k, :]
        e["clear_air_background_residual_db_rel_person"] = float(
            10 * np.log10(max((np.abs(resid) ** 2).sum(), 1e-300)) - e["person_bin_power_db"])
        if ref is None:
            ref = e
        e["delta_person_power_db_vs_B0"] = e["person_bin_power_db"] - ref["person_bin_power_db"]
        e["delta_static_power_db_vs_B0"] = e["static_scene_power_db"] - ref["static_scene_power_db"]
        dphi = np.angle(np.exp(1j * (e["person_bin_phase_rad"] - ref["person_bin_phase_rad"])))
        e["apparent_shift_mm_vs_B0"] = float(dphi * cfg.wavelength_m / (4 * np.pi) * 1e3)
        out[name] = e
    return {"n_frames": n_frames, "seed": seed, "scene": os.path.basename(scene["_path"]),
            "levels": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/radar-sim")
    ap.add_argument("--scene", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = config.preset()
    res = {"analytic": analytic(cfg), "simulated": simulated(cfg, sc.load_scene(a.scene)),
           "radar": cfg.summary(), "oob_3d_preset": config.preset("oob_3d").summary()}
    with open(os.path.join(a.out, "fog_summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(a.out, "fog_curves.json"), "w") as f:
        json.dump(curves(cfg), f)
    an = res["analytic"]
    print(json.dumps({k: an[k] for k in ("kl", "liquid_two_way_db", "o2_two_way_db", "water_film",
                                         "mist_maker_budget")}, indent=1))
    print(json.dumps(an["fog_backscatter_per_cell"]["2m"], indent=1))
    print(json.dumps(an["levels"], indent=1, default=str))
    print(json.dumps(res["simulated"], indent=1))


if __name__ == "__main__":
    main()
