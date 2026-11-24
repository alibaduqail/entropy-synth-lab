"""Aggregate the Jetson benchmark (summary.json + frames.csv) into results/radar-sim/jetson_bench/cost_table.json.

Usage: python3 -m synthlab.radar_sim.bench_table summary.json frames.csv out.json
"""
import csv, json, sys
from collections import defaultdict
import numpy as np

STAGES = ["range_fft", "clutter_removal", "doppler_fft", "detection_matrix", "cfar", "angle", "cluster", "track"]


def main(summ, frames, out):
    s = json.load(open(summ))
    rows = defaultdict(list)
    for r in s["summary"]:
        rows[r["condition"]].append(r)
    res = {"conditions": {}, "n_repeats": s["args"]["repeats"], "host": s["host"]}
    for c, rs in sorted(rows.items()):
        d = {"description": rs[0]["description"]}
        m = lambda f: float(np.mean([f(r) for r in rs]))
        d["idle_mw"] = m(lambda r: r["idle"]["vdd_in_mw_mean"])
        d["free_mw"] = m(lambda r: r["free"]["vdd_in_mw_mean"])
        d["paced_mw"] = m(lambda r: r["paced"]["vdd_in_mw_mean"])
        d["gen_ms_per_frame"] = m(lambda r: r["gen"]["ms_per_frame"])
        for k in ["fixed", "data_dependent", "total", "n_det"] + STAGES:
            d["free_%s_mean" % k] = m(lambda r: r["free"][k]["mean"])
        d["paced_fps"] = m(lambda r: r["paced"]["fps"])
        for k in rs[0]["entropy"]:
            d[k] = m(lambda r: r["entropy"][k])
        res["conditions"][c] = d
    ws = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(frames)):
        if r["mode"] != "free":
            continue
        for k in STAGES + ["fixed", "data_dependent", "total", "n_det"]:
            ws[r["condition"]][k].append(float(r[k]))
    allf = {k: np.concatenate([ws[c][k] for c in ws]) for k in ["n_det", "data_dependent", "total", "fixed"] + STAGES}
    nd = allf["n_det"]
    fit = {}
    for k in ["data_dependent", "cfar", "angle", "cluster", "track", "fixed", "range_fft", "doppler_fft", "total"]:
        p = np.polyfit(nd, allf[k], 1)
        fit[k] = {"slope_ms_per_detection": float(p[0]), "intercept_ms": float(p[1]),
                  "pearson_r_with_n_det": float(np.corrcoef(nd, allf[k])[0, 1])}
    res["pooled_free_frames"] = {"n_frames": int(len(nd)), "n_det_min": float(nd.min()), "n_det_max": float(nd.max()),
                                 "fit_vs_n_det": fit}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
