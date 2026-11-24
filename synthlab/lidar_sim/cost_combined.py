"""Combine the entropy-only and clustering-only Jetson campaigns into one cost table and figure.

  python -m synthlab.lidar_sim.cost_combined ent_unp clu_unp

Inputs: results/lidar-sim/cost_runs_<campaign>.csv (written by cost_analysis.py).
Outputs: results/lidar-sim/cost_combined.json, cost_combined.md, fig_cost_summary.png.
Per bank, runs are averaged over passes. H_eigen / n_voxel come from the entropy campaign (same banks).
"""
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cost_analysis import pcorr
from .figures import C, OUT, save

COST = ["total_ms", "power_w", "idle_power_w", "energy_per_window_j", "energy_above_idle_per_window_j"]


def per_bank(campaign):
    g = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(OUT, "cost_runs_%s.csv" % campaign))):
        g[r["bank"]].append(r)
    out = {}
    for b, rr in g.items():
        d = {k: float(np.mean([float(r[k]) for r in rr])) for k in COST + ["H_eigen", "n_voxel", "n_roi",
                                                                            "n_person_candidates", "n_clusters"]
             if rr[0].get(k) not in (None, "", "nan")}
        d.update(n_runs=len(rr), group=rr[0]["group"], fog_alpha=float(rr[0]["fog_alpha"]))
        out[b] = d
    return out


def main():
    ent, clu = per_bank(sys.argv[1]), (per_bank(sys.argv[2]) if len(sys.argv) > 2 else {})
    banks = sorted(ent)
    res = {"campaigns": sys.argv[1:], "note": "unpaced: windows back to back, so power = loaded power; "
           "energy per window = time-weighted VDD_IN x duration / windows (module input, not wall)"}
    for name, src in (("entropy", ent), ("cluster", clu)):
        bb = [b for b in banks if b in src]
        if len(bb) < 6:
            continue
        nv = np.array([ent[b]["n_voxel"] for b in bb])
        he = np.array([ent[b]["H_eigen"] for b in bb])
        d = {"n_banks": len(bb), "runs_per_bank": sorted({src[b]["n_runs"] for b in bb})}
        for y in ("total_ms", "energy_per_window_j", "energy_above_idle_per_window_j", "power_w"):
            yy = np.array([src[b][y] for b in bb])
            d[y] = dict(r_n_voxel=float(np.corrcoef(yy, nv)[0, 1]), r_H_eigen=float(np.corrcoef(yy, he)[0, 1]),
                        partial_r_H_eigen_given_n_voxel=pcorr(yy, he, nv),
                        partial_r_n_voxel_given_H_eigen=pcorr(yy, nv, he),
                        slope_per_1000_voxels=float(np.polyfit(nv, yy, 1)[0] * 1000),
                        clear_A1=src["A1"][y] if "A1" in src else None)
        # matched point count: thinned dense-fog vs thinned/unthinned clear with similar voxel count
        pairs = [("thin_clear_012", "thin_fog1p00_012"), ("thin_clear_025", "thin_fog1p00_025"),
                 ("A1", "thin_fog1p00_050")]
        d["matched_count"] = [dict(clear=a, fog=b, n_voxel=(ent[a]["n_voxel"], ent[b]["n_voxel"]),
                                   H_eigen=(ent[a]["H_eigen"], ent[b]["H_eigen"]),
                                   ms=(src[a]["total_ms"], src[b]["total_ms"]),
                                   ms_per_1000_vox=(1000 * src[a]["total_ms"] / ent[a]["n_voxel"],
                                                    1000 * src[b]["total_ms"] / ent[b]["n_voxel"]))
                              for a, b in pairs if a in src and b in src]
        lev = {}
        for lab, b in (("B0 (=A1)", "A1"), ("B1", "B1"), ("B2", "B2"), ("B3", "B3"), ("alpha 0.45", "fog_a0p45"),
                       ("alpha 3.0", "fog_a3p00"), ("P3 plume", "P3"), ("A3 clutter", "A3")):
            if b in src:
                lev[lab] = {k: src[b].get(k) for k in COST + ["n_person_candidates"]}
                lev[lab].update(n_voxel=ent[b]["n_voxel"], H_eigen=ent[b]["H_eigen"])
        d["levels"] = lev
        res[name] = d
    with open(os.path.join(OUT, "cost_combined.json"), "w") as f:
        json.dump(res, f, indent=1)

    lines = ["| condition | n_voxel | H_eigen | entropy ms | entropy J/window | cluster ms | cluster J/window | "
             "VDD_IN loaded W (ent / clu) | person candidates |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, e in res["entropy"]["levels"].items():
        c = res.get("cluster", {}).get("levels", {}).get(lab, {})
        f = lambda v, fm: (fm % v) if isinstance(v, float) else "-"
        lines.append("| %s | %.0f | %.3f | %.0f | %.2f | %s | %s | %.2f / %s | %s |" % (
            lab, e["n_voxel"], e["H_eigen"], e["total_ms"], e["energy_per_window_j"], f(c.get("total_ms"), "%.1f"),
            f(c.get("energy_per_window_j"), "%.3f"), e["power_w"], f(c.get("power_w"), "%.2f"),
            f(c.get("n_person_candidates"), "%.2f")))
    with open(os.path.join(OUT, "cost_combined.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: v for k, v in res.items() if k in ("entropy", "cluster")}, indent=1)[:3000])

    groups = ["clear scenes (A0-A3)", "uniform fog", "plume", "thinned control"]
    col = dict(zip(groups, [C["blue"], C["orange"], C["aqua"], C["violet"]]))
    rows = [("entropy", ent, "HL entropy"), ("cluster", clu, "clustering / person candidates")]
    rows = [r for r in rows if r[1]]
    fig, axs = plt.subplots(len(rows), 3, figsize=(13, 3.6 * len(rows)), squeeze=False)
    for i, (nm, src, title) in enumerate(rows):
        for j, (x, xl) in enumerate([("n_voxel", "occupied 5 cm voxels"), ("H_eigen", "H_eigen"),
                                     ("fog_alpha", "extinction alpha [1/m] (0.01 = clear)")]):
            ax = axs[i, j]
            for g in groups:
                bb = [b for b in banks if b in src and ent[b]["group"] == g
                      and (x != "fog_alpha" or g == "uniform fog" or b == "A1")]
                if bb:
                    xs = [max(ent[b]["fog_alpha"], 0.01) if x == "fog_alpha" else ent[b][x] for b in bb]
                    ax.scatter(xs, [src[b]["energy_per_window_j"] * 1000 for b in bb], c=col[g], s=26,
                               label=g, edgecolors=C["surface"], linewidths=0.6)
            if x == "fog_alpha":
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("%s [mJ / window]" % title, fontsize=8)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Jetson Orin Nano: energy per 1 s window (VDD_IN, unpaced) for simulated Mid-360 banks. "
                 "Cost follows voxel count, not entropy", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_cost_summary.png")


if __name__ == "__main__":
    main()
