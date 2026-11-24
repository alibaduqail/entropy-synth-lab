"""Tables for docs/lidar-sim.md, computed only from files in results/lidar-sim/.

  python -m synthlab.lidar_sim.summarize      -> results/lidar-sim/summary_tables.md + summary.json
"""
import json
import os
from collections import defaultdict

import numpy as np

from .figures import OUT, read

PRE = ["H_eigen", "H_DS", "mean_S", "frac_high_S", "frac_tag_particle", "n_voxel"]
METRICS = ["H_eigen", "H_DS", "mean_S", "frac_high_S", "n_voxel", "n_raw", "frac_tag_particle"]


def table(rows, key, metrics, order=None):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    keys = order or sorted(g)
    out = {}
    for k in keys:
        if k in g:
            out[k] = {m: (float(np.mean([r[m] for r in g[k]])), float(np.std([r[m] for r in g[k]])))
                      for m in metrics}
            out[k]["n"] = len(g[k])
    return out


def md(tab, metrics, keyname, fmt=None):
    fmt = fmt or {}
    lines = ["| %s | n | %s |" % (keyname, " | ".join(metrics)), "|---|---|" + "---|" * len(metrics)]
    for k, v in tab.items():
        cells = []
        for m in metrics:
            f = fmt.get(m, "%.3f")
            cells.append((f + " +/- " + f) % v[m])
        lines.append("| %s | %d | %s |" % (k, v["n"], " | ".join(cells)))
    return "\n".join(lines)


def main():
    fmt = {"n_voxel": "%.0f", "n_raw": "%.0f", "H_eigen": "%.4f", "H_DS": "%.4f"}
    S, parts = {}, []
    cond = read("conditions.csv")
    order = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "P1", "P2", "P3"]
    S["conditions"] = table(cond, "condition", METRICS, order)
    parts += ["### Conditions (5 windows each, mean +/- SD across windows)", md(S["conditions"], METRICS, "cond", fmt)]

    sw = read("fog_sweep.csv")
    S["fog_sweep"] = {str(k): v for k, v in table(sw, "fog_alpha", METRICS).items()}
    parts += ["### Fog sweep, default model (3 windows each)", md(table(sw, "fog_alpha", METRICS), METRICS, "alpha [1/m]", fmt)]

    # which metric responds: effect size relative to clear-air window-to-window SD, and monotonicity
    sweeps = [("fog_sweep.csv", "strongest-echo rule", "metric_response")]
    if os.path.exists(os.path.join(OUT, "fog_sweep_first.csv")):
        sweeps.append(("fog_sweep_first.csv", "first-return rule (literal Mid-360 behaviour)", "metric_response_first"))
        swf = read("fog_sweep_first.csv")
        S["fog_sweep_first"] = {str(k): v for k, v in table(swf, "fog_alpha", METRICS).items()}
        parts += ["### Fog sweep, first-return rule (3 windows each)",
                  md(table(swf, "fog_alpha", METRICS), METRICS, "alpha [1/m]", fmt)]
    for fname, title, key in sweeps:
        t = table(read(fname), "fog_alpha", METRICS)
        alphas = sorted(t)
        resp = {}
        for m in METRICS:
            base, sd0 = t[0.0][m]
            vals = np.array([t[a][m][0] for a in alphas])
            rho = float(np.corrcoef(np.argsort(np.argsort(vals)), np.arange(len(alphas)))[0, 1])
            resp[m] = dict(clear=base, clear_sd=sd0, max_change_pct=float(100 * (vals.max() - base) / base),
                           min_change_pct=float(100 * (vals.min() - base) / base),
                           spearman_vs_alpha=rho, alpha_at_max=float(alphas[int(vals.argmax())]),
                           **{"d_" + lev: float(t[a][m][0] - base) for lev, a in (("B1", 0.1), ("B2", 0.3), ("B3", 1.0))},
                           **{"z_" + lev: float((t[a][m][0] - base) / max(sd0, 1e-9))
                              for lev, a in (("B1", 0.1), ("B2", 0.3), ("B3", 1.0))})
        S[key] = resp
        lines = ["| metric | clear | max change (at alpha) | min change | Spearman vs alpha | B1 shift / clear SD | B2 | B3 |",
                 "|---|---|---|---|---|---|---|---|"]
        for m, d in resp.items():
            lines.append("| %s | %.4g | %+.1f %% (%.2f) | %+.1f %% | %.2f | %.0f | %.0f | %.0f |" % (
                m, d["clear"], d["max_change_pct"], d["alpha_at_max"], d["min_change_pct"],
                d["spearman_vs_alpha"], d["z_B1"], d["z_B2"], d["z_B3"]))
        parts += ["### Which metric responds to fog (%s)" % title, "\n".join(lines)]

    if os.path.exists(os.path.join(OUT, "fog_variants.csv")):
        va = read("fog_variants.csv")
        names = []
        for r in va:
            if r["variant"] not in names:
                names.append(r["variant"])
        S["variants"], lines = {}, []
        pre = {}
        for nm in names:
            tt = table([r for r in va if r["variant"] == nm], "fog_alpha", METRICS)
            S["variants"][nm] = {str(k): v for k, v in tt.items()}
            al = sorted(tt)
            he = np.array([tt[a]["H_eigen"][0] for a in al])
            half = he[0] + 0.5 * (he.max() - he[0])
            knee = next((a for a, h in zip(al, he) if h >= half), float("nan"))
            onset = next((a for a, h in zip(al, he) if h >= he[0] + 0.02), float("nan"))
            nv = np.array([tt[a]["n_voxel"][0] for a in al])
            lines.append("| %s | %.3f | %.3f | %.2f | %.2f | %.2f | %.2f (at %.2f) | %.2f |" % (
                nm, he[0], he.max(), onset, knee, tt[1.0]["H_eigen"][0] - he[0],
                nv.max() / nv[0], al[int(nv.argmax())], nv[-1] / nv[0]))
            for lev, a in (("B1", 0.1), ("B2", 0.3), ("B3", 1.0)):
                for m in PRE:
                    v = tt[a][m][0] - tt[0.0][m][0] if m != "n_voxel" else tt[a][m][0] / tt[0.0][m][0]
                    pre.setdefault(lev, {}).setdefault(m, []).append(float(v))
        parts += ["### Fog sweep under model variants (2 windows each)",
                  "| variant | H_eigen clear | H_eigen max | onset alpha (+0.02) | half-rise alpha | dH_eigen at alpha 1 | n_voxel peak ratio | n_voxel ratio at alpha 3 |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(lines)]
        # headline = first-return 3-window sweep (literal Mid-360 rule) if present; range = all variants
        head_f = "fog_sweep_first.csv" if os.path.exists(os.path.join(OUT, "fog_sweep_first.csv")) else "fog_sweep.csv"
        th = table(read(head_f), "fog_alpha", METRICS)
        S["preregistration"] = {}
        for lev, a in (("B1", 0.1), ("B2", 0.3), ("B3", 1.0)):
            S["preregistration"][lev] = {}
            for m in PRE:
                h = th[a][m][0] / th[0.0][m][0] if m == "n_voxel" else th[a][m][0] - th[0.0][m][0]
                v = pre[lev][m] + [float(h)]
                S["preregistration"][lev][m] = dict(headline=float(h), min=min(v), max=max(v),
                                                    clear_sd=float(th[0.0][m][1]))
        S["preregistration_headline_source"] = head_f
        lines = ["| level (alpha, MOR) | " + " | ".join("d " + m if m != "n_voxel" else "n_voxel ratio" for m in PRE) + " |",
                 "|---|" + "---|" * len(PRE)]
        for (lev, d), lab in zip(S["preregistration"].items(), ("B1 (0.1 /m, 30 m)", "B2 (0.3 /m, 10 m)", "B3 (1.0 /m, 3 m)")):
            lines.append("| %s | %s |" % (lab, " | ".join(
                "%+.3f [%+.3f, %+.3f]" % (d[m]["headline"], d[m]["min"], d[m]["max"]) if m != "n_voxel"
                else "x%.2f [%.2f, %.2f]" % (d[m]["headline"], d[m]["min"], d[m]["max"]) for m in PRE)))
        lines.append("| clear-air window SD (1 s) | " + " | ".join(
            "%.4f" % S["preregistration"]["B1"][m]["clear_sd"] if m != "n_voxel"
            else "%.0f voxels" % S["preregistration"]["B1"][m]["clear_sd"] for m in PRE) + " |")
        parts += ["### Pre-registration table: predicted change vs clear (B0); headline = %s, [range over all model variants]" % head_f,
                  "\n".join(lines)]

    if os.path.exists(os.path.join(OUT, "hl_sensitivity.csv")):
        se = read("hl_sensitivity.csv")
        S["sensitivity"] = {}
        lines = ["| param | value | alpha 0: H_eigen / H_DS / n_voxel | alpha 0.3 | alpha 1.0 | dH_eigen(1.0 - 0) | dH_DS(1.0 - 0) |", "|---|---|---|---|---|---|---|"]
        for p in ("window_s", "leaf_m", "radius_m"):
            for v in sorted({r["value"] for r in se if r["param"] == p}):
                cells, he, hd = [], {}, {}
                for a in (0.0, 0.3, 1.0):
                    rr = [r for r in se if r["param"] == p and r["value"] == v and r["fog_alpha"] == a
                          and isinstance(r["H_eigen"], float)]
                    if rr:
                        he[a], hd[a] = np.mean([r["H_eigen"] for r in rr]), np.mean([r["H_DS"] for r in rr])
                        nv = np.mean([r["n_voxel"] for r in rr])
                        cells.append("%.3f / %.3f / %.0f" % (he[a], hd[a], nv))
                        S["sensitivity"]["%s=%s,alpha=%s" % (p, v, a)] = dict(H_eigen=float(he[a]), H_DS=float(hd[a]), n_voxel=float(nv))
                    else:
                        cells.append("too sparse")
                d1 = "%+.3f" % (he[1.0] - he[0.0]) if 1.0 in he and 0.0 in he else "-"
                d2 = "%+.3f" % (hd[1.0] - hd[0.0]) if 1.0 in hd and 0.0 in hd else "-"
                lines.append("| %s | %s | %s | %s | %s |" % (p, v, " | ".join(cells), d1, d2))
        parts += ["### HL sensitivity to method parameters (up to 4 windows each)", "\n".join(lines)]

    ref = json.load(open(os.path.join(OUT, "real_reference.json")))
    m0, a0, a3 = ref["lab_M0_empty"], S["conditions"]["A0"], S["conditions"]["A3"]
    lines = ["| metric | real lab M0 (empty) | sim A0 (empty) | sim A3 (people + clutter) | real M1 |", "|---|---|---|---|---|"]
    for m in ("H_DS", "H_eigen", "mean_S", "frac_high_S", "n_voxel", "n_raw", "frac_tag_particle"):
        lines.append("| %s | %.4g | %.4g | %.4g | %s |" % (m, m0[m], a0[m][0], a3[m][0],
                                                           ref["lab_M1_clutter_people"].get(m, "-")))
    parts += ["### Clear-air sanity check against real Mid-360 summaries (different room)", "\n".join(lines)]

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    with open(os.path.join(OUT, "summary_tables.md"), "w") as f:
        f.write("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
