#!/usr/bin/env python3
"""Loop a LiDAR frame bank through the LiDAR-entropy workload on the Jetson and record
per-window stage timings with tegrastats power alongside.

  bank dir : frame_00000.npz ... each one 100 ms Mid-360 frame with xyz float32 (M,3) in
             the sensor frame, optional intensity uint8, optional tag uint8;
             manifest.json (optional; copied into the output)
  workload : entropy = the HL computation (ROI -> 5 cm voxels -> radius kNN -> covariance
                       eigenvalues -> H_DS, H_eigen), as in the LiDAR track
             cluster = Euclidean clustering + person-candidate filter on the voxelised cloud
             both    = entropy then cluster (default)
  outputs  : <out>/windows.csv, tegrastats.csv, summary.json, bank_manifest.json

Frames are accumulated into windows of --window seconds (10 frames per 1 s window). Paced
mode starts one window every --window seconds, as the live system would; --unpaced runs
windows back to back. Timed runs must hold the shared lock:
  tools/jetson.sh measured 'python3 ~/synthlab/harness/run_lidar_bank.py --bank B --out O'
Excluded: sensor driver, UDP/ROS transport, rosbag decoding, disk reads (bank is in RAM).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lidar_pipeline as lp  # noqa: E402
import telemetry as tm  # noqa: E402

HARNESS = Path(__file__).resolve().parent
STAGES = lp.ENTROPY_STAGES + lp.CLUSTER_STAGES + ["entropy_total_ms", "cluster_total_ms", "total_ms"]
VALUES = ["H_DS", "H_eigen", "mean_L", "mean_P", "mean_S", "frac_high_S", "n_raw", "n_roi",
          "n_voxel", "n_valid", "mean_neighbors", "frac_capped", "frac_tag_particle",
          "n_pairs", "n_clusters", "n_person_candidates"]


def load_bank(folder, frames_per_window):
    folder = Path(folder).expanduser()
    files = sorted(folder.glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frame_*.npz in {folder}")
    frames = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            xyz = np.asarray(data["xyz"], dtype=np.float32)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise SystemExit(f"{path.name}: xyz must be (M,3), got {xyz.shape}")
            tag = (np.asarray(data["tag"], dtype=np.uint8) if "tag" in data.files
                   else np.zeros(xyz.shape[0], np.uint8))
            if tag.shape[0] != xyz.shape[0]:
                raise SystemExit(f"{path.name}: tag length differs from xyz")
            frames.append((np.ascontiguousarray(xyz), tag))
    n_windows = len(frames) // frames_per_window
    if n_windows < 1:
        raise SystemExit(f"bank has {len(frames)} frames; one window needs {frames_per_window}")
    windows = [frames[i * frames_per_window:(i + 1) * frames_per_window] for i in range(n_windows)]
    manifest = {}
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text())
    info = {"bank_name": folder.name, "bank_dir": str(folder), "bank_frames": len(frames),
            "bank_windows": n_windows, "frames_per_window": frames_per_window,
            "bank_frames_unused": len(frames) - n_windows * frames_per_window,
            "bank_points_per_frame": float(np.mean([f[0].shape[0] for f in frames])),
            "bank_track": folder.parent.parent.name if folder.parent.name == "banks" else None,
            "bank_generator": manifest.get("generator") if isinstance(manifest, dict) else None}
    return windows, manifest, info


def run_trial(windows, params, out, args, info):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    use_tegra = (not args.no_tegrastats) and tm.tegrastats_available()
    interval_s = args.tegra_interval_ms / 1000.0
    logger = tm.TegraLogger(out / "tegrastats.csv", args.tegra_interval_ms) if use_tegra else None
    period, n, rows = args.window, len(windows), []
    if logger:
        logger.start()
    try:
        idle_start = time.time()
        time.sleep(args.idle_seconds)
        idle_end = time.time()
        warm_start, index = time.monotonic(), 0
        while time.monotonic() - warm_start < args.warmup:
            if not args.unpaced:
                tm.wait_until(warm_start + index * period)
            lp.process_window(windows[index % n], params, args.workload)
            index += 1
        base_mono, base_unix = time.monotonic(), time.time()
        target = None if args.unpaced else max(1, int(round(args.seconds / period)))
        index = 0
        while True:
            if args.unpaced:
                if time.monotonic() - base_mono >= args.seconds:
                    break
                late = 0.0
            else:
                if index >= target:
                    break
                late = tm.wait_until(base_mono + index * period)
            row = {"window": index, "bank_window": index % n, "unix_time": time.time(),
                   "offset_s": time.monotonic() - base_mono, "lateness_ms": late * 1e3}
            row.update(lp.process_window(windows[index % n], params, args.workload))
            rows.append(row)
            index += 1
        if not args.unpaced:
            tm.wait_until(base_mono + target * period)
        end_mono, end_unix = time.monotonic(), time.time()
        time.sleep(max(0.5, 2 * interval_s))
    finally:
        if logger:
            logger.stop()
    with open(out / "windows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    duration = end_mono - base_mono
    col = lambda key: np.asarray([r[key] for r in rows], dtype=float)
    summary = {"status": "complete", **info, "workload": args.workload,
               "mode": "unpaced" if args.unpaced else "paced", "window_s": args.window,
               "seconds": args.seconds, "warmup_s": args.warmup,
               "windows": len(rows), "duration_s": duration,
               "windows_per_s": len(rows) / duration, "bank_loops": len(rows) / n,
               "late_windows": int((col("lateness_ms") > 50).sum()),
               "overrun_windows": int((col("total_ms") > period * 1e3).sum()),
               "duty_cycle": float(col("total_ms").sum() / 1e3 / duration),
               "values": {k: tm.dist(col(k)) for k in VALUES},
               "stages_ms": {k: tm.dist(col(k)) for k in STAGES},
               "params": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in vars(params).items()},
               "measurement_start_unix": base_unix, "measurement_end_unix": end_unix,
               "excluded": "sensor driver, transport, bag decoding, disk I/O; bank looped from RAM",
               "power_note": "tegrastats VDD_IN = module input power, not wall power"}
    problems = []
    if int(np.isnan(col("H_DS")).sum()) and args.workload != "cluster":
        problems.append(f"{int(np.isnan(col('H_DS')).sum())} windows too sparse for entropy")
    if logger:
        tele = tm.summarize(out / "tegrastats.csv", base_unix, end_unix, interval_s, len(rows))
        idle = tm.idle_power(out / "tegrastats.csv", idle_start, idle_end)
        summary["telemetry"] = tele
        problems += tele["problems"]
        if "energy_j" in tele:
            mean = lambda key: (tele["metrics"].get(key) or {}).get("mean")
            summary.update(power_w=tele["power_w_time_weighted"], energy_j=tele["energy_j"],
                           energy_per_window_j=tele["energy_per_unit_j"], idle_power_w=idle,
                           gpu_pct=mean("gpu_pct"), cpu_pct=mean("cpu_pct"), ram_mb=mean("ram_mb"),
                           temp_cpu_c=mean("temp_cpu_c"),
                           power_cpu_gpu_cv_w=mean("vdd_cpu_gpu_cv_w"))
            if idle is not None:
                summary["energy_above_idle_per_window_j"] = (
                    (tele["power_w_time_weighted"] - idle) * duration / len(rows))
    else:
        problems.append("tegrastats not available: no power data")
    summary["problems"] = problems
    summary["ok"] = not problems
    tm.write_json(out / "summary.json", summary)
    return summary


def one_line(s):
    v, st, power = s["values"], s["stages_ms"], s.get("power_w")
    get = lambda d, k: (d[k] or {}).get("mean", float("nan"))
    return (f"{s['bank_name']} [{s['workload']}/{s['mode']}] windows={s['windows']} "
            f"H_DS={get(v, 'H_DS'):.4f} H_eigen={get(v, 'H_eigen'):.4f} "
            f"n_roi={get(v, 'n_roi'):.0f} n_vox={get(v, 'n_voxel'):.0f} "
            f"knn={get(st, 'knn_ms'):.1f}ms eig={get(st, 'eig_ms'):.1f}ms "
            f"entropy_total={get(st, 'entropy_total_ms'):.1f}ms "
            f"cluster_total={get(st, 'cluster_total_ms'):.1f}ms total={get(st, 'total_ms'):.1f}ms "
            + (f"P={power:.3f}W E/window={s['energy_per_window_j']:.2f}J " if power is not None
               else "P=n/a ")
            + ("OK" if s["ok"] else "PROBLEMS: " + "; ".join(s["problems"])))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, type=Path, help="dir of frame_*.npz (+manifest.json)")
    ap.add_argument("--out", required=True, type=Path, help="new output dir")
    ap.add_argument("--window", type=float, default=1.0, help="accumulation window (s)")
    ap.add_argument("--frame-period", type=float, default=0.1, help="bank frame duration (s)")
    ap.add_argument("--workload", choices=["entropy", "cluster", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--unpaced", action="store_true", help="windows back to back")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--leaf", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=0.20)
    ap.add_argument("--k-max", type=int, default=150)
    ap.add_argument("--k-min", type=int, default=10)
    ap.add_argument("--min-voxels", type=int, default=200)
    ap.add_argument("--z-min", type=float, default=-0.85, help="floor cut in the sensor frame")
    ap.add_argument("--roi", type=float, nargs=6, default=[-6, 6, -6, 6, -2, 3],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    ap.add_argument("--cluster-tol", type=float, default=0.10)
    ap.add_argument("--cluster-min-pts", type=int, default=30)
    ap.add_argument("--idle-seconds", type=float, default=3)
    ap.add_argument("--tegra-interval-ms", type=int, default=200)
    ap.add_argument("--no-tegrastats", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seconds <= 0 or args.warmup < 0 or args.window <= 0 or args.repeats < 1:
        raise SystemExit("need seconds > 0, warmup >= 0, window > 0, repeats >= 1")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output dir {args.out} is not empty; choose a new one")
    windows, manifest, info = load_bank(args.bank, max(1, int(round(args.window / args.frame_period))))
    params = lp.Params(roi=tuple(args.roi), z_min=args.z_min, leaf=args.leaf, radius=args.radius,
                       k_max=args.k_max, k_min=args.k_min, min_voxels=args.min_voxels,
                       cluster_tol=args.cluster_tol, cluster_min_pts=args.cluster_min_pts)
    args.out.mkdir(parents=True, exist_ok=True)
    tm.write_json(args.out / "bank_manifest.json", manifest)
    tm.write_json(args.out / "run_config.json",
                  {"args": {k: str(v) for k, v in vars(args).items()}, **info,
                   "environment": tm.environment(),
                   "code_sha256": {p.name: tm.sha256(p) for p in sorted(HARNESS.glob("*.py"))}})
    summaries = []
    for repeat in range(args.repeats):
        folder = args.out if args.repeats == 1 else args.out / f"trial_{repeat:02d}"
        summary = run_trial(windows, params, folder, args, info)
        summary["repeat"] = repeat
        summaries.append(summary)
        print(one_line(summary), flush=True)
    if args.repeats > 1:
        table = {k: tm.dist([s[k] for s in summaries if s.get(k) is not None])
                 for k in ["power_w", "energy_per_window_j"]}
        for stage in STAGES:
            table[stage + "_mean"] = tm.dist([s["stages_ms"][stage]["mean"] for s in summaries])
        tm.write_json(args.out / "repeats.json",
                      {"n_trials": len(summaries), "trial_to_trial": table, **info})
    return 0 if all(s["ok"] for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
