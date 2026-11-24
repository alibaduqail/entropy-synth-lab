#!/usr/bin/env python3
"""Run one randomised round over many banks in a single process (model loaded once).

Each call measures every bank once, in an order shuffled by (seed + round), so repeated
rounds interleave banks and slow drifts (chip temperature) do not line up with any bank.
One round per locked command keeps the lock short:

  tools/jetson.sh measured 'python3 ~/synthlab/harness/run_matrix.py --kind thermal \
      --banks ~/synthlab/*/banks/* --round 0 --seed 2026 --out ~/synthlab/jetson-harness/runs/cmp \
      -- --seconds 20 --warmup 5 --unpaced'

Everything after `--` goes to run_thermal_bank.py / run_lidar_bank.py unchanged.
Outputs: <out>/round_RR/<order>_<track>__<bank>/{frames|windows}.csv, summary.json, ...
         <out>/round_RR/order.json
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry as tm  # noqa: E402


def bank_id(path):
    path = Path(path)
    track = path.parent.parent.name if path.parent.name == "banks" else path.parent.name
    return f"{track}__{path.name}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["thermal", "lidar"], required=True)
    ap.add_argument("--banks", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-minutes", type=float, default=18.0,
                    help="stop starting new trials after this long (lock etiquette)")
    args = ap.parse_args(argv)
    marker = "frames.npy" if args.kind == "thermal" else "frame_00000.npz"
    banks = sorted({p.expanduser().resolve() for p in args.banks
                    if (p.expanduser() / marker).exists()}, key=bank_id)
    if not banks:
        raise SystemExit("no valid banks given")
    round_dir = args.out.expanduser() / f"round_{args.round:02d}"
    if round_dir.exists() and any(round_dir.iterdir()):
        raise SystemExit(f"{round_dir} already has data")
    round_dir.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(args.seed + args.round).permutation(len(banks))
    tm.write_json(round_dir / "order.json",
                  {"seed": args.seed, "round": args.round, "extra_args": extra,
                   "order": [bank_id(banks[i]) for i in order], "environment": tm.environment()})
    if args.kind == "thermal":
        import run_thermal_bank as runner
    else:
        import run_lidar_bank as runner
    parser = runner.build_parser()
    detector, start, failures = None, time.monotonic(), 0
    for position, index in enumerate(order):
        bank = banks[index]
        name = f"{position:02d}_{bank_id(bank)}"
        if (time.monotonic() - start) / 60 > args.max_minutes:
            print(f"SKIPPED {name}: round exceeded {args.max_minutes} min", flush=True)
            failures += 1
            continue
        out = round_dir / name
        trial_args = parser.parse_args(["--bank", str(bank), "--out", str(out)] + extra)
        try:
            if args.kind == "thermal":
                data, manifest, info = runner.load_bank(bank)
                if detector is None:
                    detector = runner.make_detector(trial_args)
                for raw in data[:8]:
                    runner.process_frame(raw, detector)
                summary = runner.run_trial(data, detector, out, trial_args, info)
            else:
                frames_per_window = max(1, int(round(trial_args.window / trial_args.frame_period)))
                data, manifest, info = runner.load_bank(bank, frames_per_window)
                import lidar_pipeline as lp
                params = lp.Params(roi=tuple(trial_args.roi), z_min=trial_args.z_min,
                                   leaf=trial_args.leaf, radius=trial_args.radius,
                                   k_max=trial_args.k_max, k_min=trial_args.k_min,
                                   min_voxels=trial_args.min_voxels,
                                   cluster_tol=trial_args.cluster_tol,
                                   cluster_min_pts=trial_args.cluster_min_pts)
                summary = runner.run_trial(data, params, out, trial_args, info)
            tm.write_json(out / "bank_manifest.json", manifest)
            print(runner.one_line(summary), flush=True)
            failures += 0 if summary["ok"] else 1
        except (Exception, SystemExit) as exc:  # one bad bank must not end the round
            failures += 1
            out.mkdir(parents=True, exist_ok=True)
            tm.write_json(out / "FAILED.json", {"bank": str(bank), "error": repr(exc)})
            print(f"FAILED {name}: {exc!r}", flush=True)
    print(f"round {args.round} done: {len(order)} banks, {failures} with problems, "
          f"{(time.monotonic() - start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
