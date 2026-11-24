"""Generate the standard set of LiDAR banks (run on the Jetson under the heavy lock).

  python3 -m synthlab.lidar_sim.make_banks cond|sweep|thin [--root ~/synthlab/lidar-sim/banks]

Groups
  cond  : A0 A1 A2 A3 B1 B2 B3 P1 P2 P3                (B0 == A1: same scene, no fog)
  sweep : fog_aXpXX = A1 scene + uniform fog, 13 extinction levels (Hypothesis C knee)
  thin  : point-count controls. clear (A1) and dense fog (alpha 1.0) clouds randomly thinned to
          50 / 25 / 12 % of their points -> point count varies while the scene/fog does not.
Existing banks are skipped. Each bank = 5 windows (50 frames), seed recorded in manifest.json.
"""
import argparse
import os
import sys

from .bank import write_bank

SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00]


def plan(group):
    if group == "cond":
        return [(c, dict(condition=c, seed=1000 + i)) for i, c in
                enumerate(["A0", "A1", "A2", "A3", "B1", "B2", "B3", "P1", "P2", "P3"])]
    if group == "sweep":
        return [("fog_a%sp%02d" % (int(a), round(a % 1 * 100)), dict(condition="A1", alpha=a, seed=2000 + i))
                for i, a in enumerate(SWEEP)]
    if group == "thin":
        out = []
        for tag, a in (("clear", 0.0), ("fog1p00", 1.0)):
            for f in (0.50, 0.25, 0.12):
                out.append(("thin_%s_%03d" % (tag, round(f * 100)),
                            dict(condition="A1", alpha=a, thin=f, seed=3000 + len(out))))
        return out
    raise SystemExit("unknown group " + group)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group")
    ap.add_argument("--root", default="~/synthlab/lidar-sim/banks")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    for name, kw in plan(a.group):
        if a.only and name not in a.only:
            continue
        out = os.path.join(os.path.expanduser(a.root), name)
        if os.path.exists(os.path.join(out, "manifest.json")):
            print("skip", name)
            continue
        cond, seed = kw.pop("condition"), kw.pop("seed")
        man = write_bank(out, cond, a.windows * 10, seed, **kw)
        print("%s: %.0f ms/frame, %.1f s total" % (name, man["gen_ms_per_frame_mean"], man["gen_wall_s"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
