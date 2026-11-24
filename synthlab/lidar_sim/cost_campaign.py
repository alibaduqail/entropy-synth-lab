"""Drive the shared Jetson harness over all lidar-sim banks (run from the Mac worktree).

  python -m synthlab.lidar_sim.cost_campaign <campaign> --passes 2 [--seconds 30] [--unpaced]

Each pass visits every bank once in a freshly shuffled order (seeded), so thermal drift and
order effects do not line up with fog level. Every chunk (<= 10 banks, well under the 20 min
lock limit) is one `tools/jetson.sh measured` call. Results are pulled to
data_local/runs/<campaign>/ and summarised by cost_analysis.py.
"""
import argparse
import os
import random
import subprocess

from .scene import REPO

JET = os.path.join(REPO, "tools", "jetson.sh")
REMOTE = "/home/jetsonnano/synthlab/lidar-sim"
ROI = "-4 4 -4 4 -0.3 1.3"          # sensor frame; z cut removes the ceiling (as the lab ROI does)


def sh(mode, cmd):
    return subprocess.run([JET, mode, cmd], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--unpaced", action="store_true")
    ap.add_argument("--workload", default="both")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    banks = sorted(sh("run", "ls %s/banks" % REMOTE).stdout.split())
    if a.only:
        banks = [b for b in banks if b in a.only]
    rng = random.Random(a.seed)
    for p in range(a.passes):
        order = banks[:]
        rng.shuffle(order)
        for c in range(0, len(order), a.chunk):
            cmds = []
            for b in order[c:c + a.chunk]:
                out = "%s/runs/%s/p%d_%s" % (REMOTE, a.campaign, p, b)
                cmds.append("[ -e %s/summary.json ] || python3 -W ignore ~/synthlab/harness/run_lidar_bank.py "
                            "--bank %s/banks/%s --out %s --seconds %d --warmup %d --workload %s "
                            "--roi %s --z-min -0.3 %s 2>&1 | tail -1"
                            % (out, REMOTE, b, out, a.seconds, a.warmup, a.workload, ROI,
                               "--unpaced" if a.unpaced else ""))
            r = sh("measured", "mkdir -p %s/runs/%s; date +%%T; %s; date +%%T"
                   % (REMOTE, a.campaign, "; ".join(cmds)))
            print(r.stdout, r.stderr[-300:], flush=True)
    local = os.path.join(REPO, "data_local", "runs")
    os.makedirs(local, exist_ok=True)
    subprocess.run([JET, "pull", "%s/runs/%s" % (REMOTE, a.campaign), local + "/"])


if __name__ == "__main__":
    main()
