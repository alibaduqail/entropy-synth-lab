"""Background tegrastats logger: VDD_IN (module input power, mW), GPU load, RAM, temperature."""
import csv
import re
import subprocess
import threading
import time

_PAT = {
    "vdd_in_mw": re.compile(r"VDD_IN (\d+)mW"),
    "gpu_pct": re.compile(r"GR3D_FREQ (\d+)%"),
    "ram_mb": re.compile(r"RAM (\d+)/"),
    "gpu_c": re.compile(r"gpu@([\d.]+)C"),
}


def parse_line(line):
    out = {}
    for k, p in _PAT.items():
        m = p.search(line)
        out[k] = float(m.group(1)) if m else float("nan")
    return out


class PowerLog:
    def __init__(self, path, interval_ms=1000, tag=""):
        self.path, self.interval_ms, self.tag = path, interval_ms, tag
        self.rows = []
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return self
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            r = parse_line(line)
            r["unix_time"] = time.time()
            r["tag"] = self.tag
            self.rows.append(r)

    def mark(self, tag):
        self.tag = tag

    def mean_w(self, tag=None):
        v = [r["vdd_in_mw"] for r in self.rows if tag is None or r["tag"] == tag]
        v = [x for x in v if x == x]
        return sum(v) / len(v) / 1000.0 if v else float("nan")

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        if self.path and self.rows:
            new = True
            try:
                new = open(self.path).read(1) == ""
            except OSError:
                pass
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["unix_time", "tag", "vdd_in_mw", "gpu_pct", "ram_mb", "gpu_c"])
                if new:
                    w.writeheader()
                w.writerows(self.rows)
