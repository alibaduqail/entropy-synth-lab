"""Background tegrastats reader: timestamps + VDD_IN (module input power, mW)."""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import List, Optional, Tuple

_VDD = re.compile(r"VDD_IN (\d+)mW")
_TEMP = re.compile(r"cpu@([\d.]+)C")


class TegraLogger:
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: List[Tuple[float, float, float]] = []   # (unix time, mW, cpu temp C)
        self.proc: Optional[subprocess.Popen] = None
        self.available = shutil.which("tegrastats") is not None

    def start(self) -> "TegraLogger":
        if not self.available:
            return self
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            m = _VDD.search(line)
            if m:
                t = _TEMP.search(line)
                self.samples.append((time.time(), float(m.group(1)),
                                     float(t.group(1)) if t else float("nan")))

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()

    def window(self, t0: float, t1: float) -> dict:
        """Mean/std VDD_IN in [t0, t1] (unix seconds)."""
        vals = [mw for (t, mw, _) in self.samples if t0 <= t <= t1]
        temps = [c for (t, _, c) in self.samples if t0 <= t <= t1]
        if not vals:
            return {"vdd_in_mw_mean": None, "vdd_in_mw_std": None, "n_power_samples": 0,
                    "cpu_temp_c": None}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"vdd_in_mw_mean": mean, "vdd_in_mw_std": var ** 0.5,
                "n_power_samples": len(vals), "cpu_temp_c": sum(temps) / len(temps)}
