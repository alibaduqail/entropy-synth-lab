"""Print per-bank mean +- sd from a measure.py result file (runs anywhere)."""
import json
import sys

import numpy as np

KEYS = ["frame_ms", "bie_ms", "pre_ms", "inf_ms", "post_ms", "overhead_ms", "log_ms",
        "vdd_in_mw", "energy_mj"]


def main(path):
    rows = json.load(open(path))["rows"]
    for b in dict.fromkeys(r["bank"] for r in rows):
        rr = [r for r in rows if r["bank"] == b]
        i = rr[0]
        txt = " ".join(f"{k.replace('_ms','')} {np.mean([r[k] for r in rr]):.2f}"
                       f"±{np.std([r[k] for r in rr]):.2f}" for k in KEYS)
        print(f"{b:16s} n={len(rr)} {txt} | det {i.get('n_det')} cand {i.get('n_cand')} "
              f"ht {i.get('ht', 0):.4f} sh {i.get('shannon', 0):.2f} {i.get('classes')}")


if __name__ == "__main__":
    main(sys.argv[1])
