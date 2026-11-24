"""Plot the all-night tegrastats log: python tools/plot_overnight.py <log> <out.png>"""
import datetime as dt
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np

T, P, C = [], [], []
for line in open(sys.argv[1]):
    m = re.match(r"(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d)", line)
    p = re.search(r"VDD_IN (\d+)mW", line)
    t = re.search(r"tj@([\d.]+)C", line)
    if m and p and t:
        T.append(dt.datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S"))
        P.append(int(p.group(1)) / 1000)
        C.append(float(t.group(1)))
P, C = np.array(P), np.array(C)
sm = np.convolve(P, np.ones(12) / 12, mode="same")
fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
ax[0].plot(T, P, color="#c9d6ea", lw=0.5)
ax[0].plot(T, sm, color="#1f5fbf", lw=1.2)
ax[0].set_ylabel("Module power VDD_IN (W)")
ax[1].plot(T, C, color="#c0392b", lw=1)
ax[1].set_ylabel("Junction temp (°C)")
ax[1].set_xlabel("Time (EST), 23–24 Nov 2026")
ax[1].xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print(f"{len(T)} samples to {T[-1]}; mean {P.mean():.2f} W, max {P.max():.2f} W, Tj max {C.max():.1f} C, {P.sum() * 5 / 3600:.1f} Wh")
