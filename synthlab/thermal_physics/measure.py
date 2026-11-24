"""Sensor-statistics measurements that run identically on real and simulated stacks.

Input everywhere: `x` float array (N,120,160) in TLinear counts (0.01 K) and `t`
capture times in seconds. All outputs are plain floats in counts unless named _mk/_k.
"""
import csv
import glob
import os

import numpy as np

from . import bie

DATA_ROOT = "/Users/alibaduqail/Documents/thermal camera research work/data"


def load_run(run_dir, max_frames=None):
    """Load a real Experiment-A trial folder -> (frames uint16 (N,H,W), t seconds, csv rows)."""
    files = sorted(glob.glob(os.path.join(run_dir, "thermal_raw", "raw_*.npy")))
    if max_frames:
        files = files[:max_frames]
    x = np.stack([np.load(f) for f in files])
    rows = []
    p = os.path.join(run_dir, "trial_data.csv")
    if os.path.exists(p):
        with open(p) as f:
            rows = list(csv.DictReader(f))[: len(files)]
    if rows:
        t = np.array([float(r["thermal_capture_unix_time"]) for r in rows])
        t = t - t[0]
    else:
        t = np.arange(len(files)) / 8.7
    return x, t, rows


def robust_sd(v):
    v = np.asarray(v, float).ravel()
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def find_ffc(t, min_gap_s=1.0):
    """FFC (shutter) events show up as a capture gap > 1 s. Index = first frame after the gap."""
    dt = np.diff(t)
    return [int(i + 1) for i in np.nonzero(dt > min_gap_s)[0] if i > 0]


def quiet_mask(t, ffc_idx, before_s=1.0, after_s=45.0):
    m = np.ones(len(t), bool)
    for i in ffc_idx:
        m &= ~((t > t[i] - before_s - 2.0) & (t < t[i] + after_s))
    return m


def ffc_transient(t, frame_mean, idx):
    """Describe the global-offset transient after one FFC: step, peak, time to peak, decay time."""
    pre = frame_mean[max(0, idx - 30): idx - 1]
    base = float(np.median(pre))
    post_t = t[idx:] - t[idx]
    post = frame_mean[idx:] - base
    sm = np.convolve(post, np.ones(5) / 5, mode="same")
    k = int(np.argmax(sm[: max(5, int(np.searchsorted(post_t, 25.0)))]))
    peak = float(sm[k])
    after = np.nonzero((np.arange(len(sm)) > k) & (sm < peak / 2))[0]
    return dict(gap_s=float(t[idx] - t[idx - 1]), first_frame_step=float(post[0]),
                peak=peak, t_peak_s=float(post_t[k]),
                t_half_decay_s=float(post_t[after[0]] - post_t[k]) if len(after) else None,
                seconds_observed=float(post_t[-1]))


def temporal_noise(x, t, ffc_idx=()):
    """Temporal noise from frame-to-frame differences of a static scene."""
    x = np.asarray(x, np.float64)
    ok = quiet_mask(t, ffc_idx, 1.0, 3.0)
    pair = ok[1:] & ok[:-1]
    d = np.diff(x, axis=0)[pair]
    fm = d.mean((1, 2))
    e = d - fm[:, None, None]
    rowm = e.mean(2)                    # (n, H)
    colm = e.mean(1)                    # (n, W)
    pix = e - rowm[:, :, None] - colm[:, None, :]
    s2 = np.sqrt(2.0)
    out = dict(
        total_sd=float(np.median(d.std(0)) / s2),
        frame_flicker_sd=float(robust_sd(fm) / s2),
        row_sd=float(rowm.std() / s2),
        col_sd=float(colm.std() / s2),
        pixel_sd=float(np.median(pix.std(0)) / s2),
    )
    # spatial correlation of the pixel noise between neighbours
    out["neigh_corr_h"] = float((pix[:, :, 1:] * pix[:, :, :-1]).mean() / pix.var())
    out["neigh_corr_v"] = float((pix[:, 1:, :] * pix[:, :-1, :]).mean() / pix.var())
    # temporal autocorrelation of the noise (lag-1) from detrended pixel series; white noise
    # differenced once gives lag-1 corr of the differences = -0.5; AR(1) with rho gives -(1-rho)/2.
    dd = np.diff(x, axis=0)
    both = pair[1:] & pair[:-1]
    a, b = dd[1:][both], dd[:-1][both]
    a = a - a.mean((1, 2), keepdims=True)
    b = b - b.mean((1, 2), keepdims=True)
    r = float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))
    out["diff_lag1_corr"] = r
    out["ar1_rho"] = float(1.0 + 2.0 * r)
    out["nedt_mk"] = out["total_sd"] * 10.0
    return out


def temporal_psd(x, t, ffc_idx=(), seg=128):
    """Mean per-pixel temporal power spectrum over quiet segments. Returns (freq Hz, psd counts^2/Hz)."""
    ok = quiet_mask(t, ffc_idx)
    fs = 1.0 / np.median(np.diff(t))
    runs, start = [], None
    for i, o in enumerate(np.append(ok, False)):
        if o and start is None:
            start = i
        if not o and start is not None:
            runs.append((start, i))
            start = None
    acc, n = 0.0, 0
    win = np.hanning(seg)
    for a, b in runs:
        for s in range(a, b - seg + 1, seg // 2):
            blk = np.asarray(x[s:s + seg, ::4, ::4], np.float64)
            blk = blk - blk.mean(0, keepdims=True)
            f = np.fft.rfft(blk * win[:, None, None], axis=0)
            acc = acc + (np.abs(f) ** 2).mean((1, 2))
            n += 1
    if n == 0:
        return None, None
    psd = acc / n / (fs * (win ** 2).sum()) * 2.0
    return np.fft.rfftfreq(seg, 1.0 / fs), psd


def spatial_fpn(x, t, ffc_idx=()):
    """Decompose the temporal-mean image of a FLAT scene into low-frequency
    non-uniformity (2nd-order polynomial), column, row and pixel fixed-pattern noise."""
    ok = quiet_mask(t, ffc_idx)
    if ok.sum() < 30:
        ok = np.ones(len(t), bool)
    m = np.asarray(x, np.float64)[ok].mean(0)
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)
    a = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(a, m.ravel(), rcond=None)
    low = (a @ coef).reshape(h, w)
    res = m - low
    col = res.mean(0)
    row = (res - col[None]).mean(1)
    pix = res - col[None] - row[:, None]
    n_eff = ok.sum()
    return dict(lowfreq_sd=float(low.std()), lowfreq_p2p=float(low.max() - low.min()),
                poly_coef=[float(c) for c in coef],
                col_sd=float(col.std()), row_sd=float(row.std()), pixel_sd=float(robust_sd(pix)),
                residual_sd=float(res.std()), n_frames=int(n_eff)), m, res


def fpn_change_across_ffc(x, t, idx, settle_s=40.0, n_avg=60):
    """SD of (mean image long after FFC) - (mean image before): the part of the FPN that the
    shutter re-draws. Scene must be static. Returns None if the run is too short."""
    after = np.nonzero(t > t[idx] + settle_s)[0]
    if idx < n_avg or len(after) < n_avg:
        return None
    a = np.asarray(x[idx - n_avg - 2: idx - 2], np.float64).mean(0)
    b = np.asarray(x[after[0]: after[0] + n_avg], np.float64).mean(0)
    d = b - a
    d = d - d.mean()
    col = d.mean(0)
    row = (d - col[None]).mean(1)
    pix = d - col[None] - row[:, None]
    return dict(total_sd=float(d.std()), col_sd=float(col.std()), row_sd=float(row.std()),
                pixel_sd=float(robust_sd(pix)))


def drift_per_min(t, frame_mean, ffc_idx=()):
    ok = quiet_mask(t, ffc_idx)
    if ok.sum() < 50:
        return None
    # longest quiet stretch
    idx = np.nonzero(ok)[0]
    splits = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    seg = max(splits, key=len)
    if t[seg[-1]] - t[seg[0]] < 15:
        return None
    slope = np.polyfit(t[seg], frame_mean[seg], 1)[0]
    return dict(slope_counts_per_min=float(slope * 60.0), span_s=float(t[seg[-1]] - t[seg[0]]),
                detrended_sd=float(np.std(frame_mean[seg] - np.polyval(np.polyfit(t[seg], frame_mean[seg], 1), t[seg]))))


def edge_psf_sigma(img, n_edges=40, half=6):
    """Gaussian-PSF sigma (px) from the sharpest edges of a (temporal-mean) image.
    For each of the strongest gradient pixels take the 1-D profile along the dominant axis and
    use sigma = step / (sqrt(2 pi) * max slope) (exact for an ideal step blurred by a Gaussian).
    Real edges are not ideal steps, so this is an UPPER bound; we report the low quantile."""
    img = np.asarray(img, np.float64)
    gy, gx = np.gradient(img)
    g = np.hypot(gx, gy)
    h, w = img.shape
    g[:half + 1] = g[-half - 1:] = 0
    g[:, :half + 1] = g[:, -half - 1:] = 0
    order = np.argsort(g.ravel())[::-1]
    sig, used = [], np.zeros_like(g, bool)
    for k in order:
        y, x = divmod(int(k), w)
        if g[y, x] <= 0:
            break
        if used[y, x]:
            continue
        used[max(0, y - 3): y + 4, max(0, x - 3): x + 4] = True
        prof = img[y, x - half: x + half + 1] if abs(gx[y, x]) >= abs(gy[y, x]) else img[y - half: y + half + 1, x]
        slope = np.abs(np.diff(prof)).max()
        step = abs(prof[-3:].mean() - prof[:3].mean())
        # correction for oblique edges: profile along an axis sees sigma / cos(angle)
        cosang = max(abs(gx[y, x]), abs(gy[y, x])) / (g[y, x] + 1e-12)
        if slope > 0 and step > 8 * 2.5:
            sig.append(step / (np.sqrt(2 * np.pi) * slope) * cosang)
        if len(sig) >= n_edges:
            break
    if len(sig) < 5:
        return None
    sig = np.array(sig)
    return dict(sigma_px_p10=float(np.percentile(sig, 10)), sigma_px_median=float(np.median(sig)), n_edges=int(len(sig)))


def radial_psd(img):
    """Radially averaged 2-D power spectrum of one frame (mean removed). Returns (cycles/px, power)."""
    a = np.asarray(img, np.float64)
    a = a - a.mean()
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.fftshift(np.fft.fft2(a * win))) ** 2 / (win ** 2).sum()
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.hypot(fx, fy)
    bins = np.linspace(0, 0.5, 31)
    which = np.digitize(r.ravel(), bins) - 1
    out = np.array([p.ravel()[which == i].mean() if np.any(which == i) else np.nan for i in range(30)])
    return 0.5 * (bins[1:] + bins[:-1]), out


def ht_stats(x, step=1):
    v = bie.ht_series(x[::step])
    return dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()), max=float(v.max()), n=int(len(v)))


def summarize(x, t, flat=False, ht_step=4):
    """One-call summary used for both real and simulated stacks."""
    xf = np.asarray(x, np.float64)
    fm = xf.mean((1, 2))
    ffc = find_ffc(t)
    out = dict(n_frames=int(len(x)), duration_s=float(t[-1] - t[0]),
               fps_effective=float((len(t) - 1) / (t[-1] - t[0])), dt_median_s=float(np.median(np.diff(t))),
               mean_c=float(fm.mean() / 100 - 273.15), min_c=float(xf.min() / 100 - 273.15),
               max_c=float(xf.max() / 100 - 273.15),
               spatial_sd_frame=float(np.median(xf.std((1, 2)))),
               n_ffc=len(ffc), ffc=[ffc_transient(t, fm, i) for i in ffc],
               temporal=temporal_noise(xf, t, ffc), drift=drift_per_min(t, fm, ffc), ht=ht_stats(x, ht_step))
    if flat:
        out["fpn"], mean_img, _ = spatial_fpn(xf, t, ffc)
        out["fpn_change_across_ffc"] = [fpn_change_across_ffc(xf, t, i) for i in ffc]
    return out
