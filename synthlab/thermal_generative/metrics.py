"""Sample-quality and memorisation metrics for uint16 TLinear frame sets (N,120,160)."""
import numpy as np

from .mapping import bie_image, counts_to_celsius, ht_of_counts


def temp_hist(frames, lo=15.0, hi=60.0, bins=450):
    t = counts_to_celsius(frames).ravel()
    h, edges = np.histogram(np.clip(t, lo, hi), bins=bins, range=(lo, hi))
    return h / h.sum(), edges


def wasserstein_c(frames_a, frames_b, n_q=2000):
    """1-D Wasserstein-1 distance between the two pixel-temperature distributions, in deg C."""
    q = (np.arange(n_q) + 0.5) / n_q
    a = np.quantile(counts_to_celsius(frames_a).ravel(), q)
    b = np.quantile(counts_to_celsius(frames_b).ravel(), q)
    return float(np.mean(np.abs(a - b)))


def radial_spectrum(frames):
    """Mean radially averaged power spectrum of the mean-removed temperature frames (K^2)."""
    t = counts_to_celsius(frames)
    t = t - t.mean(axis=(1, 2), keepdims=True)
    p = np.mean(np.abs(np.fft.fft2(t)) ** 2, axis=0) / (t.shape[1] * t.shape[2])
    fy = np.fft.fftfreq(t.shape[1])[:, None]
    fx = np.fft.fftfreq(t.shape[2])[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    edges = np.linspace(0, 0.5, 31)
    which = np.digitize(r.ravel(), edges) - 1
    spec = np.array([p.ravel()[which == i].mean() if np.any(which == i) else np.nan for i in range(30)])
    return 0.5 * (edges[:-1] + edges[1:]), spec


def spectrum_distance(spec_a, spec_b):
    """Mean |log10 ratio| over radial bins, and the high-frequency (f>0.25 cyc/px) power ratio a/b."""
    ok = np.isfinite(spec_a) & np.isfinite(spec_b) & (spec_a > 0) & (spec_b > 0)
    d = float(np.mean(np.abs(np.log10(spec_a[ok] / spec_b[ok]))))
    hf = float(np.sum(spec_a[15:][ok[15:]]) / np.sum(spec_b[15:][ok[15:]]))
    return d, hf


def highpass_noise_mk(frames):
    """Per-frame spatial fine-scale noise: SD of (frame - 3x3 box mean) in mK, robust (MAD-based)
    so edges of people/lamp do not dominate. Returns one value per frame."""
    t = counts_to_celsius(frames)
    pad = np.pad(t, ((0, 0), (1, 1), (1, 1)), mode="edge")
    box = sum(pad[:, i:i + t.shape[1], j:j + t.shape[2]] for i in range(3) for j in range(3)) / 9.0
    res = (t - box).reshape(len(t), -1)
    mad = np.median(np.abs(res - np.median(res, axis=1, keepdims=True)), axis=1)
    return 1.4826 * mad * 1000.0


def pixel_sd_map_c(frames):
    """Per-pixel SD across the frame set (deg C) - across-sample variability at each pixel."""
    return counts_to_celsius(frames).std(axis=0)


def ht_values(frames):
    return np.array([ht_of_counts(f) for f in frames])


def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def lamp_hot_fraction(frames, thresh_c=45.0, min_px=4):
    """Fraction of frames with at least `min_px` pixels above `thresh_c` (the heat lamp signature;
    skin in these recordings stays below ~37 C)."""
    t = counts_to_celsius(frames)
    return float(np.mean((t > thresh_c).sum(axis=(1, 2)) >= min_px))


def warm_fraction(frames, delta_c=3.0):
    """Mean fraction of pixels more than delta_c above the frame median (body-heat area proxy)."""
    t = counts_to_celsius(frames)
    med = np.median(t.reshape(len(t), -1), axis=1)[:, None, None]
    return float(np.mean(t > med + delta_c))


def nn_rmse_c(query, ref, dev="cuda", exclude_same=None):
    """For each query frame, RMSE (deg C, on the pipeline's 10-50 C clipped temperature) to its
    nearest reference frame. exclude_same=(q_group, r_group) masks pairs with equal group id
    (used for the train->train baseline, leaving out the query's own trial).
    Returns (distances, nearest_index)."""
    import torch
    f = lambda a: torch.from_numpy(np.clip(counts_to_celsius(a), 10, 50).astype(np.float32)).reshape(len(a), -1)
    r = f(ref).to(dev)
    out_d, out_i = [], []
    for i in range(0, len(query), 64):
        q = f(query[i:i + 64]).to(dev)
        d = torch.cdist(q, r)
        if exclude_same is not None:
            qg = torch.as_tensor(exclude_same[0][i:i + 64], device=dev)[:, None]
            rg = torch.as_tensor(exclude_same[1], device=dev)[None, :]
            d = d.masked_fill(qg == rg, float("inf"))
        m, j = d.min(dim=1)
        out_d.append((m / np.sqrt(q.shape[1])).cpu().numpy())
        out_i.append(j.cpu().numpy())
    return np.concatenate(out_d), np.concatenate(out_i)


def to_u8_batch(frames):
    return np.stack([bie_image(f) for f in frames])
