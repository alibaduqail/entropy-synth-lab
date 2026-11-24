"""Extract real foreground layers (people, hand-held heat lamp, heated surfaces) from A1 trials.

Method: frame - (session empty-room background + per-frame offset) = temperature residual.
A person is the connected region whose smoothed |residual| exceeds a threshold and that
contains a clearly warm core.  For each frame we keep
  alpha   soft matte, = residual / (foreground temperature - background) at edge pixels,
          where the foreground temperature of an edge pixel is taken from the nearest
          interior pixel (so the Lepton's own optical blur is what softens the edge);
  fg      "un-mixed" foreground temperature in counts (absolute, as measured);
  label   person index (1..n, ordered left to right), 0 elsewhere;
  lamp    mask of pixels hotter than LAMP_T_C (the hand-held heat lamp and its halo).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

H, W = 120, 160
LAMP_T_C = 38.0                 # hotter than exposed skin as seen by a Lepton -> lamp
LAMP_COUNTS = (LAMP_T_C + 273.15) / 0.01
THR = 60.0                      # counts (0.6 K) on the 3x3-smoothed |residual|
CORE = 150.0                    # counts: a layer must contain pixels this much warmer
MIN_AREA = 60


def frame_offset(diff, lo_pct=15, band=40.0, iters=3):
    """Robust per-frame offset of (frame - bg): sigma-clipped median starting from a low pct."""
    o = np.percentile(diff, lo_pct)
    for _ in range(iters):
        sel = np.abs(diff - o) < band
        if sel.sum() < 200:
            break
        o = np.median(diff[sel])
    return float(o)


def split_column(mask, max_people):
    """Column at which a blob holding two side-by-side people should be cut, or None."""
    if max_people < 2:
        return None
    cols = mask.sum(0).astype(float)
    xs = np.nonzero(cols)[0]
    if len(xs) < 50:
        return None
    prof = ndi.uniform_filter1d(cols, 5)
    x0, x1 = xs[0], xs[-1]
    lo, hi = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
    v = lo + int(np.argmin(prof[lo:hi + 1]))
    if prof[v] > 0.8 * min(prof[x0:v].max(), prof[v:x1 + 1].max()):
        return None
    return v


def extract_frame(frame, bg, max_people=1):
    """Return dict(alpha float32, fg float32, label uint8, lamp bool, offset float, res)."""
    frame = np.asarray(frame, np.float32)
    diff = frame - bg
    off = frame_offset(diff)
    res = diff - off
    sm = ndi.uniform_filter(res, 3)
    cand = np.abs(sm) > THR
    cand = ndi.binary_opening(cand, iterations=1)
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3), bool), iterations=2)
    lab, n = ndi.label(cand)
    blobs = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < MIN_AREA or (res[m] > CORE).sum() < 20:
            continue
        blobs.append(ndi.binary_fill_holes(m))
    blobs.sort(key=lambda m: -m.sum())
    alpha = np.zeros((H, W), np.float32)
    fg = np.zeros((H, W), np.float32)
    label = np.zeros((H, W), np.uint8)
    parts = []                                  # (x_centre, selector mask)
    for m in blobs[:max_people]:
        interior = ndi.binary_erosion(m, iterations=1, border_value=1)
        if interior.sum() < 20:
            interior = m
        outer = ndi.binary_dilation(m, iterations=2) & (alpha == 0)
        _, (iy, ix) = ndi.distance_transform_edt(~interior, return_indices=True)
        fgk = frame[iy, ix]
        denom = fgk - (bg + off)
        good = np.abs(denom) > 40.0
        a = np.where(good, res / np.where(good, denom, 1.0), m.astype(np.float32))
        a = np.clip(a, 0.0, 1.0)
        a[interior] = 1.0
        a[~outer] = 0.0
        a[outer & ~m] = np.minimum(a[outer & ~m], 0.6)   # outside the hard mask: halo only
        sel = a > 0.02
        alpha[sel] = a[sel]
        fg[sel] = fgk[sel]
        v = split_column(m, max_people) if len(parts) + 1 < max_people else None
        if v is None:
            parts.append(sel)
        else:
            left = sel.copy(); left[:, v:] = False
            right = sel.copy(); right[:, :v] = False
            parts.extend([left, right])
    parts = [p for p in parts if p.sum() >= MIN_AREA][:max_people]
    parts.sort(key=lambda p: np.nonzero(p.any(0))[0].mean())
    for k, p in enumerate(parts, 1):
        label[p] = k
    alpha[label == 0] = 0.0
    lamp = ndi.binary_dilation(frame > LAMP_COUNTS, iterations=2) & (label > 0)
    return {"alpha": alpha, "fg": fg, "label": label, "lamp": lamp, "offset": off, "res": res}


def occupied_background(stacks, ref_bg, t_pct=15, trial_pct=25):
    """Background of a session that was never empty (person always somewhere in view).

    People are warmer than the room, so a low temporal percentile of each trial, then a low
    percentile across trials (each offset-aligned to `ref_bg`), keeps the unoccluded scene.
    Returns counts on the offset level of `ref_bg`.
    """
    per = []
    for s in stacks:
        s = np.asarray(s, np.float32)
        offs = np.array([frame_offset(f - ref_bg) for f in s[::10]])
        offs = np.interp(np.arange(len(s)), np.arange(len(s))[::10], offs)
        per.append(np.percentile(s - offs[:, None, None], t_pct, axis=0))
    return np.percentile(np.stack(per), trial_pct, axis=0).astype(np.float32)


def extract_trial(stack, bg, max_people=1):
    """Run extract_frame over a trial; returns arrays + a heated-surface map (counts)."""
    n = len(stack)
    alpha = np.zeros((n, H, W), np.uint8)
    fg = np.zeros((n, H, W), np.uint16)
    label = np.zeros((n, H, W), np.uint8)
    lamp = np.zeros((n, H, W), bool)
    off = np.zeros(n, np.float32)
    res_bg = np.full((n, H, W), np.nan, np.float32)
    for t in range(n):
        r = extract_frame(stack[t], bg, max_people)
        alpha[t] = np.rint(r["alpha"] * 255)
        fg[t] = np.rint(r["fg"])
        label[t] = r["label"]
        lamp[t] = r["lamp"]
        off[t] = r["offset"]
        clear = ~ndi.binary_dilation(r["label"] > 0, iterations=4)
        res_bg[t][clear] = r["res"][clear]
    seen = np.isfinite(res_bg).mean(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            surf = np.nanmedian(res_bg, axis=0)
    surf = np.where(seen > 0.2, surf, 0.0)
    surf = ndi.gaussian_filter(np.nan_to_num(surf), 1.0)
    surf = np.where(surf > 25.0, surf, 0.0).astype(np.float32)   # > 0.25 K static warming
    return {"alpha": alpha, "fg": fg, "label": label, "lamp": lamp, "offset": off,
            "surface": surf}


def layer_meta(alpha, label, lamp, k):
    """Bounding box / truncation / size facts for person k of one frame; None if absent."""
    m = (label == k) & (alpha >= 128)
    area = int(m.sum())
    if area < MIN_AREA:
        return None
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    rows = m[y0:y0 + max(1, (y1 - y0) // 2)]
    widths = rows.sum(1)
    return {"box": [x0, y0, x1, y1], "area": area,
            "cut_bottom": y1 >= H, "cut_top": y0 <= 0, "cut_left": x0 <= 0, "cut_right": x1 >= W,
            "body_width_px": float(np.median(widths[widths > 0])),
            "lamp_px": int((lamp & (label == k)).sum())}
