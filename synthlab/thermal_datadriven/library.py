"""Layer library: real person clips (with or without the hand-held lamp) and lamp blobs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LAYER_DIR = REPO / "data_local" / "thermal_datadriven" / "layers"
FOCAL_PX = 80.0 / np.tan(np.radians(57.0 / 2))      # 147 px from the 57 deg HFOV
BODY_WIDTH_M = 0.42                                  # ASSUMED shoulder width for depth ordering


class Track:
    """One person slot of one build trial: per-frame alpha / fg / lamp + metadata."""

    def __init__(self, tag, k, z):
        self.tag, self.k = tag, k
        sel = z["label"] == k
        self.alpha = np.where(sel, z["alpha"], 0).astype(np.uint8)
        self.fg = z["fg"]
        self.lamp = z["lamp"] & sel
        self.src_amb = float(np.median(z["src_bg"]))
        metas = json.loads(str(z["meta"]))
        self.meta = [m[k - 1] for m in metas]
        self.present = np.array([m is not None for m in self.meta])
        self.has_lamp = np.array([bool(m and m["lamp_px"] > 0) for m in self.meta])
        self.cut_bottom = np.array([bool(m and m["cut_bottom"]) for m in self.meta])
        self.area = np.array([m["area"] if m else 0 for m in self.meta])
        w = np.array([m["body_width_px"] if m else np.nan for m in self.meta])
        self.depth_m = FOCAL_PX * BODY_WIDTH_M / np.where(np.isfinite(w), w, np.nan)
        self.n = len(self.meta)

    def clip_indices(self, start, n):
        """Frame indices for an n-frame clip from `start`, ping-ponging at the trial ends."""
        period = 2 * (self.n - 1)
        idx = (start + np.arange(n)) % period
        return np.where(idx < self.n, idx, period - idx)


class LayerLibrary:
    def __init__(self, layer_dir=DEFAULT_LAYER_DIR, tags=None):
        self.dir = Path(layer_dir)
        files = sorted(self.dir.glob("*.npz"))
        if tags is not None:
            files = [f for f in files if any(f.stem.startswith(t) for t in tags)]
        if not files:
            raise FileNotFoundError(f"no layer files in {self.dir}; run build_models first")
        self.tracks = []
        for f in files:
            z = dict(np.load(f))
            for k in range(1, int(z["label"].max()) + 1):
                tr = Track(f.stem, k, z)
                if tr.present.mean() > 0.3:
                    self.tracks.append(tr)
        self._blobs = None

    def pick_clip(self, rng, n, lamp=None, max_area_frac=0.30, tries=200):
        """Pick (track, frame indices) with the person present >= 90 % of the clip.

        lamp=False -> no frame of the clip shows lamp pixels; lamp=True -> most frames do.
        """
        for _ in range(tries):
            tr = self.tracks[int(rng.integers(len(self.tracks)))]
            idx = tr.clip_indices(int(rng.integers(tr.n)), n)
            if tr.present[idx].mean() < 0.9:
                continue
            if np.median(tr.area[idx]) > max_area_frac * 120 * 160:
                continue
            lf = tr.has_lamp[idx].mean()
            if lamp is False and lf > 0:
                continue
            if lamp is True and lf < 0.5:
                continue
            return tr, idx
        raise RuntimeError(f"no clip found for lamp={lamp}, n={n}")

    def lamp_blobs(self, max_blobs=200, seed=0):
        """Isolated crops of the real lit heat lamp: list of (alpha float32, fg float32)."""
        if self._blobs is not None:
            return self._blobs
        rng = np.random.default_rng(seed)
        out = []
        cands = [(tr, i) for tr in self.tracks for i in np.nonzero(tr.has_lamp)[0][::7]]
        for j in rng.permutation(len(cands)):
            tr, i = cands[j]
            lab, n = ndi.label(tr.lamp[i])
            if n == 0:
                continue
            sizes = ndi.sum(tr.lamp[i], lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
            if not (40 <= m.sum() <= 700):
                continue
            ys, xs = np.nonzero(m)
            if ys.min() == 0 or xs.min() == 0 or ys.max() == 119 or xs.max() == 159:
                continue
            a = ndi.gaussian_filter(m.astype(np.float32), 0.8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            f = tr.fg[i][ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
            if (f > 0).mean() < 0.95:
                continue
            f[f == 0] = np.median(f[f > 0])
            out.append((a, f, tr.src_amb))
            if len(out) >= max_blobs:
                break
        self._blobs = out
        return out
