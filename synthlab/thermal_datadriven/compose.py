"""Recombine measured pieces into new TLinear frames with controls and automatic labels.

frame_t = round( B + ambient_shift + lamp * heated_surface          (static scene)
                 (+ warm objects, + people pasted far -> near)      (real foreground layers)
                 + drift_t * drift_mult                             (replayed real drift / FFC)
                 + w * noise_t )                                    (sampled measured noise)

Assumptions (not measured, stated in docs/thermal-datadriven.md):
  * horizon row HORIZON_Y; scaling a layer about the horizon by s moves the person from
    depth Z to Z/s (pin-hole camera, level floor);
  * bottom-truncated layers are only ever enlarged (s >= 1) - shrinking would show a
    person cut off in mid-air;
  * clothing temperature follows ambient linearly, skin (SKIN_C) does not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .library import LayerLibrary
from .sensor_model import SessionModel

H, W = 120, 160
HORIZON_Y = 53.0            # ASSUMED: camera pitched ~3 deg down (scene/room_v0.json look_dir)
SKIN_C = 34.0
LAMP_C = 38.0
PSF_SIGMA = 0.7             # ASSUMED native blur (px) restored after down-scaling a layer
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "results" / "thermal-datadriven" / "models"


def c2k(c):
    return (c + 273.15) * 100.0


@dataclass
class SceneSpec:
    session: str = "hl"             # background / noise model: hl | rj | w120
    n_frames: int = 128
    n_people: int = 0
    lamp: float = 0.0               # 0 = off; 1 = as measured; scales lamp excess + heated floor
    people_hold_lamp: bool = False  # use the person-holding-lit-lamp clips (needs lamp > 0)
    n_warm_objects: int = 0         # static pasted lamp blobs
    warm_object_intensity: float = 0.5
    noise_mult: float = 1.0
    drift_mult: float = 1.0
    ambient_shift_c: float = 0.0
    motion_px: float = 12.0         # peak lateral wander of each person over the bank
    layer_tags: tuple = ()          # restrict layer sources (prefixes), () = all
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _warp(img, s, dx, xc, interp=cv2.INTER_LINEAR):
    m = np.float32([[s, 0, (1 - s) * xc + dx], [0, s, (1 - s) * HORIZON_Y]])
    return cv2.warpAffine(img, m, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def _retarget_temperature(fg, src_amb, new_amb, lamp_gain):
    """Clothing follows ambient, skin stays, lamp excess above LAMP_C is scaled."""
    skin, lampk = c2k(SKIN_C), c2k(LAMP_C)
    out = fg.copy()
    cool = fg < skin
    gain = (skin - new_amb) / max(skin - src_amb, 100.0)
    out[cool] = new_amb + (fg[cool] - src_amb) * gain
    hot = fg > lampk
    out[hot] = lampk + (fg[hot] - lampk) * lamp_gain
    return out


def _smooth_path(rng, n, amp):
    """Smooth random lateral path (px), zero at t=0, peak |amp|."""
    if amp <= 0 or n < 3:
        return np.zeros(n, np.float32)
    k = rng.standard_normal(4) / np.arange(1, 5)
    ph = rng.uniform(0, 2 * np.pi, 4)
    t = np.linspace(0, 1, n)
    p = sum(k[i] * np.sin(np.pi * (i + 1) * t + ph[i]) for i in range(4))
    p = p - p[0]
    return (amp * p / max(np.abs(p).max(), 1e-6)).astype(np.float32)


def _box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


class Generator:
    def __init__(self, model_dir=MODEL_DIR, layer_dir=None, library=None):
        self.model_dir = Path(model_dir)
        self._models = {}
        self._lib = library
        self._layer_dir = layer_dir
        z = np.load(self.model_dir / "lamp_surface_hl.npz")
        self.surface = z["surface"]

    def model(self, name):
        if name not in self._models:
            self._models[name] = SessionModel.load(self.model_dir / f"session_{name}.npz")
        return self._models[name]

    def library(self):
        if self._lib is None:
            self._lib = LayerLibrary(self._layer_dir) if self._layer_dir else LayerLibrary()
        return self._lib

    # ------------------------------------------------------------------
    def _plan_people(self, spec, rng, new_amb):
        lib = self.library()
        tracks = lib.tracks
        if spec.layer_tags:
            sub = [t for t in tracks if any(t.tag.startswith(p) for p in spec.layer_tags)]
            lib = LayerLibrary.__new__(LayerLibrary)
            lib.tracks, lib._blobs = sub, None
        plans, centres = [], []
        for _ in range(spec.n_people):
            want_lamp = True if (spec.people_hold_lamp and spec.lamp > 0) else False
            tr, idx = lib.pick_clip(rng, spec.n_frames, lamp=want_lamp)
            ok = np.nonzero(tr.present[idx])[0]
            boxes = np.array([tr.meta[idx[i]]["box"] for i in ok])
            xc = float(np.median((boxes[:, 0] + boxes[:, 2]) / 2))
            cut = bool(tr.cut_bottom[idx].mean() > 0.2)
            s = float(rng.uniform(1.0, 1.25)) if cut else float(rng.uniform(0.6, 1.2))
            # best-of-8 lateral placement: keep people apart when there is room
            best = None
            for _c in range(8):
                x_new = float(rng.uniform(18, W - 18))
                d = min([abs(x_new - c) for c in centres], default=1e9)
                if best is None or d > best[0]:
                    best = (d, x_new)
            x_new = best[1]
            centres.append(x_new)
            depth = float(np.nanmedian(tr.depth_m[idx])) / s
            plans.append({"track": tr, "idx": idx, "s": s, "xc": xc, "dx0": x_new - xc,
                          "path": _smooth_path(rng, spec.n_frames, spec.motion_px),
                          "depth": depth, "new_amb": new_amb,
                          "info": {"source": tr.tag, "slot": tr.k, "start": int(idx[0]),
                                   "scale": s, "x": x_new, "depth_m_est": depth,
                                   "bottom_truncated": cut}})
        plans.sort(key=lambda p: -p["depth"])          # far first
        return plans

    def _plan_objects(self, spec, rng, new_amb):
        objs = []
        n = spec.n_warm_objects + (1 if (spec.lamp > 0 and not spec.people_hold_lamp) else 0)
        if n == 0:
            return objs
        blobs = self.library().lamp_blobs()
        for j in range(n):
            a, f, src_amb = blobs[int(rng.integers(len(blobs)))]
            is_lamp = j >= spec.n_warm_objects
            q = spec.lamp if is_lamp else spec.warm_object_intensity * rng.uniform(0.6, 1.4)
            s = float(rng.uniform(0.7, 1.2))
            a2 = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            f2 = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            h, w = a2.shape
            x0 = int(rng.integers(2, W - w - 2))
            y0 = int(rng.integers(int(HORIZON_Y) - 25, H - h - 2)) if H - h - 2 > HORIZON_Y - 25 \
                else int(rng.integers(2, max(3, H - h - 2)))
            temp = new_amb + q * (f2 - src_amb)
            objs.append({"alpha": a2, "fg": temp, "x0": x0, "y0": y0,
                         "info": {"kind": "lamp" if is_lamp else "warm_object", "intensity": float(q),
                                  "box": [x0, y0, x0 + w, y0 + h],
                                  "peak_c": float(temp.max() / 100 - 273.15)}})
        return objs

    # ------------------------------------------------------------------
    def generate(self, spec: SceneSpec):
        """Return (frames uint16 (N,H,W), manifest dict with per-frame labels)."""
        rng = np.random.default_rng(spec.seed)
        m = self.model(spec.session)
        n = spec.n_frames
        base = m.bg + spec.ambient_shift_c * 100.0
        if spec.lamp > 0:
            base = base + spec.lamp * self.surface
        new_amb = float(np.median(base))
        objs = self._plan_objects(spec, rng, new_amb)
        for o in objs:
            a = o["alpha"]
            sl = (slice(o["y0"], o["y0"] + a.shape[0]), slice(o["x0"], o["x0"] + a.shape[1]))
            base[sl] = base[sl] * (1 - a) + o["fg"] * a
        plans = self._plan_people(spec, rng, new_amb) if spec.n_people else []
        drift, dinfo = m.sample_drift(n, rng, spec.drift_mult)
        noise = m.sample_noise(n, rng, 1.0)
        frames = np.empty((n, H, W), np.uint16)
        labels = []
        for t in range(n):
            img = base.astype(np.float32).copy()
            cover = np.zeros((H, W), np.float32)
            alphas = []
            for p in plans:
                tr, i = p["track"], int(p["idx"][t])
                a = tr.alpha[i].astype(np.float32) / 255.0
                if a.max() == 0:
                    alphas.append(None)
                    continue
                f = _retarget_temperature(tr.fg[i].astype(np.float32), tr.src_amb, p["new_amb"],
                                          spec.lamp if spec.lamp > 0 else 1.0)
                dx = p["dx0"] + p["path"][t]
                aw = _warp(a, p["s"], dx, p["xc"])
                fw = _warp(f * a, p["s"], dx, p["xc"]) / np.maximum(aw, 1e-3)
                if p["s"] < 1.0:
                    sg = PSF_SIGMA * np.sqrt(1 - p["s"] ** 2)
                    pm = cv2.GaussianBlur(fw * aw, (0, 0), sg)
                    aw = cv2.GaussianBlur(aw, (0, 0), sg)
                    fw = pm / np.maximum(aw, 1e-3)
                img = img * (1 - aw) + fw * aw
                cover = cover * (1 - aw) + aw
                alphas.append(aw)
            # visible boxes: own alpha times transparency of everyone nearer
            boxes, nearer = [], np.ones((H, W), np.float32)
            for p, aw in zip(reversed(plans), reversed(alphas)):
                if aw is None:
                    continue
                vis = (aw * nearer) >= 0.5
                if vis.sum() >= 30:
                    boxes.append(_box(vis) + [round(float(vis.sum() / max((aw >= 0.5).sum(), 1)), 3)])
                nearer = nearer * (1 - aw)
            # layers carry their own real noise; add synthetic noise so total variance = mult^2
            wgt = np.sqrt(np.maximum(spec.noise_mult ** 2 - cover ** 2, 0.0))
            img = img + drift[t] + wgt * noise[t]
            frames[t] = np.clip(np.rint(img), 0, 65535).astype(np.uint16)
            labels.append({"n_people": len(boxes), "boxes": [b[:4] for b in boxes],
                           "visible_fraction": [b[4] for b in boxes]})
        manifest = {
            "generator": "thermal_datadriven.compose", "units": "TLinear 0.01 K/count",
            "parameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()},
            "seed": spec.seed, "n_frames": n, "lamp_on": bool(spec.lamp > 0),
            "people": [p["info"] for p in plans], "objects": [o["info"] for o in objs],
            "drift": dinfo, "n_people_requested": spec.n_people,
            "labels": labels,
        }
        return frames, manifest
