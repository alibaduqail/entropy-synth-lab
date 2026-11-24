"""Parametric thermal scene generator for the sponge search.

A search vector x in [0,1]^D is decoded into a JSON-able scene description (physical
values: temperatures in Celsius, sizes in Lepton pixels) and rendered into TLinear uint16
frames (0.01 K per count, 160x120) - the format the real pipeline consumes.

Tiers
-----
layout   : "plausible" - discrete warm objects (people-like silhouettes, mugs, hand
           warmers, a regular grid = hand-warmer array / lamp behind a perforated mask) on
           a room-temperature background, optical blur >= the assumed Lepton PSF, NEDT noise.
panel    : "plausible" - one heated panel with a free two-level pattern (warm plate with
           low-emissivity tape or a perforated mask), coarse cells, same blur and noise.
pixels   : "unconstrained" - any 8-bit image after the pipeline's fixed 10-50 C mapping
           (upper bound; not buildable).

Everything here is simulated; the PSF sigma and noise levels are assumptions (see docs).
"""
from __future__ import annotations

import math

import numpy as np

try:  # cv2 is present on the Jetson and in the dev venv
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

W, H = 160, 120
SS = 4  # supersampling factor for anti-aliased object edges
HFOV_DEG = 57.0
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))  # 147.3 px focal length
K_PER_COUNT = 0.01
PSF_SIGMA_PX = 0.7  # assumed Lepton 3.5 optical blur (Gaussian sigma, pixels)
T_OBJ_MIN, T_OBJ_MAX = 20.0, 45.0  # plausible-tier object temperature range (C)
PIPE_TMIN, PIPE_TMAX = 10.0, 50.0  # the pipeline's fixed mapping range

N_OBJ = 12
GLOBAL_KEYS = ["bg_temp", "grad_mag", "grad_angle", "blur", "mist", "nedt"]
GRID_KEYS = ["on", "rows", "cols", "cx", "cy", "pitch_x", "pitch_y", "w", "h", "temp",
             "shape", "stagger"]
OBJ_KEYS = ["on", "kind", "x", "y", "size", "aspect", "temp"]
KINDS = ["person", "ellipse", "rect", "blob"]
DIM_LAYOUT = len(GLOBAL_KEYS) + len(GRID_KEYS) + N_OBJ * len(OBJ_KEYS)

PANEL_ROWS, PANEL_COLS = 15, 20  # 8x8 px cells
DIM_PANEL = 4 + PANEL_ROWS * PANEL_COLS


def celsius_to_raw(temp_c):
    raw = np.rint((np.asarray(temp_c, dtype=np.float64) + 273.15) / K_PER_COUNT)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def raw_to_celsius(raw):
    return np.asarray(raw).astype(np.float32) * K_PER_COUNT - 273.15


def _lerp(u, lo, hi):
    return lo + (hi - lo) * float(min(1.0, max(0.0, u)))


def decode_layout(x):
    """Map x in [0,1]^DIM_LAYOUT to a scene dict (values clipped into plausible ranges)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    assert x.size == DIM_LAYOUT, (x.size, DIM_LAYOUT)
    g = dict(zip(GLOBAL_KEYS, x[:6]))
    scene = {
        "tier": "layout",
        "bg_temp": _lerp(g["bg_temp"], 18.0, 30.0),
        "grad_mag": _lerp(g["grad_mag"], 0.0, 2.0),  # C across the frame
        "grad_angle": _lerp(g["grad_angle"], 0.0, 2 * math.pi),
        "blur": PSF_SIGMA_PX + _lerp(g["blur"], 0.0, 1.5),  # PSF + defocus/mist
        "mist": _lerp(g["mist"], 0.0, 0.5),  # fractional contrast loss
        "nedt": _lerp(g["nedt"], 0.03, 0.12),  # temporal noise sigma (K)
        "objects": [],
    }
    q = dict(zip(GRID_KEYS, x[6:18]))
    w = _lerp(q["w"], 3.0, 24.0)
    h = _lerp(q["h"], 3.0, 24.0)
    scene["grid"] = {
        "on": bool(q["on"] > 0.5),
        "rows": int(round(_lerp(q["rows"], 1, 8))),
        "cols": int(round(_lerp(q["cols"], 1, 12))),
        "cx": _lerp(q["cx"], 30.0, 130.0),
        "cy": _lerp(q["cy"], 25.0, 95.0),
        "w": w,
        "h": h,
        "pitch_x": w * _lerp(q["pitch_x"], 1.15, 3.0),
        "pitch_y": h * _lerp(q["pitch_y"], 1.15, 3.0),
        "temp": _lerp(q["temp"], 26.0, T_OBJ_MAX),
        "shape": "rect" if q["shape"] < 0.5 else "ellipse",
        "stagger": bool(q["stagger"] > 0.5),
    }
    objs = x[18:].reshape(N_OBJ, len(OBJ_KEYS))
    for row in objs:
        o = dict(zip(OBJ_KEYS, row))
        kind = KINDS[min(len(KINDS) - 1, int(o["kind"] * len(KINDS)))]
        size = _lerp(o["size"], 20.0, 110.0) if kind == "person" else _lerp(o["size"], 4.0, 40.0)
        scene["objects"].append({
            "on": bool(o["on"] > 0.5),
            "kind": kind,
            "x": _lerp(o["x"], 0.0, W),
            "y": _lerp(o["y"], 0.0, H),
            "size": size,  # person: height px; others: width px
            "aspect": _lerp(o["aspect"], 0.5, 2.0),  # height/width (ignored for person)
            "temp": _lerp(o["temp"], 24.0, T_OBJ_MAX),
        })
    return scene


def _ell(canvas, cx, cy, ax, ay, temp):
    cv2.ellipse(canvas, (int(round(cx * SS)), int(round(cy * SS))),
                (max(1, int(round(ax * SS))), max(1, int(round(ay * SS)))), 0, 0, 360,
                float(temp), -1, cv2.LINE_8)


def _rect(canvas, cx, cy, w, h, temp):
    p0 = (int(round((cx - w / 2) * SS)), int(round((cy - h / 2) * SS)))
    p1 = (int(round((cx + w / 2) * SS)), int(round((cy + h / 2) * SS)))
    cv2.rectangle(canvas, p0, p1, float(temp), -1)


def _person(canvas, cx, cy, hgt, temp, bg):
    """Upright silhouette centred at (cx, cy): bare head hottest, clothed body cooler."""
    top = cy - hgt / 2
    body = max(bg + 1.0, temp - 4.0)
    legs = max(bg + 0.5, temp - 5.5)
    hw = hgt * 0.13
    _rect(canvas, cx - hw * 0.5, top + hgt * 0.76, hw * 0.85, hgt * 0.48, legs)
    _rect(canvas, cx + hw * 0.5, top + hgt * 0.76, hw * 0.85, hgt * 0.48, legs)
    _ell(canvas, cx, top + hgt * 0.36, hw * 1.15, hgt * 0.20, body)
    _rect(canvas, cx - hw * 1.45, top + hgt * 0.38, hw * 0.6, hgt * 0.36, body)
    _rect(canvas, cx + hw * 1.45, top + hgt * 0.38, hw * 0.6, hgt * 0.36, body)
    _ell(canvas, cx, top + hgt * 0.075, hgt * 0.065, hgt * 0.075, temp)


def _background(scene, background):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if background is not None:
        base = np.asarray(background, dtype=np.float32)  # real empty-room frame (C)
    else:
        base = np.full((H, W), scene["bg_temp"], np.float32)
        ang = scene.get("grad_angle", 0.0)
        ramp = ((xx - W / 2) * math.cos(ang) + (yy - H / 2) * math.sin(ang)) / W
        base = base + scene.get("grad_mag", 0.0) * ramp
    return base


def render_clean(scene, background=None):
    """Noise-free temperature image (float32 Celsius, 120x160) for a layout scene."""
    base = _background(scene, background)
    bg = float(np.median(base))
    canvas = cv2.resize(base, (W * SS, H * SS), interpolation=cv2.INTER_LINEAR)
    g = scene.get("grid")
    if g and g["on"]:
        for r in range(g["rows"]):
            off = g["pitch_x"] / 2 if (g["stagger"] and r % 2) else 0.0
            cy = g["cy"] + (r - (g["rows"] - 1) / 2) * g["pitch_y"]
            for c in range(g["cols"]):
                cx = g["cx"] + (c - (g["cols"] - 1) / 2) * g["pitch_x"] + off
                if g["shape"] == "rect":
                    _rect(canvas, cx, cy, g["w"], g["h"], g["temp"])
                else:
                    _ell(canvas, cx, cy, g["w"] / 2, g["h"] / 2, g["temp"])
    blobs = []
    for o in scene.get("objects", []):
        if not o["on"]:
            continue
        if o["kind"] == "person":
            _person(canvas, o["x"], o["y"], o["size"], o["temp"], bg)
        elif o["kind"] == "ellipse":
            _ell(canvas, o["x"], o["y"], o["size"] / 2, o["size"] * o["aspect"] / 2, o["temp"])
        elif o["kind"] == "rect":
            _rect(canvas, o["x"], o["y"], o["size"], o["size"] * o["aspect"], o["temp"])
        else:
            blobs.append(o)
    img = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    if blobs:  # diffuse warm patches (e.g. a hand warmer under cloth): additive Gaussians
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        for o in blobs:
            sx, sy = o["size"] / 2.5, o["size"] * o["aspect"] / 2.5
            gauss = np.exp(-0.5 * (((xx - o["x"]) / sx) ** 2 + ((yy - o["y"]) / sy) ** 2))
            img = np.maximum(img, bg + (o["temp"] - bg) * gauss)
    mist = scene.get("mist", 0.0)
    if mist > 0:  # mist pulls apparent temperature toward ambient
        img = bg + (img - bg) * (1.0 - mist)
    img = cv2.GaussianBlur(img, (0, 0), float(scene.get("blur", PSF_SIGMA_PX)))
    return img.astype(np.float32)


def add_noise(clean, n_frames, nedt, seed):
    """Per-frame temporal noise + a small fixed column pattern; returns uint16 TLinear."""
    rng = np.random.default_rng(seed)
    fpn = rng.normal(0.0, 0.02, size=(1, 1, W)).astype(np.float32)
    noise = rng.normal(0.0, nedt, size=(n_frames, H, W)).astype(np.float32)
    return celsius_to_raw(clean[None] + noise + fpn)


def render_layout(x, n_frames=4, seed=0, background=None):
    scene = decode_layout(x)
    clean = render_clean(scene, background)
    return add_noise(clean, n_frames, scene["nedt"], seed), scene


def decode_panel(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    assert x.size == DIM_PANEL
    return {
        "tier": "panel",
        "bg_temp": _lerp(x[0], 18.0, 30.0),
        "t_hot": _lerp(x[1], 30.0, T_OBJ_MAX),
        "blur": PSF_SIGMA_PX + _lerp(x[2], 0.0, 1.5),
        "nedt": _lerp(x[3], 0.03, 0.12),
        "pattern": (x[4:].reshape(PANEL_ROWS, PANEL_COLS) > 0.5).astype(np.uint8).tolist(),
    }


def render_panel(x, n_frames=4, seed=0):
    scene = decode_panel(x)
    pat = np.asarray(scene["pattern"], np.float32)
    img = scene["bg_temp"] + (scene["t_hot"] - scene["bg_temp"]) * cv2.resize(
        pat, (W, H), interpolation=cv2.INTER_NEAREST)
    clean = cv2.GaussianBlur(img, (0, 0), scene["blur"]).astype(np.float32)
    return add_noise(clean, n_frames, scene["nedt"], seed), scene


def u8_to_raw(img_u8):
    """Unconstrained tier: the TLinear frame whose pipeline 8-bit image equals img_u8."""
    t = PIPE_TMIN + np.asarray(img_u8, dtype=np.float64) * (PIPE_TMAX - PIPE_TMIN) / 255.0
    return celsius_to_raw(t)


def pipeline_u8(raw):
    """The pipeline's fixed-range 8-bit image (same formula as radiometric_bie_image)."""
    t = np.clip(raw_to_celsius(raw), PIPE_TMIN, PIPE_TMAX)
    return np.rint((t - PIPE_TMIN) * (255.0 / (PIPE_TMAX - PIPE_TMIN))).astype(np.uint8)


def empty_room(n_frames=8, seed=0, bg_temp=27.5, nedt=0.05):
    """Synthetic empty-room baseline (uniform wall + mild gradient + noise)."""
    x = np.zeros(DIM_LAYOUT)
    scene = decode_layout(x)
    scene.update(bg_temp=bg_temp, grad_mag=1.0, grad_angle=1.2, nedt=nedt)
    return add_noise(render_clean(scene), n_frames, nedt, seed), scene


def physical_size_cm(px, distance_m):
    """Size in cm of something spanning `px` Lepton pixels at `distance_m`."""
    return 100.0 * px * distance_m / F_PX


def describe(scene, distance_m=1.5):
    """Plain-language build sheet for a layout scene at an assumed stand-off distance."""
    lines = [f"background {scene['bg_temp']:.1f} C, blur sigma {scene['blur']:.2f} px, "
             f"mist {scene['mist']:.2f}, stand-off {distance_m} m"]
    g = scene.get("grid")
    if g and g["on"]:
        lines.append(
            f"grid {g['rows']}x{g['cols']} of {g['shape']}s at {g['temp']:.1f} C, each "
            f"{physical_size_cm(g['w'], distance_m):.1f}x{physical_size_cm(g['h'], distance_m):.1f} cm, "
            f"pitch {physical_size_cm(g['pitch_x'], distance_m):.1f}x"
            f"{physical_size_cm(g['pitch_y'], distance_m):.1f} cm"
            f"{', staggered rows' if g['stagger'] else ''}")
    for o in scene.get("objects", []):
        if o["on"]:
            wpx = o["size"] * (0.4 if o["kind"] == "person" else 1.0)
            hpx = o["size"] if o["kind"] == "person" else o["size"] * o["aspect"]
            lines.append(f"{o['kind']} at pixel ({o['x']:.0f},{o['y']:.0f}), {o['temp']:.1f} C, "
                         f"{physical_size_cm(wpx, distance_m):.0f}x"
                         f"{physical_size_cm(hpx, distance_m):.0f} cm")
    return lines
