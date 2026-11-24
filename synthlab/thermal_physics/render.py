"""Scene -> apparent 8-14 um band radiance at the camera aperture.

Per pixel:  L = tau * L_surf + (1 - tau) * B(T_mist)
  L_surf = e(theta) * B(T_surface) + (1 - e(theta)) * [ (1-s) * B(T_ambient) + s * L_specular ]
with e(theta) from a Schlick grazing-angle roll-off, a first-order reflected
ambient term, optional single-bounce specular, a steady-state heat-lamp warm
patch on static surfaces, and Beer-Lambert fog with path emission.

The static layer (room + furniture + fixed warm objects) is traced once and
cached; only dynamic primitives (people, hand-held lamp) are re-traced per frame.
"""
import numpy as np

from . import radiometry
from .raycast import Backend, camera_rays, trace, INF, ROOM, FURNITURE, LAMP
from .scene import PrimSet

N_TEX = 4


def _texture_tables(props, seed):
    """Per-primitive sinusoid wavevectors/phases for a smooth temperature texture."""
    rng = np.random.default_rng(seed)
    n = len(props)
    freq = np.zeros((n, N_TEX, 3))
    for i, p in enumerate(props):
        lo, hi = (0.4, 2.5) if p["label"] == ROOM else (0.05, 0.35)
        wl = np.exp(rng.uniform(np.log(lo), np.log(hi), N_TEX))
        v = rng.normal(size=(N_TEX, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        freq[i] = v * (2 * np.pi / wl)[:, None]
    phase = rng.uniform(0, 2 * np.pi, (n, N_TEX))
    return freq, phase


def _ref_points(ps):
    refs = [np.zeros(3)] * 6 + [b[0] for b in ps.box] + [e[0] for e in ps.ell] + [c[0] for c in ps.cap]
    return np.stack(refs)


class Layer:
    """Backend copies of a PrimSet plus its texture tables."""

    def __init__(self, bk, ps, seed):
        self.ps = ps
        self.prims = ps.to_backend(bk)
        f, ph = _texture_tables(ps.all_props(), seed)
        self.freq, self.phase, self.ref = bk.arr(f), bk.arr(ph), bk.arr(_ref_points(ps))


class Renderer:
    def __init__(self, room, cam, backend="numpy", device=None, supersample=2, seed=0,
                 wall_gradient_k_per_m=0.3, ambient_c=None, lut=None):
        self.room, self.cam = room, dict(cam)
        self.bk = Backend(backend, device)
        self.ss = int(supersample)
        self.seed = int(seed)
        self.lut = lut or radiometry.default_lut()
        self.h, self.w = int(cam["height"]), int(cam["width"])
        self.o = self.bk.arr(np.asarray(cam["position"], float))
        self.d = self.bk.arr(camera_rays(cam, self.ss))
        self.size = self.bk.arr(np.asarray(room["room"]["size_xyz"], float))
        self.air_c = float(room["room"]["air_temp_c"])
        self.ambient_c = self.air_c if ambient_c is None else float(ambient_c)
        self.wall_grad = float(wall_gradient_k_per_m)
        self.static = None
        self.lamp = None

    # ---- radiometry helpers on the active backend
    def B(self, t_c):
        if self.bk.kind == "numpy":
            return self.lut.radiance(t_c + 273.15)
        return self.lut.radiance_t(t_c + 273.15)

    def B_scalar(self, t_c):
        return float(self.lut.radiance(np.array([t_c + 273.15]))[0])

    # ---- shading
    def _surface_temp(self, layer, idx, p, n, is_static):
        xp = self.bk.xp
        pr = layer.prims
        q = p - layer.ref[idx]
        arg = (layer.freq[idx] * q[:, None, :]).sum(-1) + layer.phase[idx]
        tex = xp.sin(arg).sum(-1) * (1.0 / np.sqrt(N_TEX / 2.0))        # ~unit variance
        t = pr["temp_c"][idx] + pr["tex"][idx] * tex
        if is_static:
            wall = (idx < 4).to(t.dtype) if self.bk.kind != "numpy" else (idx < 4).astype(t.dtype)
            t = t + wall * self.wall_grad * (p[:, 2] - 0.5 * self.size[2])
            if self.lamp is not None:
                t = t + self._lamp_dt(p, n, pr["label"][idx])
        return t

    def _lamp_dt(self, p, n, label):
        """Steady-state surface warming under the lamp: dT = a * E / h (first order, no shadows)."""
        xp, lp = self.bk.xp, self.lamp
        pos = self.bk.arr(np.asarray(lp["position"], float))
        aim = np.asarray(lp.get("aim", (0, 0, -1.0)), float)
        aim = self.bk.arr(aim / np.linalg.norm(aim))
        v = p - pos
        r2 = (v * v).sum(-1).clip(0.02, None) if self.bk.kind == "numpy" else (v * v).sum(-1).clamp(min=0.02)
        r = xp.sqrt(r2)
        cos_off = ((v * aim).sum(-1) / r).clip(-1.0, 1.0)
        theta = xp.arccos(cos_off)
        th0 = np.radians(lp.get("cone_half_angle_deg", 35.0))
        intensity = lp.get("power_w", 250.0) * lp.get("radiant_fraction", 0.7) / (np.pi * th0 ** 2)   # W/sr on axis
        cos_i = (-(v * n).sum(-1) / r).clip(0.0, 1.0)
        e = intensity * xp.exp(-(theta / th0) ** 2) * cos_i / r2
        dt = lp.get("absorptivity", 0.9) * e / lp.get("h_w_m2k", 9.0) * lp.get("warm_fraction", 1.0)
        ok = (label == ROOM) | (label == FURNITURE)
        return xp.where(ok, dt.clip(0.0, lp.get("max_dt_k", 60.0)), xp.zeros_like(dt))

    def _shade(self, layer, t, idx, n, d, o, is_static, l_spec=None):
        xp = self.bk.xp
        pr = layer.prims
        hit = idx >= 0
        idx = xp.where(hit, idx, xp.zeros_like(idx))
        p = o.reshape(-1, 3) + xp.where(hit, t, xp.zeros_like(t))[:, None] * d
        ts = self._surface_temp(layer, idx, p, n, is_static)
        cos = (-(n * d).sum(-1)).clip(0.0, 1.0)
        r0 = 1.0 - pr["emissivity"][idx]
        refl = r0 + (1.0 - r0) * (1.0 - cos) ** 5
        l_amb = self.B_scalar(self.ambient_c)
        l_ref = l_amb if l_spec is None else (1.0 - pr["specular"][idx]) * l_amb + pr["specular"][idx] * l_spec
        return (1.0 - refl) * self.B(ts) + refl * l_ref, p

    # ---- public API
    def set_static(self, ps, lamp=None):
        """Trace and cache the static layer. `lamp` (dict: position, aim, power_w, ...) adds the
        steady-state warm patch; the lamp's own primitives must be in `ps` (scene.add_heat_lamp)."""
        self.lamp = lamp
        layer = Layer(self.bk, ps, self.seed)
        t, idx, n = trace(self.bk, self.o, self.d, layer.prims, self.size)
        l, p = self._shade(layer, t, idx, n, self.d, self.o, True)
        spec = layer.prims["specular"][idx] > 0
        self.static = dict(layer=layer, t=t, idx=idx, n=n, l=l, p=p, spec=spec,
                           any_spec=bool(self.bk.to_numpy(spec).any()))
        return self

    def _specular_pass(self, dyn_layer):
        """Single mirror bounce for static specular pixels (TV / glass)."""
        xp, st = self.bk.xp, self.static
        m = st["spec"]
        d, n, p = self.d[m], st["n"][m], st["p"][m]
        r = d - 2.0 * (d * n).sum(-1, keepdims=True) * n
        o2 = p + 1e-3 * n
        t, idx, n2 = trace(self.bk, o2, r, st["layer"].prims, self.size)
        l2, _ = self._shade(st["layer"], t, idx, n2, r, o2, True)
        if dyn_layer is not None:
            td, idd, nd = trace(self.bk, o2, r, dyn_layer.prims, None)
            ld, _ = self._shade(dyn_layer, td, idd, nd, r, o2, False)
            l2 = xp.where(td < t, ld, l2)
        l1, _ = self._shade(st["layer"], st["t"][m], st["idx"][m], n, d, self.o, True, l_spec=l2)
        return m, l1

    def _fog_tau(self, t, fog, time_s):
        xp, bk = self.bk.xp, self.bk
        od = fog.get("beta_per_m", 0.0) * t
        pl = fog.get("plume")
        if pl:
            c = bk.arr(np.asarray(pl["center"], float))
            sig = float(pl["sigma_m"])
            oc = self.o - c
            b = (self.d * oc).sum(-1)
            perp2 = (oc * oc).sum() - b * b
            k = sig * np.sqrt(np.pi / 2.0)
            od = od + pl["beta_peak_per_m"] * xp.exp(-perp2 / (2 * sig * sig)) * k * (
                bk.erf((t + b) / (sig * np.sqrt(2.0))) - bk.erf(b / (sig * np.sqrt(2.0))))
        turb = fog.get("turbulence", 0.0)
        if turb > 0:
            od = od * bk.arr(np.exp(turb * fog_noise(self.h * self.ss, self.w * self.ss, time_s,
                                                     fog.get("seed", self.seed)) - 0.5 * turb ** 2).reshape(-1))
        return xp.exp(-od)

    def render(self, dynamic=None, fog=None, time_s=0.0, dyn_seed=None):
        """Returns dict: radiance (H*ss, W*ss) backend array, plus numpy label/inst/depth maps
        at supersampled resolution. `dynamic` is a PrimSet (people etc.) or None."""
        xp, st = self.bk.xp, self.static
        if st is None:
            raise RuntimeError("call set_static() first")
        l, t = st["l"], st["t"]
        label = st["layer"].prims["label"][st["idx"]]
        inst = st["layer"].prims["inst"][st["idx"]]
        dyn_layer = None
        if dynamic is not None and len(dynamic):
            dyn_layer = Layer(self.bk, dynamic, self.seed + 1 if dyn_seed is None else dyn_seed)
        if st["any_spec"]:
            m, l1 = self._specular_pass(dyn_layer)
            l = l.clone() if self.bk.kind != "numpy" else l.copy()
            l[m] = l1
        if dyn_layer is not None:
            td, idd, nd = trace(self.bk, self.o, self.d, dyn_layer.prims, None)
            ld, _ = self._shade(dyn_layer, td, idd, nd, self.d, self.o, False)
            front = td < t
            safe = xp.where(idd >= 0, idd, xp.zeros_like(idd))
            l = xp.where(front, ld, l)
            t = xp.where(front, td, t)
            label = xp.where(front, dyn_layer.prims["label"][safe], label)
            inst = xp.where(front, dyn_layer.prims["inst"][safe], inst)
        tau = None
        if fog and (fog.get("beta_per_m", 0.0) > 0 or fog.get("plume")):
            tau = self._fog_tau(t, fog, time_s)
            l_mist = self.B_scalar(self.air_c + fog.get("mist_offset_k", 0.0))
            l = tau * l + (1.0 - tau) * l_mist
        shp = (self.h * self.ss, self.w * self.ss)
        out = dict(radiance=l.reshape(shp), label=self.bk.to_numpy(label).reshape(shp),
                   inst=self.bk.to_numpy(inst).reshape(shp), depth=self.bk.to_numpy(t).reshape(shp))
        if tau is not None:
            out["tau_mean"] = float(self.bk.to_numpy(tau).mean())
        return out

    def person_boxes(self, out, min_pixels=6):
        """Bounding boxes [x0,y0,x1,y1] in native pixels for each visible person instance."""
        boxes = []
        inst = out["inst"]
        for k in np.unique(inst[inst >= 0]):
            ys, xs = np.nonzero(inst == k)
            if len(ys) < min_pixels * self.ss ** 2:
                continue
            boxes.append(dict(inst=int(k), xyxy=[float(xs.min()) / self.ss, float(ys.min()) / self.ss,
                                                 float(xs.max() + 1) / self.ss, float(ys.max() + 1) / self.ss],
                              visible_px=float(len(ys)) / self.ss ** 2))
        return boxes


def fog_noise(h, w, time_s, seed=0, n_comp=8):
    """Smooth unit-variance screen-space field that drifts upward with time (mist turbulence proxy)."""
    rng = np.random.default_rng(seed + 7919)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scale = h / 120.0
    f = np.zeros((h, w), np.float32)
    for _ in range(n_comp):
        wl = np.exp(rng.uniform(np.log(12), np.log(90))) * scale
        ang = rng.uniform(0, 2 * np.pi)
        kx, ky = 2 * np.pi / wl * np.cos(ang), 2 * np.pi / wl * np.sin(ang)
        speed = rng.uniform(4, 16) * scale            # px / s, upward
        ph = rng.uniform(0, 2 * np.pi)
        wob = rng.uniform(0.3, 1.2)
        f += np.sin(kx * xx + ky * (yy + speed * time_s) + ph + np.sin(wob * time_s + ph))
    return f / np.sqrt(n_comp / 2.0)


def build_static(room, extra=None, skip_ids=(), include_objects=True):
    from .scene import static_prims
    ps = static_prims(room, include_objects=include_objects, skip_ids=skip_ids)
    if extra is not None:
        ps.extend(extra)
    return ps


__all__ = ["Renderer", "PrimSet", "fog_noise", "build_static", "INF", "LAMP"]
