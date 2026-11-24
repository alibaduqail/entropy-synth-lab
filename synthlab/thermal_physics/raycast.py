"""Vectorised ray caster: room interior + oriented boxes + ellipsoids + capsules.

Works on numpy arrays or torch tensors (CPU or CUDA) through the tiny `Backend`
shim: only operations with identical semantics in both libraries are used.
Rays share one origin (the camera). Output per ray: hit distance, primitive
index (into the PrimSet's property tables) and the surface normal.
"""
import numpy as np

INF = 1e9
EPS = 1e-9

# labels
ROOM, FURNITURE, SKIN, CLOTH, HAIR, WARM, LAMP = 0, 1, 2, 3, 4, 5, 6
LABEL_NAMES = ["room", "furniture", "skin", "clothing", "hair", "warm_object", "lamp"]


class Backend:
    def __init__(self, kind="numpy", device=None, dtype="float32"):
        self.kind = kind
        if kind == "numpy":
            self.xp = np
            self.dtype = np.dtype(dtype)
            self.device = "cpu"
        else:
            import torch
            self.xp = torch
            self.dtype = getattr(torch, dtype)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def arr(self, a, dtype=None):
        if self.kind == "numpy":
            return np.asarray(a, dtype=dtype or self.dtype)
        import torch
        if dtype in ("int", np.int64):
            dtype = torch.long
        return torch.as_tensor(np.asarray(a), dtype=dtype or self.dtype, device=self.device)

    def ints(self, a):
        return self.arr(a, dtype=np.int64 if self.kind == "numpy" else "int")

    def to_numpy(self, a):
        return a if self.kind == "numpy" else a.detach().cpu().numpy()

    def min_arg(self, t):
        """(min, argmin) along axis 1."""
        if self.kind == "numpy":
            i = np.argmin(t, axis=1)
            return t[np.arange(t.shape[0]), i], i
        v, i = self.xp.min(t, dim=1)
        return v, i

    def erf(self, x):
        if self.kind == "numpy":
            from scipy.special import erf
            return erf(x)
        return self.xp.erf(x)

    def sync(self):
        if self.kind != "numpy" and str(self.device).startswith("cuda"):
            self.xp.cuda.synchronize()


def camera_rays(cam, supersample=1):
    """Unit ray directions (H*ss*W*ss, 3) for a pinhole camera, row-major, numpy float64."""
    w, h = int(cam["width"]) * supersample, int(cam["height"]) * supersample
    f = (w / 2.0) / np.tan(np.radians(cam["hfov_deg"]) / 2.0)   # square pixels
    fwd = np.asarray(cam["look_dir"], dtype=np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u = (np.arange(w) + 0.5 - w / 2.0) / f
    v = -(np.arange(h) + 0.5 - h / 2.0) / f
    uu, vv = np.meshgrid(u, v)
    d = fwd[None, None, :] + uu[..., None] * right + vv[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


def _dot(a, b):
    return (a * b).sum(-1)


def trace_room(bk, o, d, size):
    """Exit point of the room AABB [0,size] seen from inside. Surface index =
    axis*2 + (1 if the +side) : x-,x+ = west,east; y-,y+ = south,north; z-,z+ = floor,ceiling."""
    xp = bk.xp
    pos = d > 0
    dd = xp.where(xp.abs(d) < EPS, xp.ones_like(d) * EPS, d)
    plane = xp.where(pos, xp.ones_like(d) * size, xp.zeros_like(d))
    t = (plane - o.reshape(-1, 3)) / dd
    t = xp.where(t <= 0, xp.ones_like(t) * INF, t)
    tmin, axis = bk.min_arg(t)
    side = xp.gather(pos.long(), 1, axis[:, None])[:, 0] if bk.kind != "numpy" else pos[np.arange(len(axis)), axis].astype(np.int64)
    surf = axis * 2 + side
    # inward normal: -sign along axis
    eye = bk.arr(np.eye(3))
    n = eye[axis] * (1.0 - 2.0 * side[:, None])
    return tmin, surf, n


def trace_boxes(bk, o, d, center, half, yaw):
    """Oriented (yaw about z) boxes. Returns t (P,B) and local-entry normals (P,B,3) in world frame."""
    xp = bk.xp
    c, s = xp.cos(yaw), xp.sin(yaw)
    rel = o.reshape(-1, 1, 3) - center[None]                    # (Po,B,3), Po = 1 or P
    ox = c * rel[..., 0] + s * rel[..., 1]
    oy = -s * rel[..., 0] + c * rel[..., 1]
    ol = xp.stack([ox, oy, rel[..., 2]], -1)                    # (Po,B,3)
    dx = d[:, 0:1] * c[None] + d[:, 1:2] * s[None]
    dy = -d[:, 0:1] * s[None] + d[:, 1:2] * c[None]
    dz = d[:, 2:3] + 0 * dx
    dl = xp.stack([dx, dy, dz], -1)                             # (P,B,3)
    dl = xp.where(xp.abs(dl) < EPS, xp.ones_like(dl) * EPS, dl)
    t1 = (-half[None] - ol) / dl
    t2 = (half[None] - ol) / dl
    tlo = xp.minimum(t1, t2)
    thi = xp.maximum(t1, t2)
    tn = tlo.max(-1) if bk.kind == "numpy" else tlo.max(-1).values
    tf = thi.min(-1) if bk.kind == "numpy" else thi.min(-1).values
    hit = (tn < tf) & (tn > 1e-4)
    t = xp.where(hit, tn, xp.ones_like(tn) * INF)
    ax = tlo.argmax(-1)                                         # (P,B)
    eye = bk.arr(np.eye(3))
    nl = eye[ax] * (-xp.sign(dl))                               # entering face
    nx = nl[..., 0] * c[None] - nl[..., 1] * s[None]
    ny = nl[..., 0] * s[None] + nl[..., 1] * c[None]
    return t, xp.stack([nx, ny, nl[..., 2]], -1)


def trace_ellipsoids(bk, o, d, center, radii, rot):
    """rot: (E,3,3) columns = local axes in world frame."""
    xp = bk.xp
    rel = o.reshape(-1, 1, 3) - center[None]                    # (Po,E,3)
    ol = xp.einsum("eji,pej->pei", rot, rel) / radii[None]
    dl = xp.einsum("eji,pj->pei", rot, d) / radii[None]
    a = _dot(dl, dl)
    b = _dot(dl, ol)
    cc = _dot(ol, ol) - 1.0
    disc = b * b - a * cc
    ok = disc > 0
    sq = xp.sqrt(xp.where(ok, disc, xp.zeros_like(disc)))
    t = (-b - sq) / a
    hit = ok & (t > 1e-4)
    t = xp.where(hit, t, xp.ones_like(t) * INF)
    return t, (ol, dl)


def ellipsoid_normal(bk, aux, t_sel, idx, radii, rot, rows):
    ol, dl = aux
    ol_sel = ol[rows, idx] if ol.shape[0] == dl.shape[0] else ol[0, idx]
    pl = ol_sel + t_sel[:, None] * dl[rows, idx]                # local unit-sphere point
    g = pl / radii[idx]
    r = rot[idx]
    n = (r @ g[..., None])[..., 0]
    return n / (bk.xp.sqrt(_dot(n, n))[:, None] + EPS)


def trace_capsules(bk, o, d, a, b, r):
    """Capsules with end points a,b (C,3) and radius r (C,). After I. Quilez' capIntersect."""
    xp = bk.xp
    ba = (b - a)[None]                                          # (1,C,3)
    oa = o.reshape(-1, 1, 3) - a[None]                          # (Po,C,3)
    dd = d[:, None, :]                                          # (P,1,3)
    baba = _dot(ba, ba)
    bard = _dot(ba, dd)
    baoa = _dot(ba, oa)
    rdoa = _dot(dd, oa)
    oaoa = _dot(oa, oa)
    r2 = (r * r)[None]
    aa = baba - bard * bard
    bb = baba * rdoa - baoa * bard
    cc = baba * oaoa - baoa * baoa - r2 * baba
    h = bb * bb - aa * cc
    ok = h >= 0
    sq = xp.sqrt(xp.where(ok, h, xp.zeros_like(h)))
    aa_s = xp.where(xp.abs(aa) < EPS, xp.ones_like(aa) * EPS, aa)
    t = (-bb - sq) / aa_s
    y = baoa + t * bard
    body = ok & (y > 0) & (y < baba)
    # caps
    use_b = y > 0
    oc = xp.where(use_b[..., None], oa - ba + 0 * dd, oa + 0 * dd)
    b2 = _dot(dd, oc)
    c2 = _dot(oc, oc) - r2
    h2 = b2 * b2 - c2
    okc = ok & (h2 > 0)
    tc = -b2 - xp.sqrt(xp.where(okc, h2, xp.zeros_like(h2)))
    tt = xp.where(body, t, xp.where(okc, tc, xp.ones_like(t) * INF))
    tt = xp.where(tt > 1e-4, tt, xp.ones_like(tt) * INF)
    return tt


def capsule_normal(bk, p, a, b, r):
    xp = bk.xp
    ba = b - a
    pa = p - a
    h = (_dot(pa, ba) / _dot(ba, ba)).clip(0.0, 1.0)
    return (pa - h[:, None] * ba) / r[:, None]


def trace(bk, o, d, prims, room_size=None):
    """Closest hit over a primitive set. `prims` is a dict of backend arrays
    (see scene.PrimSet.to_backend). Returns t (P,), prim index (P,), normal (P,3).
    Rays that hit nothing get t=INF and prim=-1."""
    xp = bk.xp
    p_count = d.shape[0]
    rows = xp.arange(p_count) if bk.kind == "numpy" else xp.arange(p_count, device=d.device)
    best_t = xp.ones_like(d[:, 0]) * INF
    best_i = bk.ints(np.full(p_count, -1))
    best_n = xp.zeros_like(d)
    if room_size is not None:
        t, surf, n = trace_room(bk, o, d, room_size)
        best_t, best_i, best_n = t, surf, n
    off = prims["offsets"]
    if prims["n_box"]:
        t, n = trace_boxes(bk, o, d, prims["box_c"], prims["box_h"], prims["box_yaw"])
        tm, im = bk.min_arg(t)
        upd = tm < best_t
        best_t = xp.where(upd, tm, best_t)
        best_i = xp.where(upd, im + off[0], best_i)
        best_n = xp.where(upd[:, None], n[rows, im], best_n)
    if prims["n_ell"]:
        t, aux = trace_ellipsoids(bk, o, d, prims["ell_c"], prims["ell_r"], prims["ell_rot"])
        tm, im = bk.min_arg(t)
        upd = tm < best_t
        tsel = xp.where(upd, tm, xp.zeros_like(tm))
        n = ellipsoid_normal(bk, aux, tsel, im, prims["ell_r"], prims["ell_rot"], rows)
        best_t = xp.where(upd, tm, best_t)
        best_i = xp.where(upd, im + off[1], best_i)
        best_n = xp.where(upd[:, None], n, best_n)
    if prims["n_cap"]:
        t = trace_capsules(bk, o, d, prims["cap_a"], prims["cap_b"], prims["cap_r"])
        tm, im = bk.min_arg(t)
        upd = tm < best_t
        tsel = xp.where(upd, tm, xp.zeros_like(tm))
        p = o.reshape(-1, 3) + tsel[:, None] * d
        n = capsule_normal(bk, p, prims["cap_a"][im], prims["cap_b"][im], prims["cap_r"][im])
        best_t = xp.where(upd, tm, best_t)
        best_i = xp.where(upd, im + off[2], best_i)
        best_n = xp.where(upd[:, None], n, best_n)
    return best_t, best_i, best_n
