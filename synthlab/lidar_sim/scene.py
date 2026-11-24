"""Room scene loading and vectorised ray casting (numpy).

Scene schema = scene/room_v0.json (axis-aligned room, box objects, sensors, dynamic_slots).
Any file with that schema loads; `default_scene_path()` picks the newest scene/room_v*.json.
Extra primitives (people = vertical capsule + head sphere, clutter boxes) are added in code.
All geometry is in WORLD coordinates (m, z up); `Scene.to_sensor` converts to the Mid-360 frame
(x = forward, y = left, z = up).
"""
import glob
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_scene_path():
    """$SYNTHLAB_SCENE, else the newest room_v*.json in scene/ or data_local/scene/ (the latter
    holds a local copy of the room-3d track's file until that branch is merged)."""
    if os.environ.get("SYNTHLAB_SCENE"):
        return os.environ["SYNTHLAB_SCENE"]
    files = (glob.glob(os.path.join(REPO, "scene", "room_v*.json"))
             + glob.glob(os.path.join(REPO, "data_local", "scene", "room_v*.json")))
    if not files:
        raise FileNotFoundError("no scene/room_v*.json")
    return max(files, key=lambda p: int(re.search(r"room_v(\d+)", p).group(1)))


@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray
    refl: float
    name: str = "box"


@dataclass
class Capsule:
    """Vertical capsule: axis from (x, y, z0) to (x, y, z1), radius r (spherical caps)."""
    xy: np.ndarray
    z0: float
    z1: float
    r: float
    refl: float
    name: str = "capsule"


@dataclass
class Sphere:
    center: np.ndarray
    r: float
    refl: float
    name: str = "sphere"


def person(xy, height=1.72, refl=0.30, name="person"):
    """Standing person = torso/legs capsule + head sphere (very coarse body model)."""
    xy = np.asarray(xy, float)[:2]
    r = 0.17
    body = Capsule(xy, r, height - 0.30 - r + 0.05, r, refl, name)
    head = Sphere(np.array([xy[0], xy[1], height - 0.11]), 0.105, 0.40, name + "_head")
    return [body, head]


@dataclass
class Scene:
    size: np.ndarray                      # room interior (3,)
    wall_refl: dict                       # floor/ceiling/wall_* -> reflectivity
    boxes: list = field(default_factory=list)
    capsules: list = field(default_factory=list)
    spheres: list = field(default_factory=list)
    sensor_pos: np.ndarray = None
    sensor_fwd: np.ndarray = None
    slots: dict = field(default_factory=dict)
    path: str = ""
    apertures: list = field(default_factory=list)   # (axis, side, lo(3), hi(3), kind)

    @classmethod
    def load(cls, path=None):
        path = path or default_scene_path()
        with open(path) as f:
            j = json.load(f)
        room = j["room"]
        refl = {k: float(v.get("lidar_reflectivity", 0.5)) for k, v in room["surfaces"].items()}
        objs = {o.get("id", str(i)): o for i, o in enumerate(j.get("objects", []))}
        boxes, apertures = [], []
        wall_axis = {"wall_west": (0, 0), "wall_east": (0, 1), "wall_south": (1, 0),
                     "wall_north": (1, 1), "floor": (2, 0), "ceiling": (2, 1)}
        ap_ids = {}
        for wname, surf in room["surfaces"].items():
            for oid in surf.get("openings", []):
                ap_ids[oid] = (wname, "open")
            if surf.get("glazing_object"):
                ap_ids[surf["glazing_object"]] = (wname, "glass")
        for oid, o in objs.items():
            if o.get("shape", "box") != "box":
                continue
            c, sz = np.array(o["center"], float), np.array(o["size"], float)
            if oid in ap_ids:                       # hole / glazing in a wall: rays escape
                ax, side = wall_axis[ap_ids[oid][0]]
                apertures.append((ax, side, c - sz / 2, c + sz / 2, ap_ids[oid][1]))
            elif o.get("solid", True) and not o.get("lidar_transmission", 0) > 0.5:
                boxes.append(Box(c, sz, float(o.get("lidar_reflectivity", 0.5)), oid))
        m = j["sensors"]["mid360"]
        return cls(np.array(room["size_xyz"], float), refl, boxes, [], [],
                   np.array(m["position"], float), np.array(m["forward"], float),
                   j.get("dynamic_slots", {}), path, apertures)

    def copy(self):
        return Scene(self.size.copy(), dict(self.wall_refl), list(self.boxes), list(self.capsules),
                     list(self.spheres), self.sensor_pos.copy(), self.sensor_fwd.copy(),
                     self.slots, self.path, list(self.apertures))

    def add(self, prims):
        for p in prims:
            {Box: self.boxes, Capsule: self.capsules, Sphere: self.spheres}[type(p)].append(p)
        return self

    # ---- frames -------------------------------------------------------------
    def rot_world_to_sensor(self):
        x = self.sensor_fwd / np.linalg.norm(self.sensor_fwd)
        z = np.array([0.0, 0.0, 1.0])
        y = np.cross(z, x)
        return np.stack([x, y, z])        # rows = sensor axes in world coords

    def to_sensor(self, pts_world):
        return (pts_world - self.sensor_pos) @ self.rot_world_to_sensor().T

    def dirs_to_world(self, d_sensor):
        return d_sensor @ self.rot_world_to_sensor()

    # ---- ray casting ----------------------------------------------------------
    def raycast(self, d, origin=None):
        """d: (N,3) unit directions in WORLD frame from `origin` (default sensor).
        Returns range t (N,), reflectivity (N,), cos incidence (N,), primitive id (N,) int
        (0 = room shell, then boxes, capsules, spheres in order)."""
        o = self.sensor_pos if origin is None else origin
        n = d.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            # room shell: ray starts inside, exits through one face
            tx = np.where(d > 0, (self.size - o) * inv, (0.0 - o) * inv)
            tx = np.where(d == 0, np.inf, tx)
            ax = np.argmin(tx, axis=1)
            t = tx[np.arange(n), ax]
            cosi = np.abs(d[np.arange(n), ax])
            names = [["wall_west", "wall_east"], ["wall_south", "wall_north"], ["floor", "ceiling"]]
            lut = np.array([[self.wall_refl.get(names[a][s], 0.5) for s in (0, 1)] for a in range(3)])
            refl = lut[ax, (d[np.arange(n), ax] > 0).astype(int)]
            pid = np.zeros(n, np.int32)
            # wall apertures (open door, glazing): the ray leaves the room -> no hard return
            # (refl 0; range = distance to the aperture so fog is only integrated inside the
            # room). Glass gives a specular return only within ~6 deg of normal incidence.
            for a_ax, a_side, lo, hi, kind in self.apertures:
                hitp = o + d * t[:, None]
                on = (ax == a_ax) & ((d[np.arange(n), ax] > 0).astype(int) == a_side)
                for b_ax in range(3):
                    if b_ax != a_ax:
                        on &= (hitp[:, b_ax] >= lo[b_ax]) & (hitp[:, b_ax] <= hi[b_ax])
                if kind == "glass":
                    refl = np.where(on, np.where(cosi > 0.995, 1.0, 0.0), refl)
                else:
                    refl = np.where(on, 0.0, refl)
            k = 0
            for b in self.boxes:
                k += 1
                lo, hi = b.center - b.size / 2, b.center + b.size / 2
                t1, t2 = (lo - o) * inv, (hi - o) * inv
                tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
                tn = np.where(np.isnan(tn), -np.inf, tn)
                tf = np.where(np.isnan(tf), np.inf, tf)
                a_in = np.argmax(tn, axis=1)
                tnear, tfar = tn[np.arange(n), a_in], tf.min(axis=1)
                hit = (tnear < tfar) & (tnear > 1e-6) & (tnear < t)
                t = np.where(hit, tnear, t)
                cosi = np.where(hit, np.abs(d[np.arange(n), a_in]), cosi)
                refl = np.where(hit, b.refl, refl)
                pid = np.where(hit, k, pid)
        t, cosi, refl = t.copy(), cosi.copy(), refl.copy()
        for c in self.capsules:
            k += 1
            sub = np.flatnonzero(_cone_xy(o, d, c.xy, c.r))      # cheap azimuth prefilter
            if sub.size:
                tc, cc = _ray_capsule(o, d[sub], c)
                hit = tc < t[sub]
                h = sub[hit]
                t[h], cosi[h], refl[h], pid[h] = tc[hit], cc[hit], c.refl, k
        for s in self.spheres:
            k += 1
            sub = np.flatnonzero(_cone_xy(o, d, s.center[:2], s.r))
            if sub.size:
                ts, cs = _ray_sphere(o, d[sub], s.center, s.r)
                hit = ts < t[sub]
                h = sub[hit]
                t[h], cosi[h], refl[h], pid[h] = ts[hit], cs[hit], s.refl, k
        return t, refl, np.clip(cosi, 0.0, 1.0), pid

    def prim_names(self):
        return (["room"] + [b.name for b in self.boxes] + [c.name for c in self.capsules]
                + [s.name for s in self.spheres])


def _cone_xy(o, d, cxy, r):
    """Rays whose horizontal bearing can touch a vertical cylinder of radius r at cxy."""
    v = cxy - o[:2]
    dist = float(np.hypot(v[0], v[1]))
    if dist <= r * 1.05:
        return np.ones(d.shape[0], bool)
    half = np.arcsin(min(1.0, r * 1.05 / dist))
    dn = np.hypot(d[:, 0], d[:, 1])
    cosang = (d[:, 0] * v[0] + d[:, 1] * v[1]) / (np.maximum(dn, 1e-9) * dist)
    return cosang >= np.cos(half)


def _ray_sphere(o, d, c, r):
    oc = o - c
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t = -b - sq
    ok &= t > 1e-6
    t = np.where(ok, t, np.inf)
    p = o + d * np.where(ok, t, 0.0)[:, None]
    nrm = (p - c) / r
    return t, np.abs(np.einsum("ij,ij->i", nrm, d))


def _ray_capsule(o, d, c):
    """Vertical capsule = infinite-cylinder hit clipped to [z0, z1] + two end spheres."""
    oxy = o[:2] - c.xy
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = d[:, 0] * oxy[0] + d[:, 1] * oxy[1]
    cc = oxy @ oxy - c.r ** 2
    disc = b * b - a * cc
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t = np.where(ok, (-b - sq) / np.where(a > 1e-12, a, 1.0), np.inf)
    z = o[2] + d[:, 2] * np.where(np.isfinite(t), t, 0.0)
    ok &= (t > 1e-6) & (z >= c.z0) & (z <= c.z1)
    t = np.where(ok, t, np.inf)
    p = o + d * np.where(ok, t, 0.0)[:, None]
    nxy = (p[:, :2] - c.xy) / c.r
    cosi = np.abs(nxy[:, 0] * d[:, 0] + nxy[:, 1] * d[:, 1])
    for zc in (c.z0, c.z1):
        ts, cs = _ray_sphere(o, d, np.array([c.xy[0], c.xy[1], zc]), c.r)
        better = ts < t
        t, cosi = np.where(better, ts, t), np.where(better, cs, cosi)
    return t, cosi
