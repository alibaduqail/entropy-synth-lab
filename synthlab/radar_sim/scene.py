"""Turn a room scene file (schema of scene/room_v0.json) into radar point scatterers.

Model (all reflectivities are ASSUMED order-of-magnitude values for 60 GHz):
- every room plane gives one specular point at the foot of the perpendicular from the
  radar. By image theory a large flat reflector at distance R returns the same power as a
  point target of RCS  sigma = pi R^2 |Gamma|^2  (|Gamma|^2 = power reflectivity).
- wall-wall and wall-floor/ceiling edges act as dihedrals (two bounces, |Gamma|^4), room
  corners as trihedrals (|Gamma|^6), each seen at the range of the edge/corner point.
- box objects: a specular point per face that faces the radar (if the perpendicular foot
  lies on the face), plus a few seeded diffuse points per visible face.
- rough-surface diffuse points on the walls.
Not modelled: multipath ghosts, diffraction, occlusion between objects.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# power reflectivity |Gamma|^2 at 60 GHz, near-normal incidence (ASSUMED)
MATERIAL_REFLECTIVITY = {
    "painted_drywall": 0.16, "carpet": 0.03, "acoustic_tile": 0.06, "glass": 0.35,
    "lcd": 0.35, "laminate_wood": 0.15, "wood": 0.15, "fabric": 0.04, "plastic": 0.08,
    "electronics": 0.5, "cardboard": 0.05, "metal": 1.0, "alumin": 1.0, "steel": 1.0,
    "enclosure": 0.3, "housing": 0.2,
}
DEFAULT_REFLECTIVITY = 0.10
DIFFUSE_SIGMA0 = 0.01        # m^2 of RCS per m^2 of rough surface (-20 dB), ASSUMED


def reflectivity(material: str) -> float:
    for key, val in MATERIAL_REFLECTIVITY.items():
        if key in material:
            return val
    return DEFAULT_REFLECTIVITY


def default_scene_path(scene_dir: Optional[str] = None) -> str:
    """Newest ``room_v*.json`` in the scene directory."""
    if scene_dir is None:
        scene_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scene")
    files = sorted(glob.glob(os.path.join(scene_dir, "room_v*.json")))
    if not files:
        raise FileNotFoundError("no room_v*.json in %s" % scene_dir)
    return os.path.abspath(files[-1])


def load_scene(path: Optional[str] = None) -> dict:
    path = path or default_scene_path()
    with open(path) as f:
        scene = json.load(f)
    scene["_path"] = path
    return scene


@dataclass
class Scatterers:
    pos: np.ndarray                 # (P,3) room frame, metres
    rcs: np.ndarray                 # (P,) m^2
    labels: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rcs)

    @staticmethod
    def concat(items: Sequence["Scatterers"]) -> "Scatterers":
        items = [s for s in items if len(s)]
        if not items:
            return Scatterers(np.zeros((0, 3)), np.zeros(0), [])
        return Scatterers(np.concatenate([s.pos for s in items]),
                          np.concatenate([s.rcs for s in items]),
                          [l for s in items for l in s.labels])


@dataclass
class RadarPose:
    position: np.ndarray            # room frame
    rot: np.ndarray                 # rows = radar x (azimuth), y (boresight), z (up) in room

    def to_radar(self, p_room: np.ndarray) -> np.ndarray:
        return (np.asarray(p_room) - self.position) @ self.rot.T

    def to_room(self, p_radar: np.ndarray) -> np.ndarray:
        return np.asarray(p_radar) @ self.rot + self.position


def radar_pose(scene: dict) -> RadarPose:
    s = scene["sensors"]["iwr6843isk"]
    pos = np.array(s["position"], dtype=float)
    y = np.array(s["boresight"], dtype=float)
    y /= np.linalg.norm(y)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, up)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return RadarPose(pos, np.stack([x, y, z]))


def _room_planes(scene: dict):
    sx, sy, sz = scene["room"]["size_xyz"]
    surf = scene["room"]["surfaces"]
    # (name, axis, coordinate)
    planes = [("wall_west", 0, 0.0), ("wall_east", 0, sx), ("wall_south", 1, 0.0),
              ("wall_north", 1, sy), ("floor", 2, 0.0), ("ceiling", 2, sz)]
    return [(n, a, c, reflectivity(surf.get(n, {}).get("material", ""))) for n, a, c in planes]


def room_scatterers(scene: dict, n_diffuse: int = 40, seed: int = 0) -> Scatterers:
    pose = radar_pose(scene)
    r0 = pose.position
    planes = _room_planes(scene)
    pos, rcs, lab = [], [], []
    # specular planes
    for name, ax, c, g in planes:
        p = r0.copy()
        p[ax] = c
        dist = abs(r0[ax] - c)
        pos.append(p); rcs.append(np.pi * dist ** 2 * g); lab.append("spec:" + name)
    # dihedral edges (pairs of planes on different axes)
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            (n1, a1, c1, g1), (n2, a2, c2, g2) = planes[i], planes[j]
            if a1 == a2:
                continue
            p = r0.copy()
            p[a1] = c1; p[a2] = c2
            dist = np.linalg.norm(p - r0)
            pos.append(p); rcs.append(np.pi * dist ** 2 * g1 * g2)
            lab.append("edge:%s+%s" % (n1, n2))
    # trihedral corners
    xs = [pl for pl in planes if pl[1] == 0]
    ys = [pl for pl in planes if pl[1] == 1]
    zs = [pl for pl in planes if pl[1] == 2]
    for px in xs:
        for py in ys:
            for pz in zs:
                p = np.array([px[2], py[2], pz[2]])
                dist = np.linalg.norm(p - r0)
                pos.append(p); rcs.append(np.pi * dist ** 2 * px[3] * py[3] * pz[3])
                lab.append("corner:%s+%s+%s" % (px[0], py[0], pz[0]))
    # diffuse wall roughness (4 walls only), seeded
    rng = np.random.default_rng(seed)
    sx, sy, sz = scene["room"]["size_xyz"]
    walls = [pl for pl in planes if pl[1] != 2]
    for k in range(n_diffuse):
        name, ax, c, g = walls[k % len(walls)]
        p = np.array([rng.uniform(0, sx), rng.uniform(0, sy), rng.uniform(0, sz)])
        p[ax] = c
        other = sy if ax == 0 else sx
        area = other * sz
        pos.append(p); rcs.append(DIFFUSE_SIGMA0 * area / max(1, n_diffuse // len(walls)))
        lab.append("diffuse:" + name)
    return Scatterers(np.array(pos), np.array(rcs), lab)


def box_scatterers(obj: dict, radar_pos: np.ndarray, wavelength: float,
                   rng: np.random.Generator, diffuse_per_m2: float = 6.0) -> Scatterers:
    ctr = np.array(obj["center"], dtype=float)
    half = np.array(obj["size"], dtype=float) / 2.0
    g = obj.get("radar_reflectivity", reflectivity(obj.get("material", "")))
    pos, rcs, lab = [], [], []
    for ax in range(3):
        for sign in (-1.0, 1.0):
            face_c = ctr.copy()
            face_c[ax] += sign * half[ax]
            if (radar_pos[ax] - face_c[ax]) * sign <= 0:
                continue                      # face points away from the radar
            others = [a for a in range(3) if a != ax]
            area = 4.0 * half[others[0]] * half[others[1]]
            foot = radar_pos.copy()
            foot[ax] = face_c[ax]
            inside = all(abs(foot[a] - ctr[a]) <= half[a] for a in others)
            dist = abs(radar_pos[ax] - face_c[ax])
            if inside and dist > 0.05:
                s = min(np.pi * dist ** 2 * g, 4 * np.pi * area ** 2 / wavelength ** 2 * g)
                pos.append(foot); rcs.append(s); lab.append("spec:" + obj["id"])
            n = int(max(2, np.ceil(area * diffuse_per_m2)))
            for _ in range(n):
                p = face_c.copy()
                for a in others:
                    p[a] = ctr[a] + rng.uniform(-half[a], half[a])
                pos.append(p); rcs.append(DIFFUSE_SIGMA0 * (g / DEFAULT_REFLECTIVITY) * area / n)
                lab.append("diffuse:" + obj["id"])
    if not pos:
        return Scatterers(np.zeros((0, 3)), np.zeros(0), [])
    return Scatterers(np.array(pos), np.array(rcs), lab)


def static_scatterers(scene: dict, wavelength: float, extra_objects: Sequence[dict] = (),
                      seed: int = 0, n_diffuse_wall: int = 40) -> Scatterers:
    """All non-moving scatterers of the scene (+ optional extra clutter objects)."""
    rng = np.random.default_rng(seed + 1000)
    pose = radar_pose(scene)
    parts = [room_scatterers(scene, n_diffuse=n_diffuse_wall, seed=seed)]
    for obj in list(scene.get("objects", [])) + list(extra_objects):
        if obj.get("solid", True) is False or obj.get("material") == "air":
            continue                                  # openings (room_v1 schema)
        if obj.get("shape", "box") == "box":
            parts.append(box_scatterers(obj, pose.position, wavelength, rng))
        elif obj.get("shape") == "point":
            parts.append(Scatterers(np.array([obj["center"]], dtype=float),
                                    np.array([obj["rcs_m2"]], dtype=float), ["point:" + obj["id"]]))
    return Scatterers.concat(parts)


def shadow_factors(scat: Scatterers, radar_pos: np.ndarray, people_xy: Sequence[Sequence[float]],
                   body_radius: float = 0.20, body_height: float = 1.75,
                   atten_db: float = 20.0) -> np.ndarray:
    """Amplitude factor per static scatterer: attenuated if a person stands in the line of
    sight (vertical cylinder test, evaluated once at t=0)."""
    fac = np.ones(len(scat))
    if len(scat) == 0:
        return fac
    for xy in people_xy:
        c = np.array(xy[:2], dtype=float)
        a = radar_pos[:2][None, :]
        d = scat.pos[:, :2] - a
        seg2 = np.maximum((d ** 2).sum(axis=1), 1e-12)
        t = np.clip(((c[None, :] - a) * d).sum(axis=1) / seg2, 0.0, 1.0)
        closest = a + t[:, None] * d
        miss = np.linalg.norm(closest - c[None, :], axis=1)
        z_at = radar_pos[2] + t * (scat.pos[:, 2] - radar_pos[2])
        hit = (miss < body_radius) & (z_at < body_height) & (t < 0.98) & (t > 0.02)
        fac[hit] *= 10 ** (-atten_db / 20.0)
    return fac


def floor_xy(scene: dict, lateral_m: float, range_m: float, margin: float = 0.25) -> List[float]:
    """Room (x, y) of a floor point given in the radar's ground-plane frame
    (lateral = to the right of boresight, range = along boresight), clipped into the room."""
    pose = radar_pose(scene)
    fwd = pose.rot[1, :2] / np.linalg.norm(pose.rot[1, :2])
    right = np.array([fwd[1], -fwd[0]])
    if np.dot(right, pose.rot[0, :2]) < 0:
        right = -right
    p = pose.position[:2] + lateral_m * right + range_m * fwd
    sx, sy = scene["room"]["size_xyz"][:2]
    return [float(np.clip(p[0], margin, sx - margin)), float(np.clip(p[1], margin, sy - margin))]
