"""Scene description: room JSON loader, primitive sets, person and warm objects.

Units: metres, z up, origin at the south-west floor corner (scene/room_v*.json).
Every primitive carries: temp_c, emissivity, specular fraction, label, instance
id and a temperature-texture amplitude (K).
"""
import glob
import json
import os

import numpy as np

from .raycast import ROOM, FURNITURE, SKIN, CLOTH, HAIR, WARM, LAMP

SURFACE_ORDER = ["wall_west", "wall_east", "wall_south", "wall_north", "floor", "ceiling"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def newest_room_path(scene_dir=None):
    """Newest room_v*.json: $SYNTHLAB_SCENE if set, else the highest version found in this
    checkout's scene/ or in the sibling room-3d worktree (where v1 lands before it is merged)."""
    if os.environ.get("SYNTHLAB_SCENE"):
        return os.environ["SYNTHLAB_SCENE"]
    dirs = [scene_dir] if scene_dir else [os.path.join(REPO_ROOT, "scene"),
                                          os.path.join(REPO_ROOT, "..", "room-3d", "scene")]
    files = sorted((f for d in dirs for f in glob.glob(os.path.join(d, "room_v*.json"))),
                   key=lambda f: os.path.basename(f))
    if not files:
        raise FileNotFoundError("no scene/room_v*.json found")
    return files[-1]


def load_room(path=None):
    with open(path or newest_room_path()) as f:
        room = json.load(f)
    room["_path"] = path or newest_room_path()
    return room


class PrimSet:
    """Growable list of primitives. Index space: 0-5 room surfaces, then boxes,
    ellipsoids, capsules (in that order) - see raycast.trace."""

    def __init__(self):
        self.box, self.ell, self.cap = [], [], []
        self.room_props = [dict(temp_c=22.5, emissivity=0.92, specular=0.0, label=ROOM,
                                inst=-1, tex=0.0, name=s) for s in SURFACE_ORDER]

    @staticmethod
    def _props(temp_c, emissivity, specular, label, inst, tex, name):
        return dict(temp_c=float(temp_c), emissivity=float(emissivity), specular=float(specular),
                    label=int(label), inst=int(inst), tex=float(tex), name=name)

    def add_box(self, center, size, temp_c, emissivity=0.92, specular=0.0, label=FURNITURE,
                inst=-1, tex=0.08, yaw=0.0, name="box"):
        self.box.append((np.asarray(center, float), np.asarray(size, float) / 2.0, float(yaw),
                         self._props(temp_c, emissivity, specular, label, inst, tex, name)))

    def add_ellipsoid(self, center, radii, temp_c, emissivity=0.95, specular=0.0, label=WARM,
                      inst=-1, tex=0.2, rot=None, name="ell"):
        rot = np.eye(3) if rot is None else np.asarray(rot, float)
        self.ell.append((np.asarray(center, float), np.asarray(radii, float), rot,
                         self._props(temp_c, emissivity, specular, label, inst, tex, name)))

    def add_capsule(self, a, b, r, temp_c, emissivity=0.95, specular=0.0, label=WARM,
                    inst=-1, tex=0.2, name="cap"):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.allclose(a, b):
            b = b + np.array([0, 0, 1e-4])
        self.cap.append((a, b, float(r), self._props(temp_c, emissivity, specular, label, inst, tex, name)))

    def extend(self, other):
        self.box += other.box
        self.ell += other.ell
        self.cap += other.cap

    def all_props(self):
        return self.room_props + [b[3] for b in self.box] + [e[3] for e in self.ell] + [c[3] for c in self.cap]

    def __len__(self):
        return len(self.box) + len(self.ell) + len(self.cap)

    def to_backend(self, bk):
        props = self.all_props()
        nb, ne, nc = len(self.box), len(self.ell), len(self.cap)
        out = dict(n_box=nb, n_ell=ne, n_cap=nc, offsets=(6, 6 + nb, 6 + nb + ne))
        if nb:
            out["box_c"] = bk.arr(np.stack([b[0] for b in self.box]))
            out["box_h"] = bk.arr(np.stack([b[1] for b in self.box]))
            out["box_yaw"] = bk.arr(np.array([b[2] for b in self.box]))
        if ne:
            out["ell_c"] = bk.arr(np.stack([e[0] for e in self.ell]))
            out["ell_r"] = bk.arr(np.stack([e[1] for e in self.ell]))
            out["ell_rot"] = bk.arr(np.stack([e[2] for e in self.ell]))
        if nc:
            out["cap_a"] = bk.arr(np.stack([c[0] for c in self.cap]))
            out["cap_b"] = bk.arr(np.stack([c[1] for c in self.cap]))
            out["cap_r"] = bk.arr(np.array([c[2] for c in self.cap]))
        for key in ("temp_c", "emissivity", "specular", "tex"):
            out[key] = bk.arr(np.array([p[key] for p in props]))
        out["label"] = bk.ints(np.array([p["label"] for p in props]))
        out["inst"] = bk.ints(np.array([p["inst"] for p in props]))
        return out


def static_prims(room, include_objects=True, skip_ids=(), near=None, near_m=0.25, skip_detail=False):
    """Room surfaces + the JSON's static objects (all `shape: box`)."""
    ps = PrimSet()
    surfaces = room["room"]["surfaces"]
    for i, name in enumerate(SURFACE_ORDER):
        s = surfaces[name]
        ps.room_props[i].update(temp_c=s["temp_c"], emissivity=s["emissivity"],
                                specular=s.get("specular", 0.0), tex=s.get("tex", 0.06))
    if include_objects:
        for ob in room.get("objects", []):
            if ob["id"] in skip_ids or ob.get("solid", True) is False or ob.get("emissivity", 0.9) <= 0:
                continue
            if skip_detail and ob.get("detail"):
                continue
            if near is not None and "center" in ob and "size" in ob:    # the rig's own boxes
                gap = np.abs(np.asarray(near, float) - np.asarray(ob["center"], float)) - np.asarray(ob["size"], float) / 2
                if np.linalg.norm(np.clip(gap, 0, None)) < near_m:
                    continue
            warm = ob["temp_c"] > room["room"]["air_temp_c"] + 3.0
            kw = dict(temp_c=ob["temp_c"], emissivity=ob.get("emissivity", 0.92),
                      specular=ob.get("specular", 0.0), label=WARM if warm else FURNITURE,
                      tex=0.4 if warm else 0.08, name=ob["id"])
            shape = ob.get("shape", "box")
            if shape == "box":
                ps.add_box(ob["center"], ob["size"], yaw=np.radians(ob.get("yaw_deg", 0.0)), **kw)
            elif shape in ("ellipsoid", "sphere"):
                r = ob.get("radii", [ob.get("radius", 0.1)] * 3)
                ps.add_ellipsoid(ob["center"], r, **kw)
            elif shape in ("capsule", "cylinder"):
                c, h, r = np.asarray(ob["center"], float), ob.get("height", 0.2), ob.get("radius", 0.05)
                ps.add_capsule(c - [0, 0, h / 2], c + [0, 0, h / 2], r, **kw)
    return ps


def camera_from_room(room, **override):
    cam = dict(room["sensors"]["lepton35"])
    cam.update(override)
    return cam


# ----------------------------------------------------------------------------- person
POSES = ("standing", "walking", "sitting", "arms_raised")


def _rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _limb_dir(flex, abd=0.0, side=1.0):
    """Unit vector for a limb hanging down, flexed forward by `flex` rad and abducted by `abd`."""
    v = np.array([side * np.sin(abd), np.sin(flex) * np.cos(abd), -np.cos(flex) * np.cos(abd)])
    return v / np.linalg.norm(v)


def person_temps(air_c, rng=None, core_skin_c=33.8):
    """Apparent surface temperatures. Clothing sits between skin and air
    (ASSUMPTION: fraction k of the skin-air difference; torso 0.55, legs 0.40)."""
    j = (lambda s: rng.normal(0, s)) if rng is not None else (lambda s: 0.0)
    face = core_skin_c + j(0.5)
    return dict(face=face, neck=face - 0.3, forearm=face - 1.5 + j(0.6), hand=face - 2.5 + j(1.2),
                hair=air_c + 0.55 * (face - air_c) + j(0.7),
                torso=air_c + 0.55 * (face - air_c) + j(1.0),
                legs=air_c + 0.40 * (face - air_c) + j(0.8),
                shoes=air_c + 0.20 * (face - air_c))


def add_person(ps, pos_xy, yaw=0.0, pose="standing", phase=0.0, height=1.75, inst=0,
               air_c=22.5, temps=None, rng=None, lamp_in_hand=False, lamp_temp_c=180.0):
    """Append an articulated person (2 ellipsoids + ~17 capsules) to `ps`.

    yaw = 0 faces +y; person-local axes: x right, y forward, z up. `phase` in
    [0,1) drives the walking cycle / idle sway. Returns dict of key joints (world).
    """
    if pose not in POSES:
        raise ValueError(f"pose must be one of {POSES}")
    t = temps or person_temps(air_c, rng)
    s = height / 1.75
    rot = _rz(yaw)
    base = np.array([pos_xy[0], pos_xy[1], 0.0])
    w = 2 * np.pi * phase
    hip_h, sh_h = 0.92 * s, 1.45 * s
    thigh, shin, uarm, farm = 0.44 * s, 0.44 * s, 0.30 * s, 0.27 * s
    hip_flex = [0.0, 0.0]
    knee = [0.0, 0.0]
    sh_flex = [0.05, 0.05]
    sh_abd = [0.12, 0.12]
    elbow = [0.15, 0.15]
    lean = 0.0
    if pose == "walking":
        hip_flex = [0.45 * np.sin(w), -0.45 * np.sin(w)]
        knee = [0.5 * max(0.0, -np.cos(w)) + 0.1, 0.5 * max(0.0, np.cos(w)) + 0.1]
        sh_flex = [-0.4 * np.sin(w), 0.4 * np.sin(w)]
        elbow = [0.35, 0.35]
        hip_h -= 0.02 * s * abs(np.sin(w))
        sh_h -= 0.02 * s * abs(np.sin(w))
    elif pose == "sitting":
        drop = hip_h - 0.50 * s
        hip_h -= drop
        sh_h -= drop
        hip_flex = [1.45, 1.45]
        knee = [1.45, 1.45]
        sh_flex = [0.5, 0.5]
        elbow = [1.0, 1.0]
        lean = 0.05
    elif pose == "arms_raised":
        sh_flex = [2.75 + 0.1 * np.sin(w), 2.75 - 0.1 * np.sin(w)]
        sh_abd = [0.25, 0.25]
        elbow = [0.6, 0.6]
    sway = 0.01 * np.sin(w) if pose != "walking" else 0.0

    def W(p):
        return base + rot @ (np.asarray(p, float) + np.array([sway, 0, 0]))

    kw = dict(inst=inst)
    pelvis = np.array([0, 0, hip_h + 0.05 * s])
    chest = np.array([0, lean * 0.3, sh_h - 0.12 * s])
    ps.add_capsule(W(pelvis), W(chest), 0.145 * s, t["torso"], label=CLOTH, tex=0.7, name="torso", **kw)
    ps.add_capsule(W([-0.15 * s, 0, sh_h - 0.03 * s]), W([0.15 * s, 0, sh_h - 0.03 * s]), 0.075 * s,
                   t["torso"], label=CLOTH, tex=0.7, name="shoulders", **kw)
    ps.add_capsule(W([0, 0.01, sh_h]), W([0, 0.015, sh_h + 0.10 * s]), 0.052 * s, t["neck"],
                   emissivity=0.98, label=SKIN, tex=0.3, name="neck", **kw)
    head_c = np.array([0, 0.02, sh_h + 0.20 * s])
    ps.add_ellipsoid(W(head_c), np.array([0.078, 0.095, 0.115]) * s, t["face"], emissivity=0.98,
                     label=SKIN, tex=0.5, rot=rot, name="head", **kw)
    ps.add_ellipsoid(W(head_c + np.array([0, -0.022, 0.028]) * s), np.array([0.086, 0.098, 0.108]) * s,
                     t["hair"], emissivity=0.95, label=HAIR, tex=0.6, rot=rot, name="hair", **kw)
    joints = dict(head=W(head_c))
    for k, side in enumerate((1.0, -1.0)):
        hip = np.array([side * 0.09 * s, 0, hip_h])
        d1 = _limb_dir(hip_flex[k])
        kn = hip + thigh * d1
        d2 = _limb_dir(hip_flex[k] - knee[k])
        an = kn + shin * d2
        ps.add_capsule(W(hip), W(kn), 0.075 * s, t["legs"], label=CLOTH, tex=0.5, name="thigh", **kw)
        ps.add_capsule(W(kn), W(an), 0.052 * s, t["legs"] - 0.5, label=CLOTH, tex=0.5, name="shin", **kw)
        ps.add_capsule(W(an + [0, 0, 0.0]), W(an + [0, 0.14 * s, -0.02 * s]), 0.045 * s, t["shoes"],
                       label=CLOTH, tex=0.3, name="foot", **kw)
        sh = np.array([side * 0.20 * s, 0, sh_h - 0.03 * s])
        u = _limb_dir(sh_flex[k], sh_abd[k], side)
        el = sh + uarm * u
        f = _limb_dir(sh_flex[k] + elbow[k], sh_abd[k] * 0.3, side if pose != "arms_raised" else -side * 0.5)
        wr = el + farm * f
        ps.add_capsule(W(sh), W(el), 0.048 * s, t["torso"] + 0.8, label=CLOTH, tex=0.5, name="upper_arm", **kw)
        ps.add_capsule(W(el), W(wr), 0.038 * s, t["forearm"], emissivity=0.98, label=SKIN, tex=0.4,
                       name="forearm", **kw)
        hand = wr + 0.07 * s * f
        ps.add_ellipsoid(W(hand), np.array([0.042, 0.042, 0.055]) * s, t["hand"], emissivity=0.98,
                         label=SKIN, tex=0.5, rot=rot, name="hand", **kw)
        joints["hand_r" if side > 0 else "hand_l"] = W(hand)
    if lamp_in_hand:
        c = 0.5 * (joints["hand_r"] + joints["hand_l"]) + rot @ np.array([0, 0.05, 0.10 * s])
        add_heat_lamp(ps, c, aim=rot @ np.array([0.0, 0.6, -0.8]), temp_c=lamp_temp_c)
        joints["lamp"] = c
    return joints


# ----------------------------------------------------------------------------- warm objects
def add_heat_lamp(ps, position, aim=(0, 0, -1.0), temp_c=180.0, housing_c=55.0):
    """250 W-class reflector lamp: warm housing bowl with a hot bulb face.
    The bulb sphere pokes out of the housing along `aim`, so it is only visible from the front."""
    p = np.asarray(position, float)
    aim = np.asarray(aim, float)
    aim = aim / np.linalg.norm(aim)
    ps.add_ellipsoid(p, [0.085, 0.085, 0.085], housing_c, emissivity=0.80, label=LAMP, tex=3.0, name="lamp_housing")
    ps.add_ellipsoid(p + 0.045 * aim, [0.060, 0.060, 0.060], temp_c, emissivity=0.90, label=LAMP, tex=8.0,
                     name="lamp_bulb")


WARM_CATALOGUE = [
    # name, kind, dims (m), temp_c, emissivity
    ("laptop", "box", (0.33, 0.23, 0.02), 34.0, 0.90),
    ("mug", "capsule", (0.04, 0.09), 55.0, 0.93),
    ("monitor", "box", (0.52, 0.05, 0.32), 31.0, 0.90),
    ("router", "box", (0.18, 0.12, 0.04), 37.0, 0.92),
    ("charger", "box", (0.06, 0.05, 0.03), 40.0, 0.92),
    ("kettle", "capsule", (0.08, 0.18), 70.0, 0.30),      # polished steel: low emissivity
    ("space_heater", "box", (0.25, 0.12, 0.40), 60.0, 0.90),
    ("desk_lamp", "ellipsoid", (0.06, 0.06, 0.05), 48.0, 0.85),
    ("pc_tower", "box", (0.20, 0.40, 0.42), 30.0, 0.90),
    ("soldering_iron", "capsule", (0.012, 0.10), 140.0, 0.80),
    ("power_brick", "box", (0.14, 0.07, 0.04), 42.0, 0.92),
    ("hot_plate", "box", (0.22, 0.22, 0.04), 90.0, 0.90),
]


def add_warm_object(ps, kind_idx, base_xyz, rng=None, inst=-1, stand_temp_c=None):
    """Place catalogue object with its bottom at base_xyz. Optional ambient stand box below it."""
    name, kind, dims, temp, emis = WARM_CATALOGUE[kind_idx % len(WARM_CATALOGUE)]
    if rng is not None:
        temp = temp + rng.normal(0, 0.05 * max(temp - 22.0, 1.0))
    x, y, z = base_xyz
    yaw = rng.uniform(0, np.pi) if rng is not None else 0.0
    if stand_temp_c is not None and z > 0.05:
        ps.add_box([x, y, z / 2], [0.30, 0.30, z], stand_temp_c, label=FURNITURE, yaw=yaw, name="stand")
    if kind == "box":
        ps.add_box([x, y, z + dims[2] / 2], dims, temp, emissivity=emis, label=WARM, tex=0.06 * (temp - 20),
                   yaw=yaw, inst=inst, name=name)
        if name == "laptop":   # open screen
            ps.add_box([x - 0.11 * np.sin(yaw), y + 0.11 * np.cos(yaw), z + 0.12], (0.33, 0.015, 0.22), temp - 3.0,
                       emissivity=0.90, specular=0.2, label=WARM, tex=0.5, yaw=yaw, inst=inst, name="laptop_screen")
    elif kind == "capsule":
        r, h = dims
        ps.add_capsule([x, y, z + r], [x, y, z + max(h - r, r + 1e-3)], r, temp, emissivity=emis, label=WARM,
                       tex=0.05 * (temp - 20), inst=inst, name=name)
    else:
        ps.add_ellipsoid([x, y, z + dims[2]], dims, temp, emissivity=emis, label=WARM, tex=0.05 * (temp - 20),
                         inst=inst, name=name)
    return name


def sample_floor_points(cam, room, n, rng, depth=(0.9, 2.6), margin_deg=4.0):
    """Random points on the floor plan inside the camera's horizontal field of view."""
    pos = np.asarray(cam["position"], float)
    fwd = np.asarray(cam["look_dir"], float)
    az0 = np.arctan2(fwd[1], fwd[0])
    half = np.radians(cam["hfov_deg"] / 2.0 - margin_deg)
    size = room["room"]["size_xyz"]
    pts = []
    tries = 0
    while len(pts) < n and tries < 10000:
        tries += 1
        az = az0 + rng.uniform(-half, half)
        r = rng.uniform(*depth)
        x, y = pos[0] + r * np.cos(az), pos[1] + r * np.sin(az)
        if 0.15 < x < size[0] - 0.15 and 0.15 < y < size[1] - 0.15:
            if all((x - q[0]) ** 2 + (y - q[1]) ** 2 > 0.12 ** 2 for q in pts):
                pts.append((x, y))
    return pts
