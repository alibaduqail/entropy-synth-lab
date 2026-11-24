"""Room geometry in metres, with Z up and world-space box centres."""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SURFACES = ("floor", "ceiling", "wall_south", "wall_north", "wall_west", "wall_east")


def load_scene(path=None):
    if path is None:
        candidates = [(int(p.stem[6:]), p) for p in (ROOT / "scene").glob("room_v*.json")
                      if re.fullmatch(r"room_v\d+", p.stem)]
        if not candidates:
            raise FileNotFoundError("No versioned room scene found")
        path = max(candidates)[1]
    with (ROOT / path).open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_boxes(scene, include_detail=True, expand_children=False):
    """Children have world coordinates and inherit missing material properties."""
    def visit(raw, inherited):
        box = {**inherited, **raw}
        if not include_detail and box.get("detail", False):
            return
        if expand_children and "children" in box:
            if box.get("solid", True):
                props = {k: box[k] for k in ("material", "temp_c", "emissivity",
                         "lidar_reflectivity", "specular", "lidar_transmission") if k in box}
                for child in box["children"]:
                    yield from visit(child, props)
            return
        yield {**box, "center": np.asarray(box["center"], float),
               "size": np.asarray(box["size"], float), "yaw_deg": box.get("yaw_deg", 0.),
               "solid": box.get("solid", True)}

    for raw in scene["objects"]:
        yield from visit(raw, {})


def rotation(yaw_deg):
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box_corners(center, size, yaw_deg=0.):
    signs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
    return center + (signs * size * .5) @ rotation(yaw_deg).T


def ray_box_intersect(origin, direction, center, size, yaw_deg=0.):
    """First forward surface parameter t (exit from inside), inf on miss. Batched directions OK."""
    rot = rotation(yaw_deg)
    o = (np.asarray(origin) - center) @ rot
    d = np.asarray(direction) @ rot
    half = np.asarray(size) * .5
    parallel = np.abs(d) < 1e-14
    divisor = np.where(parallel, 1., d)
    a, b = (-half - o) / divisor, (half - o) / divisor
    entry = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=-1)
    leave = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=-1)
    miss = (parallel & (np.abs(o) > half + 1e-12)).any(axis=-1) | (leave < np.maximum(entry, 0))
    return np.where(miss, np.inf, np.where(entry >= 0, entry, leave))


def shell_quads(scene):
    """Yield (surface id, inward-wound quad), splitting the wall around the doorway."""
    size = scene["room"]["size_xyz"]
    opening = next((o for o in scene["objects"] if o["id"] == "door_opening" and "wall" in o), {})
    for name, (u, v, w, high, reverse) in zip(SURFACES, (
            (0,1,2,False,False), (0,1,2,True,True), (0,2,1,False,True),
            (0,2,1,True,False), (1,2,0,False,False), (1,2,0,True,True))):
        U, V = size[u], size[v]
        rects = [(0, U, 0, V)]
        if opening and name == "wall_" + opening["wall"].removeprefix("wall_"):
            extents = opening.get("extents", {})
            horizontal = opening.get("u_range", extents.get("xy"[u]))
            vertical = opening.get("z_range", extents.get("z"))
            if horizontal is not None and vertical is not None:
                a, b = np.clip(horizontal, 0, U)
                c, d = np.clip(vertical, 0, V)
                rects = [(0,a,0,V), (b,U,0,V), (a,b,0,c), (a,b,d,V)]
        for left, right, bottom, top in rects:
            if right <= left or top <= bottom:
                continue
            q = np.zeros((4, 3))
            q[:, u] = [left, right, right, left]
            q[:, v] = [bottom, bottom, top, top]
            q[:, w] = size[w] if high else 0.
            yield name, q[::-1] if reverse else q


def cast_scene(scene, origin, directions, boxes):
    """Nearest opaque surface distance and object name. Rays escaping a door return inf."""
    d = np.asarray(directions)
    origin = np.asarray(origin)
    best = np.full(d.shape[:-1], np.inf)
    ids = np.full(best.shape, "", dtype=object)
    for name, quad in shell_quads(scene):
        lo, hi = quad.min(axis=0), quad.max(axis=0)
        axis = np.argmin(hi - lo)
        component = d[..., axis]
        t = (lo[axis] - origin[axis]) / np.where(component != 0, component, 1.)
        point = origin + t[..., None] * d
        hit = ((component != 0) & (t >= 0) & (t < best) &
               ((lo-1e-9 <= point) & (point <= hi+1e-9)).all(axis=-1))
        best[hit], ids[hit] = t[hit], name
    for box in boxes:
        if box["solid"]:
            t = ray_box_intersect(origin, d, box["center"], box["size"], box["yaw_deg"])
            hit = t < best
            best[hit], ids[hit] = t[hit], box["id"]
    return best, ids


def schema_problems(scene):
    problems = []

    def number(x):
        return type(x) in (float, int) and np.isfinite(x)

    def check(obj, path, **fields):
        for key, kind in fields.items():
            x = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(kind, int):
                ok = isinstance(x, list) and len(x) == kind and all(number(i) for i in x)
            elif kind is float:
                ok = number(x)
            else:
                ok = isinstance(x, kind) and (kind is not int or type(x) is int)
            if not ok:
                problems.append(f"{path}.{key}: expected {kind}")

    check(scene, "scene", units=str, up=str, origin=str, room=dict, objects=list,
          sensors=dict, dynamic_slots=dict)
    if problems:
        return problems
    if scene["units"] != "m" or scene["up"] != "z":
        problems.append("scene requires metre units and z up")
    room = scene["room"]
    check(room, "room", size_xyz=3, air_temp_c=float, surfaces=dict)
    if not problems and min(room["size_xyz"]) <= 0:
        problems.append("room.size_xyz must be positive")
    props = dict(material=str, emissivity=float, temp_c=float, lidar_reflectivity=float)
    if isinstance(room.get("surfaces"), dict):
        for name in SURFACES:
            check(room["surfaces"].get(name), name, **props)
    names = set(SURFACES)

    def box(raw, path, inherited):
        if not isinstance(raw, dict):
            problems.append(f"{path}: expected object")
            return
        values = {**inherited, **raw}
        check(values, path, id=str, shape=str, center=3, size=3, **props)
        for key, kind in dict(yaw_deg=float, solid=bool, children=list,
                              specular=float, lidar_transmission=float, wall=str, u_range=2,
                              z_range=2, hinge_xy=2, open_angle_deg=float, extents=dict).items():
            if key in raw:
                check(raw, path, **{key: kind})
        if isinstance(raw.get("extents"), dict):
            for axis in ("x", "y", "z"):
                if axis in raw["extents"]:
                    check(raw["extents"], f"{path}.extents", **{axis: 2})
        if values.get("shape") != "box":
            problems.append(f"{path}: expected box")
        if isinstance(raw.get("id"), str):
            if raw["id"] in names:
                problems.append(f"{path}: duplicate id {raw['id']}")
            names.add(raw["id"])
        if isinstance(raw.get("size"), list) and any(number(n) and n <= 0 for n in raw["size"]):
            problems.append(f"{path}: nonpositive size")
        if isinstance(raw.get("children"), list):
            for i, child in enumerate(raw["children"]):
                box(child, f"{path}.children[{i}]", {k: values.get(k) for k in (*props, "shape")})

    for i, obj in enumerate(scene["objects"]):
        box(obj, f"objects[{i}]", {})
    specs = {"lepton35": dict(look_dir=3, hfov_deg=float, vfov_deg=float, width=int,
                              height=int, fps=float),
             "mid360": dict(forward=3, hfov_deg=float, vfov_deg=2, points_per_s=int, frame_hz=float),
             "iwr6843isk": dict(boresight=3, az_fov_deg=float, el_fov_deg=float)}
    for name, fields in specs.items():
        check(scene["sensors"].get(name), name, position=3, **fields)
    for name, sensor in scene["sensors"].items():
        if name.startswith("_"):
            continue
        check(sensor, name, position=3)
        if isinstance(sensor, dict) and "mounted_on" in sensor:
            check(sensor, name, mounted_on=list)
    slots = scene["dynamic_slots"]
    check(slots, "dynamic_slots", person_mark=3, heat_lamp=dict, fog_source=dict)
    for name, slot in slots.items():
        if name.startswith("_"):
            continue
        if isinstance(slot, dict):
            check(slot, name, position=3)
        else:
            check(slots, "dynamic_slots", **{name: 3})
    check(slots.get("heat_lamp"), "heat_lamp", temp_c=float)
    if "cameras" in scene:
        check(scene, "scene", cameras=list)
        if isinstance(scene["cameras"], list):
            for cam in scene["cameras"]:
                check(cam, "camera", id=str, position=3, look_at=3, up=3,
                      hfov_deg=float, vfov_deg=float, image_wh=2)
    return problems


def overlaps(a, b):
    delta = b["center"] - a["center"]
    limit = .001 + 1e-12
    def interval_overlap(half_a, half_b, distance):
        return np.minimum(2 * np.minimum(half_a, half_b), half_a + half_b - distance)

    if a["yaw_deg"] == b["yaw_deg"] == 0:
        return np.all(interval_overlap(a["size"]/2, b["size"]/2, abs(delta)) > limit)
    ra, rb = (rotation(box["yaw_deg"]) for box in (a, b))
    for axis in (*ra.T, *rb.T[:2]):
        ha = abs(axis @ ra) @ a["size"] / 2
        hb = abs(axis @ rb) @ b["size"] / 2
        if interval_overlap(ha, hb, abs(axis @ delta)) <= limit:
            return False
    return True


def validate_scene(scene):
    problems = schema_problems(scene)
    if problems:
        return problems
    size = np.array(scene["room"]["size_xyz"])
    boxes = [b for b in iter_boxes(scene) if b["solid"]]
    for b in boxes:
        corners = box_corners(b["center"], b["size"], b["yaw_deg"])
        if ((corners < -1e-6) | (corners > size+1e-6)).any():
            problems.append(f"{b['id']} outside room")
    for a, b in combinations(boxes, 2):
        if overlaps(a, b):
            problems.append(f"{a['id']} overlaps {b['id']}")

    target = np.asarray(scene["dynamic_slots"]["person_mark"]) + [0, 0, 1.]
    for group in ("sensors", "dynamic_slots"):
        for name, item in scene[group].items():
            if name.startswith("_"):
                continue
            p = np.asarray(item["position"] if isinstance(item, dict) else item)
            if ((p < -1e-6) | (p > size+1e-6)).any():
                problems.append(f"{name} outside room")
            for b in boxes:
                offsets = abs((p - b["center"]) @ rotation(b["yaw_deg"])) - b["size"]/2
                if np.max(offsets) < -1e-9:
                    problems.append(f"{name} inside {b['id']}")
                if name in ("lepton35", "iwr6843isk"):
                    surface_distance = (np.linalg.norm(np.maximum(offsets, 0)) +
                                        abs(min(np.max(offsets), 0)))
                    if b["id"] in item.get("mounted_on", []) or surface_distance <= .02:
                        continue
                    t = ray_box_intersect(p, target-p, b["center"], b["size"], b["yaw_deg"])
                    if t < 1:
                        problems.append(f"{name} sightline blocked by {b['id']}")
    az = np.deg2rad(np.arange(360))
    el = np.deg2rad(10)
    rays = np.column_stack((np.cos(az)*np.cos(el), np.sin(az)*np.cos(el),
                            np.full(360, np.sin(el))))
    ranges, _ = cast_scene(scene, scene["sensors"]["mid360"]["position"], rays, boxes)
    if np.mean(ranges >= .5) < .60:
        problems.append(f"mid360 clearance {np.mean(ranges >= .5):.1%} below 60%")
    return problems
