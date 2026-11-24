"""Study conditions -> frame banks (uint16 TLinear frames + per-frame labels).

A spec is a plain dict (all keys optional):
  n_people, poses (list or "mixed"), n_objects, lamp_on, lamp_held, fog_level (0-3) or
  fog_beta_per_m, mist_offset_k, fog_plume, fog_turbulence, air_c, cam (override dict),
  person_at_mark, empty_room (drop the JSON's furniture), seed.
"""
import copy
import hashlib
import json
import time

import numpy as np

from . import scene as sc
from .render import Renderer, build_static
from .sensor import Lepton35, SensorParams

# Fog levels as in-band (8-14 um) extinction. alpha_LWIR = k * LWC with k ~ 0.14 m^2/g
# (ASSUMED, literature range 0.13-0.15; Chylek 1978 via the research track).
K_LWIR_M2_PER_G = 0.14
FOG_BETA = {0: 0.0, 1: 0.07, 2: 0.21, 3: 0.56}        # = LWC 0, 0.5, 1.5, 4.0 g/m^3

CONDITIONS = {
    "A0_empty": dict(n_people=0),
    "A1_one_person": dict(n_people=1, poses=["walking"]),
    "A2_three_people": dict(n_people=3, poses="mixed"),
    "A3_people_warm_objects": dict(n_people=3, poses="mixed", n_objects=5, lamp_on=True),
    "B0_fog0_person": dict(n_people=1, poses=["standing"], person_at_mark=True, fog_level=0),
    "B1_fog1_person": dict(n_people=1, poses=["standing"], person_at_mark=True, fog_level=1, mist_offset_k=-1.0),
    "B2_fog2_person": dict(n_people=1, poses=["standing"], person_at_mark=True, fog_level=2, mist_offset_k=-1.0),
    "B3_fog3_person": dict(n_people=1, poses=["standing"], person_at_mark=True, fog_level=3, mist_offset_k=-1.0),
    "L0_lamp_off": dict(n_people=0, lamp_on=False),
    "L1_lamp_on": dict(n_people=0, lamp_on=True),
}
for _n in (0, 1, 2, 4, 8, 12, 16):
    CONDITIONS[f"C{_n:02d}_objects"] = dict(n_people=0, n_objects=_n)


def shift_air(room, air_c):
    """Copy of the room with air and all near-ambient surfaces shifted to a new air temperature."""
    room = copy.deepcopy(room)
    d = air_c - room["room"]["air_temp_c"]
    room["room"]["air_temp_c"] = air_c
    for s in room["room"]["surfaces"].values():
        s["temp_c"] += d
    for ob in room.get("objects", []):
        if "temp_c" in ob and ob["temp_c"] < air_c - d + 3.0:
            ob["temp_c"] += d
    return room


def _face_camera_yaw(pos_xy, cam):
    f = np.asarray(cam["position"][:2], float) - np.asarray(pos_xy, float)
    return float(np.arctan2(-f[0], f[1]))


class Scenario:
    """Builds the static layer once and yields per-frame dynamic primitives + labels."""

    def __init__(self, spec, room=None, room_path=None, backend="numpy", device=None, supersample=2):
        self.spec = dict(spec)
        room = room if room is not None else sc.load_room(room_path)
        if spec.get("air_c") is not None:
            room = shift_air(room, spec["air_c"])
        self.room = room
        self.seed = int(spec.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.cam = sc.camera_from_room(room, **spec.get("cam", {}))
        self.air_c = room["room"]["air_temp_c"]
        self.renderer = Renderer(room, self.cam, backend=backend, device=device, supersample=supersample,
                                 seed=self.seed)
        static = build_static(room, cam=self.cam, include_objects=not spec.get("empty_room", False))
        slots = room.get("dynamic_slots", {})
        # warm objects for the complexity sweep
        self.object_names = []
        n_obj = int(spec.get("n_objects", 0))
        if n_obj:
            pts = sc.sample_floor_points(self.cam, room, n_obj, self.rng, depth=spec.get("object_depth", (0.7, 3.2)))
            order = self.rng.permutation(len(sc.WARM_CATALOGUE))
            for k, (x, y) in enumerate(pts):
                z = float(self.rng.choice([0.0, 0.3, 0.74, 1.0]))
                self.object_names.append(sc.add_warm_object(static, int(order[k % len(order)]), (x, y, z), self.rng,
                                                            inst=-1, stand_temp_c=self.air_c))
        # fixed heat lamp (faces away from the camera, toward the far wall)
        lamp = None
        if spec.get("lamp_on") and not spec.get("lamp_held"):
            hl = slots.get("heat_lamp", {})
            pos = np.asarray(spec.get("lamp_position", hl.get("position", [0.6, 1.4, 1.0])), float)
            look = np.asarray(self.cam["look_dir"], float)
            aim = np.array(spec.get("lamp_aim", [look[0], look[1], -0.15]), float)
            sc.add_heat_lamp(static, pos, aim=aim, temp_c=hl.get("temp_c", 180.0))
            lamp = dict(position=pos.tolist(), aim=aim.tolist(), power_w=spec.get("lamp_power_w", 250.0),
                        warm_fraction=spec.get("lamp_warm_fraction", 1.0))
        self.lamp = lamp
        self.renderer.set_static(static, lamp=lamp)
        self.n_static_prims = len(static)
        # people
        self.people = []
        n_people = int(spec.get("n_people", 0))
        poses = spec.get("poses", ["standing"])
        if poses == "mixed":
            poses = [["walking", "standing", "sitting", "arms_raised"][k % 4] for k in range(n_people)]
        depth = spec.get("person_depth", (0.9, 3.0))
        pts = sc.sample_floor_points(self.cam, room, 2 * n_people, self.rng, depth=depth, margin_deg=8.0)
        for k in range(n_people):
            pose = poses[k % len(poses)]
            p0, p1 = np.array(pts[2 * k]), np.array(pts[2 * k + 1])
            if spec.get("person_at_mark") and k == 0 and "person_mark" in slots:
                p0 = np.array(slots["person_mark"][:2], float)
            if spec.get("person_xy") is not None:
                p0 = np.array(spec["person_xy"][k], float)
            prng = np.random.default_rng(self.seed * 1000 + k)
            self.people.append(dict(pose=pose, p0=p0, p1=p1, height=float(prng.uniform(1.60, 1.88)),
                                    temps=sc.person_temps(self.air_c, prng), phase0=float(prng.uniform()),
                                    speed=float(prng.uniform(0.6, 1.1))))
        # fog
        lvl = spec.get("fog_level", 0)
        beta = float(spec.get("fog_beta_per_m", FOG_BETA.get(lvl, 0.0)))
        self.fog = None
        if beta > 0 or spec.get("fog_plume"):
            self.fog = dict(beta_per_m=beta, mist_offset_k=float(spec.get("mist_offset_k", 0.0)),
                            turbulence=float(spec.get("fog_turbulence", 0.25)), seed=self.seed)
            if spec.get("fog_plume"):
                fs = slots.get("fog_source", {}).get("position", [1.35, 1.6, 0.3])
                self.fog["plume"] = dict(center=[fs[0], fs[1], fs[2] + 0.5], sigma_m=0.35,
                                         beta_peak_per_m=float(spec.get("plume_beta_peak", 4 * max(beta, 0.1))))

    def dynamic(self, t):
        if not self.people:
            return None
        ps = sc.PrimSet()
        for k, p in enumerate(self.people):
            if p["pose"] == "walking":
                span = p["p1"] - p["p0"]
                dist = np.linalg.norm(span) + 1e-9
                u = (t * p["speed"] / dist + p["phase0"]) % 2.0
                fwd = u < 1.0
                frac = u if fwd else 2.0 - u
                pos = p["p0"] + frac * span
                dirn = span / dist * (1 if fwd else -1)
                yaw = float(np.arctan2(-dirn[0], dirn[1]))
                phase = (t * p["speed"] / 1.4 + p["phase0"]) % 1.0
            else:
                pos, yaw = p["p0"], _face_camera_yaw(p["p0"], self.cam) + 0.15 * np.sin(0.4 * t + k)
                phase = (0.25 * t + p["phase0"]) % 1.0
            sc.add_person(ps, pos, yaw=yaw, pose=p["pose"], phase=phase, height=p["height"], inst=k,
                          air_c=self.air_c, temps=p["temps"],
                          lamp_in_hand=bool(self.spec.get("lamp_held")) and k == 0,
                          lamp_temp_c=self.spec.get("lamp_temp_c", 180.0))
        return ps

    def render(self, t):
        return self.renderer.render(self.dynamic(t), fog=self.fog, time_s=t)


def generate_bank(spec, n_frames=128, room_path=None, backend="numpy", device=None, supersample=2,
                  sensor_params=None, noise=True, first_ffc_s=None, room=None, return_ideal=False):
    """Render + sense `n_frames`. Returns (frames uint16 (N,120,160), manifest dict)."""
    t0 = time.time()
    scn = Scenario(spec, room=room, room_path=room_path, backend=backend, device=device, supersample=supersample)
    sp = sensor_params or SensorParams(seed=int(spec.get("seed", 0)))
    cam = Lepton35(sp, supersample=supersample, noise=noise,
                   first_ffc_s=spec.get("first_ffc_s", 1e12) if first_ffc_s is None else first_ffc_s)
    frames, labels, ideal = [], [], []
    render_s = sensor_s = 0.0
    for _ in range(n_frames):
        t = cam.t
        a = time.perf_counter()
        out = scn.render(t)
        rad = scn.renderer.bk.to_numpy(out["radiance"])
        b = time.perf_counter()
        if return_ideal:
            ideal.append(cam.ideal_counts(rad))
        f, info = cam.capture(rad)
        c = time.perf_counter()
        render_s += b - a
        sensor_s += c - b
        boxes = scn.renderer.person_boxes(out)
        frames.append(f)
        labels.append(dict(t=round(info["t"], 4), n_people=len(boxes), n_people_in_scene=len(scn.people),
                           boxes=[[round(v, 1) for v in bx["xyxy"]] for bx in boxes],
                           fog_level=int(spec.get("fog_level", 0)), fog_beta_per_m=scn.fog["beta_per_m"] if scn.fog else 0.0,
                           tau_mean=round(out.get("tau_mean", 1.0), 4), lamp_on=bool(spec.get("lamp_on", False)),
                           n_warm_objects=len(scn.object_names), ffc=bool(info["ffc"])))
    room_json = json.dumps({k: v for k, v in scn.room.items() if k != "_path"}, sort_keys=True)
    manifest = dict(generator="synthlab.thermal_physics (ray-cast radiometric scene + calibrated Lepton 3.5 sensor model)",
                    spec=spec, seed=int(spec.get("seed", 0)), n_frames=int(n_frames), shape=[int(n_frames), 120, 160],
                    dtype="uint16", units="TLinear 0.01 K/count", fps=sp.fps,
                    scene_file=str(scn.room.get("_path", "")), scene_sha1=hashlib.sha1(room_json.encode()).hexdigest(),
                    camera=scn.cam, air_temp_c=scn.air_c, fog=scn.fog, lamp=scn.lamp, warm_objects=scn.object_names,
                    n_static_prims=scn.n_static_prims, backend=backend, device=str(scn.renderer.bk.device),
                    supersample=supersample, sensor=cam.describe(), sensor_noise=bool(noise),
                    timing=dict(render_ms_per_frame=1e3 * render_s / n_frames, sensor_ms_per_frame=1e3 * sensor_s / n_frames,
                                total_s=time.time() - t0),
                    frames=labels)
    frames = np.stack(frames)
    if return_ideal:
        return frames, manifest, np.stack(ideal)
    return frames, manifest
