"""A person as a cluster of point scatterers with breathing, heartbeat, postural sway,
optional discrete movements ("events") and optional walking with limb swing.

RCS split is ASSUMED: torso (chest+abdomen+shoulders) ~0.53 m^2, whole body ~0.9 m^2,
in the 0.5-1 m^2 range usually quoted for an adult at mmWave.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# name, lateral offset (m), forward offset (m), height fraction, rcs (m^2), group
BODY_PARTS = [
    ("chest",      0.00, 0.10, 0.72, 0.30, "chest"),
    ("abdomen",    0.00, 0.10, 0.60, 0.15, "abdomen"),
    ("shoulder_l", -0.19, 0.05, 0.82, 0.04, "shoulder"),
    ("shoulder_r", 0.19, 0.05, 0.82, 0.04, "shoulder"),
    ("head",       0.00, 0.05, 0.94, 0.06, "head"),
    ("upperarm_l", -0.22, 0.00, 0.70, 0.02, "arm_l"),
    ("upperarm_r", 0.22, 0.00, 0.70, 0.02, "arm_r"),
    ("forearm_l",  -0.24, 0.02, 0.55, 0.02, "arm_l"),
    ("forearm_r",  0.24, 0.02, 0.55, 0.02, "arm_r"),
    ("thigh_l",    -0.09, 0.03, 0.40, 0.06, "leg_l"),
    ("thigh_r",    0.09, 0.03, 0.40, 0.06, "leg_r"),
    ("shin_l",     -0.09, 0.02, 0.20, 0.04, "leg_l"),
    ("shin_r",     0.09, 0.02, 0.20, 0.04, "leg_r"),
    ("foot_l",     -0.09, 0.08, 0.03, 0.01, "leg_l"),
    ("foot_r",     0.09, 0.08, 0.03, 0.01, "leg_r"),
]
# breathing amplitude (fraction of the chest amplitude) per group
BREATH_GAIN = {"chest": 1.0, "abdomen": 0.75, "shoulder": 0.25, "head": 0.05}
# limb swing gain along the heading while walking (fraction of the foot swing)
SWING_GAIN = {"thigh": 0.4, "shin": 0.8, "foot": 1.0, "upperarm": 0.25, "forearm": 0.5}


@dataclass
class Event:
    """Smooth (raised-cosine) displacement of the whole body by ``delta`` (room frame, m)
    starting at ``t0`` and lasting ``duration`` seconds; the person stays displaced."""
    t0: float
    duration: float
    delta: Tuple[float, float, float]

    def offset(self, t: np.ndarray) -> np.ndarray:
        u = np.clip((np.asarray(t) - self.t0) / self.duration, 0.0, 1.0)
        s = 0.5 * (1.0 - np.cos(np.pi * u))
        return s[..., None] * np.asarray(self.delta, dtype=float)


@dataclass
class Person:
    position: Sequence[float]                 # (x, y) of the feet, room frame
    facing: Optional[Sequence[float]] = None  # unit (x, y); default = toward ``face_to``
    face_to: Optional[Sequence[float]] = None  # e.g. the radar position
    height: float = 1.72
    mode: str = "still"                       # "still" | "walk"
    breath_amp_m: float = 2.0e-3              # chest moves +-2 mm (4 mm peak-to-peak)
    breath_hz: float = 0.25
    heart_amp_m: float = 0.15e-3
    heart_hz: float = 1.15
    sway_rms_m: float = 2.0e-3                # postural sway of the chest, fore-aft (ASSUMED)
    events: List[Event] = field(default_factory=list)
    walk_to: Optional[Sequence[float]] = None  # second waypoint (x, y) for mode="walk"
    walk_speed: float = 1.0
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._phase_b = rng.uniform(0, 2 * np.pi)
        self._phase_h = rng.uniform(0, 2 * np.pi)
        self._phase_w = rng.uniform(0, 1.0)
        self.breath_hz = float(self.breath_hz * rng.uniform(0.9, 1.1))
        # sway: sum of sinusoids, ~1/f amplitude, 0.05-0.8 Hz
        n = 8
        self._sway_f = rng.uniform(0.05, 0.8, size=(2, n))
        self._sway_p = rng.uniform(0, 2 * np.pi, size=(2, n))
        amp = 1.0 / self._sway_f
        amp /= np.sqrt(0.5 * (amp ** 2).sum(axis=1, keepdims=True))   # unit rms each axis
        self._sway_a = amp
        pos = np.array(self.position, dtype=float)[:2]
        if self.facing is not None:
            f = np.array(self.facing, dtype=float)[:2]
        elif self.face_to is not None:
            f = np.array(self.face_to, dtype=float)[:2] - pos
        else:
            f = np.array([0.0, 1.0])
        self._facing = f / max(np.linalg.norm(f), 1e-9)
        self.names = [p[0] for p in BODY_PARTS]
        self.rcs = np.array([p[4] for p in BODY_PARTS])

    def __len__(self) -> int:
        return len(BODY_PARTS)

    @property
    def xy0(self) -> np.ndarray:
        return np.array(self.position, dtype=float)[:2]

    # ------------------------------------------------------------------
    def _walk_state(self, t: np.ndarray):
        """Body centre (T,2) and heading (T,2) for a back-and-forth walk."""
        a = self.xy0
        b = np.array(self.walk_to if self.walk_to is not None else a + [0.0, 1.0], dtype=float)[:2]
        length = max(np.linalg.norm(b - a), 1e-6)
        period = 2.0 * length / self.walk_speed
        u = ((t / period) + self._phase_w) % 1.0
        tri = np.where(u < 0.5, 2 * u, 2 - 2 * u)             # 0..1..0
        ctr = a[None, :] + tri[:, None] * (b - a)[None, :]
        sign = np.where(u < 0.5, 1.0, -1.0)
        heading = sign[:, None] * ((b - a) / length)[None, :]
        return ctr, heading

    def positions(self, t: np.ndarray) -> np.ndarray:
        """Scatterer positions (T, S, 3) in the room frame at times ``t`` (s)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        T, S = len(t), len(BODY_PARTS)
        lat = np.array([p[1] for p in BODY_PARTS])
        fwd = np.array([p[2] for p in BODY_PARTS])
        hgt = np.array([p[3] for p in BODY_PARTS]) * self.height
        if self.mode == "walk":
            ctr, heading = self._walk_state(t)
        else:
            ctr = np.repeat(self.xy0[None, :], T, axis=0)
            heading = np.repeat(self._facing[None, :], T, axis=0)
        side = np.stack([heading[:, 1], -heading[:, 0]], axis=1)       # right-hand side
        out = np.zeros((T, S, 3))
        out[:, :, :2] = (ctr[:, None, :] + lat[None, :, None] * side[:, None, :]
                         + fwd[None, :, None] * heading[:, None, :])
        out[:, :, 2] = hgt[None, :]
        # breathing + heartbeat: along the facing direction
        b = np.sin(2 * np.pi * self.breath_hz * t + self._phase_b)
        h = np.sin(2 * np.pi * self.heart_hz * t + self._phase_h)
        for s, part in enumerate(BODY_PARTS):
            g = BREATH_GAIN.get(part[5], 0.0)
            if g:
                d = g * self.breath_amp_m * b
                if part[5] == "chest":
                    d = d + self.heart_amp_m * h
                out[:, s, :2] += d[:, None] * heading
        # postural sway: inverted pendulum, scaled by height, fore-aft and (0.6x) lateral
        if self.sway_rms_m > 0 and self.mode != "walk":
            sw = (self._sway_a[:, None, :] * np.sin(
                2 * np.pi * self._sway_f[:, None, :] * t[None, :, None]
                + self._sway_p[:, None, :])).sum(axis=2)                # (2, T)
            scale = hgt / (0.72 * self.height)
            out[:, :, :2] += (self.sway_rms_m * sw[0])[:, None, None] * scale[None, :, None] \
                * heading[:, None, :]
            out[:, :, :2] += (0.6 * self.sway_rms_m * sw[1])[:, None, None] \
                * scale[None, :, None] * side[:, None, :]
        if self.mode == "walk":
            gait_hz = self.walk_speed / 1.4                    # stride length 1.4 m
            ph = 2 * np.pi * gait_hz * t
            for s, part in enumerate(BODY_PARTS):
                key = part[0].split("_")[0]
                g = SWING_GAIN.get(key, 0.0)
                if g:
                    left = part[0].endswith("_l")
                    arm = "arm" in part[0]
                    sgn = (1.0 if left else -1.0) * (-1.0 if arm else 1.0)
                    out[:, s, :2] += (sgn * g * 0.30 * np.sin(ph))[:, None] * heading
            out[:, :, 2] += 0.015 * np.sin(2 * ph)[:, None]     # vertical bob
        for ev in self.events:
            out += ev.offset(t)[:, None, :]
        return out


def radial_event(person: Person, radar_xy: Sequence[float], t0: float, distance_m: float,
                 duration: float = 1.0, lateral: bool = False) -> Event:
    """Event moving the person ``distance_m`` toward the radar (or sideways if ``lateral``)."""
    d = np.array(radar_xy, dtype=float)[:2] - person.xy0
    d /= max(np.linalg.norm(d), 1e-9)
    if lateral:
        d = np.array([d[1], -d[0]])
    return Event(t0, duration, (d[0] * distance_m, d[1] * distance_m, 0.0))
