"""Experiment conditions A0-A3 (scene complexity) and B0-B3 (fog, same still person),
plus two extra motion conditions (M1, M3) that are NOT in the project Guide but are
needed to span the detection-count axis.

The Guide only says "several people" / "several people plus warm objects"; the exact
positions below are this track's ASSUMPTION. Warm objects matter to the thermal camera;
to the radar they are just extra reflectors (a heat lamp has a metal reflector bowl).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .fog import FOG_LEVELS, FogState
from .person import Person
from .scene import floor_xy, radar_pose

CONDITIONS = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3", "M1", "M3"]

# extra people / walk waypoints in the radar ground frame (lateral m, range m)
EXTRA_PEOPLE_LR = [(-0.65, 0.95), (0.65, 1.15)]
WALKS_LR = [((-0.8, 0.7), (0.7, 1.6)), ((0.8, 0.6), (-0.6, 1.7)), ((0.0, 0.6), (0.1, 1.7))]


def warm_objects(scene: dict) -> List[dict]:
    lamp = scene.get("dynamic_slots", {}).get("heat_lamp", {}).get("position", [0.6, 1.4, 1.0])
    k_xy, b_xy = floor_xy(scene, 0.9, 1.3), floor_xy(scene, -0.9, 1.5)
    return [
        {"id": "heat_lamp_reflector", "shape": "point", "center": lamp, "rcs_m2": 0.05},
        {"id": "heat_lamp_stand", "shape": "box", "center": [lamp[0], lamp[1], lamp[2] / 2],
         "size": [0.04, 0.04, lamp[2]], "material": "metal"},
        {"id": "kettle", "shape": "box", "center": k_xy + [0.45], "size": [0.2, 0.2, 0.25],
         "material": "metal"},
        {"id": "warm_box", "shape": "box", "center": b_xy + [0.25], "size": [0.4, 0.4, 0.5],
         "material": "cardboard"},
    ]


def build_condition(name: str, scene: dict, seed: int = 0) -> Dict[str, object]:
    """Return dict(people=[Person], extra_objects=[...], fog=FogState, description=str)."""
    pose = radar_pose(scene)
    mark = scene["dynamic_slots"]["person_mark"][:2]
    still = lambda xy, k: Person(xy, face_to=pose.position, seed=seed * 100 + k)
    people: List[Person] = []
    extra: List[dict] = []
    fog: Optional[FogState] = FogState()
    if name == "A0":
        desc = "empty room"
    elif name == "A1":
        people = [still(mark, 1)]
        desc = "one still (breathing, swaying) person on the mark"
    elif name == "A2":
        people = [still(mark, 1)] + [still(floor_xy(scene, *lr), 2 + i)
                                    for i, lr in enumerate(EXTRA_PEOPLE_LR)]
        desc = "three still people"
    elif name == "A3":
        people = [still(mark, 1)] + [still(floor_xy(scene, *lr), 2 + i)
                                    for i, lr in enumerate(EXTRA_PEOPLE_LR)]
        extra = warm_objects(scene)
        desc = "three still people + warm objects (lamp, kettle, box)"
    elif name in FOG_LEVELS:
        people = [still(mark, 1)]
        fog = FOG_LEVELS[name]
        desc = "one still person, fog %s (LWC %.2f g/m^3)" % (fog.name, fog.lwc_gm3)
    elif name in ("M1", "M3"):
        speeds = [1.0, 0.8, 1.3]
        n = 1 if name == "M1" else 3
        people = [Person(floor_xy(scene, *WALKS_LR[i][0]), mode="walk",
                         walk_to=floor_xy(scene, *WALKS_LR[i][1]), walk_speed=speeds[i],
                         seed=seed * 100 + 7 + i) for i in range(n)]
        desc = "%d person(s) walking back and forth (extra condition, not in the Guide)" % n
    else:
        raise ValueError("unknown condition %r" % name)
    return {"name": name, "people": people, "extra_objects": extra, "fog": fog,
            "description": desc}
