"""Room geometry regression checks. Run directly or collect with pytest."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthlab.room3d.build_mesh import build_mesh, scene_to_triangles
from synthlab.room3d.render_preview import cameras, near_clip, render
from synthlab.room3d.scene import (SURFACES, cast_scene, iter_boxes, load_scene,
                                   overlaps, ray_box_intersect, schema_problems, shell_quads,
                                   validate_scene)


def v0():
    return load_scene("scene/room_v0.json")


def test_versioned_scenes():
    paths = sorted((ROOT / "scene").glob("room_v*.json"), key=lambda p: int(p.stem[6:]))
    assert paths and load_scene() == load_scene(paths[-1])
    for path in paths:
        scene = load_scene(path)
        assert not schema_problems(scene), (path, schema_problems(scene))
        issues = validate_scene(scene)
        if int(path.stem[6:]) >= 1:
            assert not issues, (path, issues)
        else:
            print(f"{path.name} (provisional): " + (", ".join(issues) or "valid"))


def test_ray_box():
    center, size = np.zeros(3), np.array([2,1,1])
    def hit(o, d, yaw=0):
        return ray_box_intersect(o, d, center, size, yaw)
    assert hit([-3,0,0], [1,0,0]) == 2
    assert hit([0,0,0], [2,0,0]) == .5
    assert hit([-3,.5,0], [1,0,0]) == 2
    assert np.isinf(hit([-3,1,0], [1,0,0]))
    assert np.isinf(hit([-3,0,0], [-1,0,0]))
    assert np.isclose(hit([-3,0,0], [1,0,0], 45), 3-2**-.5)
    assert np.isclose(hit([0,-3,0], [0,1,0], 90), 2)


def test_validation():
    s = v0()
    s["objects"] = []
    assert validate_scene(s) == []
    template = v0()["objects"][0]
    eye = s["sensors"]["lepton35"]["position"]
    target = np.asarray(s["dynamic_slots"]["person_mark"]) + [0,0,1]
    blocker = dict(template, id="blocker", center=((eye+target)/2).tolist(), size=[.2]*3)
    s["objects"] = [blocker]
    assert "lepton35 sightline blocked by blocker" in validate_scene(s)
    s["sensors"]["lepton35"]["mounted_on"] = ["blocker"]
    assert "lepton35 sightline blocked by blocker" not in validate_scene(s)
    blocker.update(center=[eye[0],eye[1]-.11,eye[2]])
    del s["sensors"]["lepton35"]["mounted_on"]
    assert "lepton35 sightline blocked by blocker" not in validate_scene(s)
    blocker.update(center=s["sensors"]["mid360"]["position"], size=[.1]*3)
    assert any("mid360 clearance" in line for line in validate_scene(s))
    blocker.update(center=s["dynamic_slots"]["fog_source"]["position"])
    assert "fog_source inside blocker" in validate_scene(s)
    blocker["solid"] = False
    assert not validate_scene(s)
    s["sensors"]["_comment"] = "ignore"
    del s["sensors"]["lepton35"]["position"]
    assert any("lepton35.position" in line for line in validate_scene(s))


def test_rotated_overlaps():
    a = dict(center=np.ones(3), size=np.array([1,.1,.1]), yaw_deg=45)
    normal = np.array([-1,1,0])/np.sqrt(2)
    for gap, expected in ((.1, False), (.0991, False), (.0989, True)):
        b = dict(a, center=a["center"] + gap*normal)
        assert bool(overlaps(a,b)) == expected
    a.update(yaw_deg=0)
    for gap, expected in ((.9991,False), (.9989,True)):
        assert bool(overlaps(a,dict(a,center=a["center"]+[gap,0,0]))) == expected


def test_door_and_children():
    s = v0()
    parent = s["objects"][0]
    child = dict(id="rotated_leaf", shape="box", center=parent["center"],
                 size=[.2,.01,.01], yaw_deg=45, detail=True)
    parent["children"] = [child]
    opening = dict(parent, id="door_opening", solid=False, children=[], wall="east",
                   u_range=[.4,1.2], z_range=[0,2])
    s["objects"] = [parent, opening]
    assert not validate_scene(s)
    assert [b["id"] for b in iter_boxes(s)] == [parent["id"], "door_opening"]
    assert not list(iter_boxes(s, include_detail=False, expand_children=True))
    *_, ids = scene_to_triangles(s)
    assert len(ids) == 12 + 10 + 6
    assert sum(ids == "rotated_leaf") == 12 and "door_opening" not in ids
    rays = np.array([[1,0,0],[-1,0,0],[0,0,1]])
    dist, ids = cast_scene(s, [1,.8,1], rays, iter_boxes(s,expand_children=True))
    assert np.allclose(dist, [np.inf,1,1.7])
    assert list(ids) == ["","wall_west","ceiling"]
    opening["extents"] = dict(y=opening.pop("u_range"), z=opening.pop("z_range"))
    assert np.allclose(cast_scene(s, [1,.8,1], rays, [])[0], dist)
    for name, q in shell_quads(s):
        normal = np.cross(q[1]-q[0], q[2]-q[0])
        assert normal @ (np.array(s["room"]["size_xyz"])/2-q[0]) > 0, name


def test_mesh():
    s = v0()
    verts, faces, mat_ids, mats, ids = scene_to_triangles(s)
    leaf_ids = {b["id"] for b in iter_boxes(s, expand_children=True) if b["solid"]}
    assert len(faces) == 12 * (len(leaf_ids)+1)
    assert set(ids) == leaf_ids.union(SURFACES)
    assert all(sum(ids == name) == 12 for name in leaf_ids)
    assert len(set(mat_ids)) == len(mats)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/"mesh"
        build_mesh(s, out)
        lines = out.with_suffix(".obj").read_text().splitlines()
        assert {line[2:] for line in lines if line.startswith("g ")} == set(ids)
        assert sum(line.startswith("f ") and len(line.split()) == 5 for line in lines) == 6
        mtl = out.with_suffix(".mtl").read_text()
        assert mtl.count("# emissivity ") == mtl.count("# lidar_reflectivity ") == len(mats)
        blob = out.with_suffix(".glb").read_bytes()
        assert struct.unpack_from("<4sII",blob) == (b"glTF",2,len(blob))
        n, tag = struct.unpack_from("<I4s",blob,12)
        assert tag == b"JSON" and n % 4 == 0
        tree = json.loads(blob[20:20+n])
        size, tag = struct.unpack_from("<I4s",blob,20+n)
        binary = blob[28+n:]
        assert tag == b"BIN\0" and size == len(binary) == tree["buffers"][0]["byteLength"]
        assert [m["name"] for m in tree["meshes"]] == mats
        for i, mesh in enumerate(tree["meshes"]):
            prim = mesh["primitives"][0]
            attr = prim["attributes"]
            def read(index, dtype, width):
                access = tree["accessors"][index]
                view = tree["bufferViews"][access["bufferView"]]
                return np.frombuffer(binary,dtype,access["count"]*width,
                                     view["byteOffset"]).reshape(-1,width)
            pos = read(attr["POSITION"],"<f4",3)
            ix = read(prim["indices"],"<u4",1).ravel()
            expected = verts[faces[mat_ids == i]].reshape(-1,3)
            assert np.allclose(pos[ix], expected[:,[0,2,1]] * [1,1,-1])
            assert np.allclose(np.linalg.norm(read(attr["NORMAL"],"<f4",3),axis=1),1)


def test_preview():
    s = v0()
    image = render(s,cameras(s)[0])
    assert image.size[0] <= 525 and image.size[1] <= 700
    assert len(np.unique(np.asarray(image).reshape(-1,3),axis=0)) > 5
    triangle = np.array([[-1,-1,-1],[1,-1,1],[0,1,1]])
    clipped = np.asarray(near_clip(triangle))
    assert clipped.shape == (4,3) and clipped[:,2].min() >= .01-1e-12


if __name__ == "__main__":
    tests = [f for name,f in globals().copy().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} tests passed")
