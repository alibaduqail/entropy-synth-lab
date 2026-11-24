"""Export a Z-up Wavefront model and a Y-up binary glTF model."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from .scene import box_corners, iter_boxes, load_scene, shell_quads


def color(material):
    name = material.lower()
    for token, rgb in (("teal", (.08,.49,.48)), ("cardboard", (.69,.52,.32)),
                       ("wood", (.57,.38,.23)), ("carpet", (.35,.36,.36)),
                       ("lcd", (.09,.11,.12)), ("glass", (.45,.79,.84)),
                       ("plastic", (.13,.14,.15)), ("electronics", (.17,.18,.19)),
                       ("metal", (.52,.55,.57)), ("fabric", (.36,.41,.39))):
        if token in name:
            return rgb
    return (.84,.83,.79)


def parts(scene):
    for name, quad in shell_quads(scene):
        yield name, name, scene["room"]["surfaces"][name], quad, [(0,1,2,3)]
    sides = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
    triangles = [tri for a,b,c,d in sides for tri in ((a,b,c), (a,c,d))]
    for b in iter_boxes(scene, expand_children=True):
        if b["solid"]:
            yield (b["id"], b["material"], b, box_corners(b["center"], b["size"], b["yaw_deg"]),
                   triangles)


def scene_to_triangles(scene):
    """Return Z-up vertices, indexed triangles, material indices, material names, face IDs.

    Shell materials use the surface ID so each wall can be styled independently.
    """
    vertices, faces, indices, materials, ids = [], [], [], [], []
    for name, mat, _, points, polygons in parts(scene):
        if mat not in materials:
            materials.append(mat)
        offset = len(vertices)
        vertices.extend(points)
        for poly in polygons:
            for i in range(1, len(poly)-1):
                faces.append([offset + poly[j] for j in (0,i,i+1)])
                indices.append(materials.index(mat))
                ids.append(name)
    return (np.asarray(vertices), np.asarray(faces), np.asarray(indices), materials, np.asarray(ids))


def material_properties(scene):
    result = {}
    for _, mat, props, _, _ in parts(scene):
        result.setdefault(mat, props)
    return result


def write_glb(path, scene, props):
    vertices, faces, indices, materials, _ = scene_to_triangles(scene)
    binary = bytearray()
    views, accessors, meshes, nodes, paints = [], [], [], [], []

    def accessor(array, component, shape, target):
        i = len(views)
        views.append(dict(buffer=0, byteOffset=len(binary), byteLength=array.nbytes, target=target))
        binary.extend(array.tobytes())
        desc = dict(bufferView=i, componentType=component, count=len(array), type=shape)
        if shape == "VEC3":
            desc.update(min=array.min(axis=0).tolist(), max=array.max(axis=0).tolist())
        accessors.append(desc)
        return i

    yup = vertices[:, [0,2,1]] * [1,1,-1]
    for i, name in enumerate(materials):
        tris = yup[faces[indices == i]]
        normals = np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pos = tris.reshape(-1,3).astype("<f4")
        attrs = dict(POSITION=accessor(pos, 5126, "VEC3", 34962),
                     NORMAL=accessor(normals.repeat(3, axis=0).astype("<f4"), 5126, "VEC3", 34962))
        ix = accessor(np.arange(len(pos), dtype="<u4"), 5125, "SCALAR", 34963)
        meshes.append(dict(name=name, primitives=[dict(attributes=attrs, indices=ix, material=i)]))
        nodes.append(dict(mesh=i, name=name))
        p = props[name]
        glass = "glass" in p["material"].lower()
        paints.append(dict(name=name, doubleSided=True, alphaMode="BLEND" if glass else "OPAQUE",
                           pbrMetallicRoughness=dict(baseColorFactor=[*color(p["material"]),
                                                    .3 if glass else 1.], metallicFactor=0.,
                                                    roughnessFactor=1.-float(p.get("specular", 0.))),
                           extras={k: p[k] for k in ("emissivity", "lidar_reflectivity",
                                                    "lidar_transmission") if k in p}))
    tree = dict(asset=dict(version="2.0"), scene=0, scenes=[dict(nodes=list(range(len(nodes))))],
                nodes=nodes, meshes=meshes, materials=paints, buffers=[dict(byteLength=len(binary))],
                bufferViews=views, accessors=accessors)
    meta = json.dumps(tree, separators=(",", ":"), allow_nan=False).encode()
    meta += b" " * (-len(meta) % 4)
    total = 28 + len(meta) + len(binary)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) +
                     struct.pack("<I4s", len(meta), b"JSON") + meta +
                     struct.pack("<I4s", len(binary), b"BIN\0") + binary)


def build_mesh(scene, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    obj, mtl, glb = [Path(str(out) + ext) for ext in (".obj", ".mtl", ".glb")]
    props = material_properties(scene)
    with mtl.open("w") as f:
        for name, p in props.items():
            f.write(f"newmtl {name}\nKd %g %g %g\n" % color(p["material"]))
            for key in ("emissivity", "lidar_reflectivity"):
                f.write(f"# {key} {p[key]}\n")
            if "glass" in p["material"].lower():
                f.write("d 0.3\n")
            f.write("\n")
    with obj.open("w") as f:
        f.write(f"mtllib {mtl.name}\n")
        offset = 1
        for name, mat, _, points, polygons in parts(scene):
            f.write(f"g {name}\nusemtl {mat}\n")
            for point in points:
                f.write("v %.10g %.10g %.10g\n" % tuple(point))
            for poly in polygons:
                f.write("f " + " ".join(str(offset+i) for i in poly) + "\n")
            offset += len(points)
    write_glb(glb, scene, props)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_mesh(load_scene(args.scene), args.out)


if __name__ == "__main__":
    main()
