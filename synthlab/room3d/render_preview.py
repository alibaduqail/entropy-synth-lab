"""Render camera previews, a labelled plan and idealised first-surface sensor views."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Wedge
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .build_mesh import color, material_properties, scene_to_triangles
from .scene import box_corners, cast_scene, iter_boxes, load_scene, shell_quads


def cameras(scene):
    if "cameras" in scene:
        return scene["cameras"]
    size = np.asarray(scene["room"]["size_xyz"])
    return [dict(id=f"view_{i}", position=size * pos, look_at=size * target, up=[0,0,1],
                 hfov_deg=71., vfov_deg=56., image_wh=[525,394])
            for i, (pos, target) in enumerate([
                ((.06,.06,.62), (.6,.76,.40)), ((.94,.09,.62), (.40,.75,.40)),
                ((.07,.95,.65), (.64,.37,.35))], 1)]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("camera direction and up must define a coordinate frame")
    return np.asarray(v) / norm


def basis(direction, up):
    forward = normalize(direction)
    right = normalize(np.cross(forward, up))
    return np.stack((right, np.cross(right, forward), forward), axis=1)


def near_clip(triangle, near=.01):
    polygon = []
    for i, end in enumerate(triangle):
        start = triangle[i-1]
        if (start[2] >= near) != (end[2] >= near):
            polygon.append(start + (end-start) * ((near-start[2]) / (end[2]-start[2])))
        if end[2] >= near:
            polygon.append(end)
    return polygon


def rasterize(rgb, depth, triangle, focal, shade):
    h, w = depth.shape
    xy = triangle[:,:2] / triangle[:,2,None] * focal + [w/2,h/2]
    x0, y0 = np.maximum(0, np.floor(xy.min(axis=0))).astype(int)
    x1, y1 = np.minimum([w,h], np.ceil(xy.max(axis=0))).astype(int)
    if x0 >= x1 or y0 >= y1:
        return
    a, b, c = xy
    ab, ac = b-a, c-a
    det = ab[0]*ac[1] - ab[1]*ac[0]
    if abs(det) < 1e-12:
        return
    y, x = np.ogrid[y0:y1,x0:x1]
    x, y = x+.5-a[0], y+.5-a[1]
    u = (x*ac[1] - y*ac[0])/det
    v = (ab[0]*y - ab[1]*x)/det
    bary_a = 1-u-v
    inverse_z = bary_a/triangle[0,2] + u/triangle[1,2] + v/triangle[2,2]
    buf = depth[y0:y1,x0:x1]
    hit = (u >= -1e-10) & (v >= -1e-10) & (bary_a >= -1e-10) & (inverse_z > buf)
    buf[hit] = inverse_z[hit]
    rgb[y0:y1,x0:x1][hit] = shade


def render(scene, camera):
    overrides = camera.get("object_overrides", {})
    if overrides:
        scene = dict(scene, objects=[{**o, **overrides.get(o["id"], {})} for o in scene["objects"]])
    w, h = camera["image_wh"]
    scale = min(1., 525/w, 700/h)
    w, h = max(1, round(w*scale)), max(1, round(h*scale))
    focal = np.array([w,-h]) / (2*np.tan(np.deg2rad([camera["hfov_deg"],camera["vfov_deg"]])/2))
    origin = np.asarray(camera["position"])
    frame = basis(np.asarray(camera["look_at"])-origin, camera["up"])
    verts, faces, indices, mats, _ = scene_to_triangles(scene)
    local = (verts-origin) @ frame
    # Preview-only continuation of the photographed corridor floor beyond the model boundary.
    yy, xx = np.ogrid[:h, :w]
    ray_z = frame[2,2] + (xx+.5-w/2)/focal[0]*frame[2,0] + (yy+.5-h/2)/focal[1]*frame[2,1]
    floor_rgb = 255*np.array(color(scene["room"]["surfaces"]["floor"]["material"]))
    rgb = np.where(ray_z[...,None] < 0, floor_rgb, [221,219,212]).astype(np.uint8)
    depth = np.zeros((h,w))
    props = material_properties(scene)
    colors = np.asarray([color(props[m]["material"]) for m in mats])
    light = normalize([-.35,-.45,.82])
    for face, idx in zip(faces, indices):
        a, b, c = verts[face]
        normal = normalize(np.cross(b-a,c-a))
        lighting = .73 + .27 * (normal @ light)
        if mats[idx] in scene["room"]["surfaces"]:
            lighting = .94 + .06 * (normal @ light)
        shade = (255 * colors[idx] * lighting).astype(np.uint8)
        poly = near_clip(local[face])
        for i in range(1, len(poly)-1):
            rasterize(rgb, depth, np.asarray([poly[0],poly[i],poly[i+1]]), focal, shade)
    return Image.fromarray(rgb)


def floorplan(scene, path):
    sx, sy, sz = scene["room"]["size_xyz"]
    fig, ax = plt.subplots(figsize=(9,8), constrained_layout=True)
    for name, quad in shell_quads(scene):
        if name.startswith("wall_") and quad[:,2].min() == 0:
            endpoints = quad[quad[:,2] == 0, :2]
            ax.plot(*endpoints.T, color="#343434", lw=3)
    for b in iter_boxes(scene):
        xy = box_corners(b["center"], b["size"], b["yaw_deg"])[:4,:2]
        glass = "glass" in b["material"].lower() and "lcd" not in b["material"].lower()
        ax.add_patch(Polygon(xy, facecolor=color(b["material"]) if b["solid"] else "none",
                             alpha=.75, edgecolor="darkcyan" if glass else "#444444",
                             linestyle="--" if glass or not b["solid"] else "-"))
        ax.annotate(b["id"], b["center"][:2], fontsize=6, ha="center")
        if "hinge_xy" in b and "open_angle_deg" in b:
            span = b.get("u_range")
            radius = span[1]-span[0] if span else max(b["size"][:2])
            wall = b.get("wall", "").removeprefix("wall_")
            along = 1 if wall in ("east", "west") else 0
            start = 90. if along else 0.
            if span and abs(b["hinge_xy"][along]-span[1]) < abs(b["hinge_xy"][along]-span[0]):
                start += 180.
            swing = b["open_angle_deg"]
            if "closed_center" in b:
                hinge = b["hinge_xy"]
                closed = np.asarray(b["closed_center"][:2]) - hinge
                opened = b["center"][:2] - hinge
                start = np.rad2deg(np.arctan2(*closed[::-1]))
                cross = closed[0]*opened[1] - closed[1]*opened[0]
                if abs(cross) > 1e-9:
                    swing = np.copysign(abs(swing), cross)
            a, z = sorted((start, start+swing))
            ax.add_patch(Arc(b["hinge_xy"], 2*radius, 2*radius, theta1=a, theta2=z,
                             color="saddlebrown", linestyle=":"))
    for name, field, angle, col in [("lepton35","look_dir",57,"orangered"),
                                     ("iwr6843isk","boresight",120,"purple"),
                                     ("mid360","forward",360,"steelblue")]:
        s = scene["sensors"][name]
        xy = s["position"][:2]
        dx, dy = s[field][:2]
        heading = np.rad2deg(np.arctan2(dy,dx))
        if angle == 360:
            ax.add_patch(Circle(xy,.5,fill=False,linestyle="--",color=col))
        else:
            ax.add_patch(Wedge(xy,.65,heading-angle/2,heading+angle/2,color=col,alpha=.15))
        ax.scatter(*xy, color=col, s=18, label=name, zorder=5)
    for name, slot in scene["dynamic_slots"].items():
        if name.startswith("_"):
            continue
        xy = (slot["position"] if isinstance(slot,dict) else slot)[:2]
        ax.scatter(*xy, marker="x", color="darkgreen")
        ax.annotate(name, xy, xytext=(4,3), textcoords="offset points", fontsize=7)
    for cam in cameras(scene):
        xy = np.array(cam["position"][:2])
        d = np.asarray(cam["look_at"][:2])-xy
        dx, dy = .24*d/max(np.linalg.norm(d),1e-12)
        ax.arrow(*xy, dx, dy, width=.012, color="darkorange", length_includes_head=True)
        ax.annotate(cam["id"], xy, fontsize=6, color="saddlebrown")
    ax.plot([0,1],[-.23,-.23], color="black", lw=3)
    ax.text(.5,-.31,"1 m", ha="center", va="top")
    ax.annotate("N", (sx+.17,sy), (sx+.17,sy-.4), ha="center", arrowprops=dict(arrowstyle="->"))
    ax.set(xlim=(-.22,sx+.4), ylim=(-.48,sy+.17), aspect="equal",
           xlabel="East (m)", ylabel="North (m)", title=f"Room plan · {sx:g} × {sy:g} × {sz:g} m")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.09), ncol=3)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sensor_views(scene, path):
    boxes = list(iter_boxes(scene, expand_children=True))
    sensors = scene["sensors"]
    s = sensors["lepton35"]
    frame = basis(s["look_dir"], [0,0,1])
    y, x = np.mgrid[:120,:160]
    x = ((x+.5)/80-1)*np.tan(np.deg2rad(57/2))
    y = (1-(y+.5)/60)*np.tan(np.deg2rad(44/2))
    rays = np.stack((x,y,np.ones_like(x)),axis=-1) @ frame.T
    _, ids = cast_scene(scene, s["position"], rays, boxes)
    temperatures = {name: p["temp_c"] for name,p in scene["room"]["surfaces"].items()
                    if not name.startswith("_")}
    temperatures.update({b["id"]: b["temp_c"] for b in boxes})
    heat = np.full(ids.shape, np.nan)
    for name, temp in temperatures.items():
        heat[ids == name] = temp
    mid = sensors["mid360"]
    az, el = np.meshgrid(np.deg2rad(np.arange(-180,180)), np.deg2rad(np.arange(-7,53)))
    yaw = np.arctan2(mid["forward"][1],mid["forward"][0])
    rays = np.stack((np.cos(az+yaw)*np.cos(el), np.sin(az+yaw)*np.cos(el), np.sin(el)),axis=-1)
    ranges, _ = cast_scene(scene, mid["position"], rays, boxes, transmit_glass=True)
    fig, axes = plt.subplots(2,1,figsize=(8,8), constrained_layout=True)
    im = axes[0].imshow(heat.repeat(4,axis=0).repeat(4,axis=1), cmap="inferno",
                        interpolation="nearest")
    axes[0].set_title("Lepton 3.5 · first-surface temperature")
    axes[0].axis("off")
    finite = heat[np.isfinite(heat)]
    if finite.size and finite.min() == finite.max():
        axes[0].set_title(f"Lepton 3.5 · all visible surfaces {finite[0]:g} °C")
        fig.colorbar(im, ax=axes[0], ticks=[finite[0]], label="Temperature (°C)")
    else:
        fig.colorbar(im, ax=axes[0], label="Temperature (°C)")
    im = axes[1].imshow(np.ma.masked_invalid(ranges), origin="lower", aspect="auto",
                        extent=(-180.5,179.5,-7.5,52.5), interpolation="nearest", cmap="viridis")
    axes[1].set(title="Mid-360 · range conditional on pane transmission", xlabel="Azimuth from forward (°)",
                ylabel="Elevation (°)")
    fig.colorbar(im, ax=axes[1], label="Range (m)")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def comparison(photo, preview, path):
    with Image.open(photo) as original:
        pic = ImageOps.exif_transpose(original).convert("RGB")
    h = preview.height
    pic = pic.resize((max(1,round(pic.width*h/pic.height)),h), Image.Resampling.LANCZOS)
    combined = Image.new("RGB",(pic.width+preview.width,h+24),"white")
    combined.paste(pic,(0,24))
    combined.paste(preview,(pic.width,24))
    draw = ImageDraw.Draw(combined)
    draw.text((5,4),"Photo",fill="black")
    draw.text((pic.width+5,4),"Render",fill="black")
    combined.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--photos-dir", type=Path)
    args = parser.parse_args()
    scene = load_scene(args.scene)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for i, cam in enumerate(cameras(scene), 1):
        name = cam["id"]
        preview = render(scene, cam)
        preview.save(args.outdir / f"preview_{name}.png")
        if args.photos_dir:
            suffix = re.search(r"(\d+)$", name)
            n = int(suffix.group()) if suffix else i
            photo = args.photos_dir / f"photo_{n}.jpg"
            if not photo.exists():
                photo = args.photos_dir / f"room_{n}.jpg"
            comparison(photo, preview, args.outdir / f"compare_{name}.png")
    floorplan(scene, args.outdir / "floorplan.png")
    sensor_views(scene, args.outdir / "sensor_views.png")


if __name__ == "__main__":
    main()
