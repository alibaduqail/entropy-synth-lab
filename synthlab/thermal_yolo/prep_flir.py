"""Convert the Roboflow-100 'flir-camera-objects' archive (COCO json) to YOLO layout.

Source: https://huggingface.co/datasets/Francesco/flir-camera-objects (CC-BY-4.0),
a mirror of the FLIR ADAS thermal set. Class order puts person at 0 so the shared
harness (which counts class 0 as person) needs no change.

usage: python3 prep_flir.py <extracted flir-camera-objects dir> <out dir>
"""
import json
import os
import sys
from pathlib import Path

NAMES = ["person", "bicycle", "car", "dog"]


def convert(src: Path, out: Path) -> None:
    for split_src, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        coco = json.loads((src / split_src / "_annotations.coco.json").read_text())
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        images = {im["id"]: im for im in coco["images"]}
        labels = {im_id: [] for im_id in images}
        for ann in coco["annotations"]:
            name = cat[ann["category_id"]]
            if name not in NAMES:
                continue
            im = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            labels[ann["image_id"]].append(
                "%d %.6f %.6f %.6f %.6f" % (NAMES.index(name), (x + w / 2) / im["width"],
                                            (y + h / 2) / im["height"], w / im["width"],
                                            h / im["height"]))
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_box = 0
        for im_id, im in images.items():
            stem = Path(im["file_name"]).stem
            link = out / "images" / split / im["file_name"]
            if not link.exists():
                os.symlink((src / split_src / im["file_name"]).resolve(), link)
            (out / "labels" / split / (stem + ".txt")).write_text("\n".join(labels[im_id]))
            n_box += len(labels[im_id])
        print(split, len(images), "images", n_box, "boxes")
    (out / "flir.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n%s" % (
            out.resolve(), "".join("  %d: %s\n" % (i, n) for i, n in enumerate(NAMES))))


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
