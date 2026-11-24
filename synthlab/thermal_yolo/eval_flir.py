"""Score the stock COCO YOLOv8n and the FLIR fine-tune on the FLIR test split (Jetson).

The stock model keeps its 80 COCO classes, so the test labels are rewritten with COCO ids
(person 0, bicycle 1, car 2, dog 16) in a sibling folder. Both are scored on the same images.

usage: python3 eval_flir.py <yolo dataset dir> <stock.pt> <thermal.pt> <out.json> [split]
"""
import json
import os
import sys
from pathlib import Path

from ultralytics import YOLO

OURS_TO_COCO = {0: 0, 1: 1, 2: 2, 3: 16}
NAMES = ["person", "bicycle", "car", "dog"]


def coco_view(root: Path, split: str, coco_names: dict) -> Path:
    view = root.parent / "yolo_coco"
    (view / "images").mkdir(parents=True, exist_ok=True)
    (view / "labels" / split).mkdir(parents=True, exist_ok=True)
    link = view / "images" / split
    if not link.exists():
        os.symlink((root / "images" / split).resolve(), link)
    for lab in (root / "labels" / split).glob("*.txt"):
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        (view / "labels" / split / lab.name).write_text(
            "\n".join(" ".join([str(OURS_TO_COCO[int(r[0])])] + r[1:]) for r in rows))
    yaml = view / "flir_coco.yaml"
    yaml.write_text("path: %s\ntrain: images/%s\nval: images/%s\nnames:\n%s" % (
        view.resolve(), split, split,
        "".join("  %d: %s\n" % (i, n) for i, n in sorted(coco_names.items()))))
    return yaml


def score(weights: str, data: str, split_key: str, ids: dict) -> dict:
    m = YOLO(weights).val(data=data, split=split_key, imgsz=640, batch=16, device=0,
                          plots=False, verbose=False, workers=2)
    out = {"all_listed_classes_map50": None}
    index = list(m.box.ap_class_index)
    for name, cid in ids.items():
        if cid in index:
            i = index.index(cid)
            out[name] = {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                         "map50": float(m.box.ap50[i]), "map50_95": float(m.box.ap[i])}
    out["speed_ms_per_image"] = dict(m.speed)
    return out


if __name__ == "__main__":
    root, stock, thermal, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    split = sys.argv[5] if len(sys.argv) > 5 else "test"
    coco_yaml = coco_view(root, split, YOLO(stock).names)
    result = {
        "split": split,
        "stock_coco": score(stock, str(coco_yaml), "val",
                            {n: OURS_TO_COCO[i] for i, n in enumerate(NAMES)}),
        "thermal_finetune": score(thermal, str(root / "flir.yaml"), split,
                                  {n: i for i, n in enumerate(NAMES)}),
    }
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
