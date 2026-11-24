"""Which classes do the stock and FLIR-fine-tuned models report on thermal banks? (Jetson)

Same preprocessing and predict() call as the shared harness. Reports, per bank and model,
detections per frame by class and the share of frames with a person at two thresholds.

usage: python3 class_hist.py <stock.pt> <thermal.pt> <out.json> <bank dirs...>
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path.home() / "synthlab/harness"))
import thermal_pipeline as tp  # noqa: E402

src = (Path.home() / "synthlab/harness/run_thermal_bank.py").read_text()
IMGSZ = eval(re.search(r"imgsz\s*=\s*(\([^)]*\)|\[[^\]]*\]|\d+)", src).group(1))

models = {"stock": YOLO(sys.argv[1]), "thermal": YOLO(sys.argv[2])}
out = {"imgsz": IMGSZ, "banks": {}}
for bank_dir in sys.argv[4:]:
    bank = np.load(Path(bank_dir) / "frames.npy")
    frames = [tp.yolo_frame_from_thermal(tp.radiometric_bie_image(raw)) for raw in bank]
    entry = {}
    for name, model in models.items():
        counts = {0.25: collections.Counter(), 0.10: collections.Counter()}
        person = {0.25: 0, 0.10: 0}
        for frame in frames:
            r = model.predict(source=frame, imgsz=IMGSZ, conf=0.10, verbose=False)[0]
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for thr in counts:
                keep = cls[conf >= thr]
                counts[thr].update(model.names[int(c)] for c in keep)
                person[thr] += int((keep == 0).any())
        entry[name] = {("conf_%.2f" % thr): {
            "person_frame_frac": person[thr] / len(frames),
            "per_frame_by_class": {k: v / len(frames) for k, v in counts[thr].most_common()}}
            for thr in counts}
    out["banks"][Path(bank_dir).name] = entry
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
for b, e in out["banks"].items():
    for m, v in e.items():
        print(b, m, {t: (round(x["person_frame_frac"], 3),
                         {k: round(c, 2) for k, c in x["per_frame_by_class"].items()})
                     for t, x in v.items()})
