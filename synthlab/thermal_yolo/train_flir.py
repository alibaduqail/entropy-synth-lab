"""Fine-tune the stock COCO YOLOv8n on the FLIR thermal set (run on the Jetson).

usage: python3 train_flir.py <flir.yaml> <out project dir> [epochs]
Hold the shared lock:  tools/jetson.sh heavy 'python3 ... train_flir.py ...'
"""
import sys
from pathlib import Path

from ultralytics import YOLO

data, project = sys.argv[1], sys.argv[2]
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
model = YOLO(str(Path.home() / "yolov8n.pt"))  # same starting weights as the project
model.train(data=data, epochs=epochs, imgsz=640, batch=16, workers=4, device=0,
            project=project, name="flir_v8n", exist_ok=True, plots=False, seed=2026)
