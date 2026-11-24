"""Resume the FLIR fine-tune from last.pt with a smaller batch (the batch-16 run hit a
CUDA allocator failure on the 8 GB Jetson in epoch 8).

usage: python3 resume_flir.py <.../weights/last.pt>
"""
import sys

from ultralytics import YOLO

YOLO(sys.argv[1]).train(resume=True, batch=8, workers=2)
