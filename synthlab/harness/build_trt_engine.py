#!/usr/bin/env python3
"""Build a TensorRT FP16 (or FP32) engine of YOLOv8n for the Jetson from an ONNX file.

The ONNX file is exported off-device with Ultralytics 8.3.226
(YOLO("yolov8n.pt").export(format="onnx", imgsz=(480, 640), opset=17)), because the Jetson
has no `onnx` Python package and system packages must not be changed. The TensorRT Python
bindings come from NVIDIA's python3-libnvinfer 10.3.0 package unpacked (not installed)
under ~/synthlab/harness/trt/root. The engine gets the small JSON header Ultralytics
expects, so YOLO("x.engine") loads it like its own exports.

  tools/jetson.sh heavy 'python3 ~/synthlab/harness/build_trt_engine.py \
      --onnx ~/synthlab/harness/models/yolov8n_480x640.onnx \
      --engine ~/synthlab/harness/models/yolov8n_fp16_480x640.engine --fp16'
"""

import argparse
import json
import sys
import time
from pathlib import Path

TRT_SITE = Path(__file__).resolve().parent / "trt" / "root" / "usr" / "lib" / "python3.10" / "dist-packages"


def add_tensorrt_path():
    if TRT_SITE.is_dir() and str(TRT_SITE) not in sys.path:
        sys.path.insert(0, str(TRT_SITE))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--weights", type=Path, default=Path("/home/jetsonnano/yolov8n.pt"),
                    help="only read for the class names")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    args = ap.parse_args()
    add_tensorrt_path()
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")
    shapes = {network.get_input(i).name: tuple(network.get_input(i).shape)
              for i in range(network.num_inputs)}
    outputs = {network.get_output(i).name: tuple(network.get_output(i).shape)
               for i in range(network.num_outputs)}
    print("inputs", shapes, "outputs", outputs, flush=True)
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            raise SystemExit("platform has no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    start = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    from ultralytics import YOLO
    names = YOLO(str(args.weights)).names
    height, width = list(shapes.values())[0][2:]
    meta = {"description": "YOLOv8n built by synthlab/harness/build_trt_engine.py",
            "stride": 32, "task": "detect", "batch": 1, "imgsz": [int(height), int(width)],
            "names": names, "channels": 3,
            "args": {"batch": 1, "half": bool(args.fp16), "dynamic": False, "nms": False},
            "tensorrt": trt.__version__, "onnx": args.onnx.name}
    blob = json.dumps(meta).encode()
    with open(args.engine, "wb") as handle:
        handle.write(len(blob).to_bytes(4, byteorder="little", signed=True))
        handle.write(blob)
        handle.write(serialized)
    print(f"wrote {args.engine} ({args.engine.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - start:.0f} s, fp16={args.fp16}")


if __name__ == "__main__":
    main()
