"""Optional cross-check: metric monocular depth (Depth Anything V2 Metric-Indoor-Small) on the room photos.

Not part of the parametric pipeline. Needs torch + transformers (not in the default env):
    uv venv --python 3.11 data_local/venv_depth
    uv pip install --python data_local/venv_depth/bin/python torch transformers pillow numpy scipy
    data_local/venv_depth/bin/python -m synthlab.room3d.mono_depth --photos-dir <dir> --out data_local/depth
Writes depth_<name>.npy (float32 metres, HxW, upright frame downscaled to --long-side) and a PNG per photo.
The model was trained on Hypersim (~60 deg FOV); on 13 mm-equivalent ultrawide frames its absolute
scale is biased, so treat the output as relative structure plus a rough scale check only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos-dir", required=True)
    ap.add_argument("--out", default="data_local/depth")
    ap.add_argument("--long-side", type=int, default=700)
    ap.add_argument("--pattern", default="photo_*.jpg")
    args = ap.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.photos_dir).glob(args.pattern)):
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        scale = args.long_side / max(im.size)
        im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)), Image.LANCZOS)
        with torch.no_grad():
            pred = model(**proc(images=im, return_tensors="pt")).predicted_depth
        depth = torch.nn.functional.interpolate(
            pred[None], size=(im.size[1], im.size[0]), mode="bicubic", align_corners=False
        )[0, 0].numpy().astype(np.float32)
        np.save(out / f"depth_{path.stem}.npy", depth)
        norm = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-6)
        Image.fromarray((255 * (1 - norm)).astype(np.uint8)).save(out / f"depth_{path.stem}.png")
        print(path.name, depth.shape, "min/median/max m:", [round(float(v), 2) for v in (depth.min(), np.median(depth), depth.max())])


if __name__ == "__main__":
    main()
