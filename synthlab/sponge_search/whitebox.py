"""White-box shortcut: gradient ascent through YOLOv8n on the number of pre-NMS candidates.

Runs ON the Jetson (GPU, `heavy` lock). Uses the mechanics of the NMS-overload attacks
(Phantom Sponges, Overload): push each anchor's best class score above the confidence
threshold, optionally shrink boxes so NMS suppresses fewer of them.

Tiers:
  pixels : every pixel of the 8-bit pipeline image is free (unconstrained upper bound)
  panel  : temperature field bg + (t_hot-bg)*sigmoid(q), q on a coarse cell grid, blurred
           with the assumed Lepton PSF, sensor noise resampled each step (plausible:
           a warm plate with a low-emissivity pattern / perforated mask in front of a lamp)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import scenes


def gaussian_kernel(sigma, torch):
    r = max(1, int(3 * sigma + 0.5))
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = k / k.sum()
    return (k[:, None] * k[None, :])[None, None], r


def main():
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", default="pixels", choices=["pixels", "panel"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--area-w", type=float, default=0.0)
    ap.add_argument("--cell", type=int, default=4)
    ap.add_argument("--bg", type=float, default=27.5)
    ap.add_argument("--t-hot", type=float, default=45.0)
    ap.add_argument("--blur", type=float, default=scenes.PSF_SIGMA_PX)
    ap.add_argument("--nedt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--ht-w", type=float, default=0.0,
                    help="penalty weight on contrast (keeps the frame flat -> low HT)")
    ap.add_argument("--amp", type=float, default=1.0,
                    help="panel: fraction of the bg..t_hot span the pattern may use")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    net = YOLO(os.path.expanduser("~/yolov8n.pt")).model.to(dev).eval().float()
    for p in net.parameters():
        p.requires_grad_(False)
    kern, rad = gaussian_kernel(args.blur, torch)
    kern = kern.to(dev)
    lo8 = (args.bg - 10.0) / 40.0 * 255.0
    span = args.t_hot - args.bg

    if args.tier == "pixels":
        q = torch.zeros(1, 1, 120, 160, device=dev).uniform_(-0.5, 0.5).requires_grad_(True)
    else:
        q = torch.zeros(1, 1, 120 // args.cell, 160 // args.cell, device=dev).uniform_(
            -1, 1).requires_grad_(True)

    def image(noise=True):
        """Differentiable 8-bit-scale pipeline image (1,1,120,160) in [0,255]."""
        if args.tier == "pixels":
            return torch.sigmoid(q) * 255.0
        field = F.interpolate(torch.sigmoid(q), size=(120, 160), mode="nearest")
        temp = args.bg + span * args.amp * field
        temp = F.conv2d(F.pad(temp, (rad,) * 4, mode="replicate"), kern)
        if noise:
            temp = temp + torch.randn_like(temp) * args.nedt
        return (temp.clamp(10, 50) - 10.0) / 40.0 * 255.0

    def scores_of(img):
        x = F.interpolate(img / 255.0, size=(480, 640), mode="bilinear", align_corners=False)
        y = net(x.repeat(1, 3, 1, 1))
        y = y[0] if isinstance(y, (tuple, list)) else y
        return y[0, 4:].amax(0), y[0, :4]

    opt = torch.optim.Adam([q], lr=args.lr)
    hist = []
    for step in range(args.steps + 1):
        img = image()
        sc, box = scores_of(img)
        soft = torch.sigmoid((sc - args.conf) / args.tau).sum()
        loss = -soft - 0.05 * sc.sum()
        if args.area_w:
            wgt = torch.sigmoid((sc - args.conf) / args.tau).detach()
            loss = loss + args.area_w * (wgt * box[2] * box[3]).sum() / (640 * 480)
        if args.ht_w:
            loss = loss + args.ht_w * img.std()
        if step % 50 == 0:
            with torch.no_grad():
                hard = torch.round(image(noise=False))
                n = int((scores_of(hard)[0] > args.conf).sum())
            hist.append({"step": step, "soft": float(soft), "n_cand": n})
            print(hist[-1], flush=True)
        if step == args.steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        if args.tier == "pixels":
            u8 = torch.round(image(False))[0, 0].cpu().numpy().astype(np.uint8)
            frames = np.repeat(scenes.u8_to_raw(u8)[None], args.n_frames, 0)
        else:
            field = F.interpolate(torch.sigmoid(q), size=(120, 160), mode="nearest")
            temp = args.bg + span * args.amp * field
            clean = F.conv2d(F.pad(temp, (rad,) * 4, mode="replicate"), kern)[0, 0].cpu().numpy()
            frames = scenes.add_noise(clean, args.n_frames, args.nedt, args.seed)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "frames.npy"), frames)
    json.dump({"generator": "sponge_search.whitebox", "parameters": vars(args), "seed": args.seed,
               "n_frames": int(len(frames)), "history": hist},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
