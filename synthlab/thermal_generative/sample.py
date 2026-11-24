"""Sample thermal frame banks (hand-off format) from a trained checkpoint, on the Jetson.

    python3 -m synthlab.thermal_generative.sample --ckpt ddpm_main --bank gen_A1_lampon \
        --person 1 --lamp 1 --session 0 --n 128 --steps 50 --guidance 1.5

Writes ~/synthlab/thermal-generative/banks/<bank>/frames.npy (uint16 N,120,160 TLinear counts)
and manifest.json (generator, parameters, seed, per-frame labels and HT, sampling speed, power).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import diffusion as D
from .mapping import ht_of_counts, x_to_counts_torch
from .models import COND_SIZES, CVAE, UNet
from .power import PowerLog

HOME = os.path.expanduser("~/synthlab/thermal-generative")


def load_model(name, dev="cuda"):
    ck = torch.load(f"{HOME}/ckpt/{name}.pt", map_location=dev)
    kind = ck["args"]["model"]
    model = (UNet(base=ck["args"].get("base", 32)) if kind == "ddpm" else CVAE()).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, kind, ck


def parse_label(v, size):
    return size if str(v) in ("null", "-1") else int(v)


def make_cond(n, person, lamp, session, dev="cuda"):
    """session may be an int, 'null', or 'mix' (cycles through all sessions)."""
    c = torch.empty(n, 3, dtype=torch.long)
    c[:, 0] = parse_label(person, COND_SIZES[0])
    c[:, 1] = parse_label(lamp, COND_SIZES[1])
    if str(session) == "mix":
        c[:, 2] = torch.arange(n) % COND_SIZES[2]
    else:
        c[:, 2] = parse_label(session, COND_SIZES[2])
    return c.to(dev)


@torch.no_grad()
def generate(model, kind, cond, steps=50, guidance=1.0, temperature=1.0, eta=0.0, seed=0, batch=32):
    """Returns uint16 counts (N,120,160) and pure sampling seconds (GPU-synchronised)."""
    dev = cond.device
    g = torch.Generator(device=dev).manual_seed(seed)
    out, secs = [], 0.0
    for i in range(0, len(cond), batch):
        c = cond[i:i + batch]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if kind == "ddpm":
            z = D.ddim_sample(model, c, steps=steps, guidance=guidance, temperature=temperature, eta=eta, generator=g)
        else:
            zz = torch.randn(len(c), model.zdim, device=dev, generator=g) * temperature
            z = model.decode(zz, c).float()
        torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        out.append(x_to_counts_torch(z[:, 0] / D.ZSCALE).cpu().numpy().astype(np.uint16))
    return np.concatenate(out), secs


def write_bank(bank, frames, manifest):
    d = f"{HOME}/banks/{bank}"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/frames.npy", frames)
    with open(f"{d}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--person", default="0")
    ap.add_argument("--lamp", default="0")
    ap.add_argument("--session", default="0")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    model, kind, ck = load_model(a.ckpt)
    cond = make_cond(a.n, a.person, a.lamp, a.session)
    generate(model, kind, cond[:a.batch], steps=2, seed=1, batch=a.batch)  # warm-up (cudnn, allocator)
    with PowerLog(None, interval_ms=500) as pw:
        frames, secs = generate(model, kind, cond, a.steps, a.guidance, a.temperature, a.eta, a.seed, a.batch)
        watts = pw.mean_w()
    ht = [ht_of_counts(f) for f in frames]
    c = cond.cpu().numpy()
    man = dict(generator=f"thermal-generative/{kind}", checkpoint=a.ckpt, train_step=ck["state"]["step"],
               train_seconds=ck["state"]["train_seconds"], parameters=vars(a), seed=a.seed, n_frames=int(len(frames)),
               units="Lepton TLinear counts, 0.01 K/count", sampling_seconds=secs, frames_per_s=len(frames) / secs,
               sampling_mean_vdd_in_w=watts, joules_per_frame=watts * secs / len(frames),
               ht_mean=float(np.mean(ht)), ht_sd=float(np.std(ht)),
               labels=dict(person=c[:, 0].tolist(), n_people_nominal=c[:, 0].tolist(), lamp_on=c[:, 1].tolist(),
                           session=c[:, 2].tolist(), ht=[float(h) for h in ht]),
               note="labels are the conditioning inputs, not verified content; value == n_classes means null (unconditioned)")
    d = write_bank(a.bank, frames, man)
    print(f"{a.bank}: {len(frames)} frames, {man['frames_per_s']:.2f} frames/s, {watts:.2f} W, "
          f"HT {man['ht_mean']:.4f} +- {man['ht_sd']:.4f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
