"""Time-boxed, resumable training of the conditional DDPM or CVAE on the Jetson.

    python3 -m synthlab.thermal_generative.train --model ddpm --name ddpm_main --minutes 17

Each call is one chunk (<= 20 min, run under `tools/jetson.sh heavy`): load checkpoint if present,
train until the time budget, evaluate held-out loss, save. Appends to <name>_log.jsonl and
<name>_power.csv (tegrastats VDD_IN at 1 Hz).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from . import diffusion as D
from .data import Pack
from .mapping import counts_to_x
from .models import CVAE, UNet, n_params
from .power import PowerLog

HOME = os.path.expanduser("~/synthlab/thermal-generative")


def make_model(kind, base=32):
    return UNet(base=base) if kind == "ddpm" else CVAE()


def vae_loss(model, z0, cond, beta):
    rec, mu, logvar = model(z0, cond)
    mse = torch.mean((rec.float() - z0) ** 2)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)) / z0[0].numel()
    return mse + beta * kl, mse, kl


def parse_exclude(s):
    """'person=1,lamp=0' -> dict; frames matching ALL keys are removed from training."""
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))} if s else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ddpm", "vae"], default="ddpm")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pack", default=f"{HOME}/data/pack.npz")
    ap.add_argument("--minutes", type=float, default=17.0)
    ap.add_argument("--max-steps", type=int, default=10 ** 9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--beta", type=float, default=1.0, help="VAE KL weight (per-pixel normalised)")
    ap.add_argument("--exclude", default="", help="drop a label combination from training")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    ckpt_path = f"{HOME}/ckpt/{a.name}.pt"
    log_path = f"{HOME}/logs/{a.name}_log.jsonl"

    pack = Pack(a.pack)
    tr = pack.mask(split="train")
    ex = parse_exclude(a.exclude)
    if ex:
        tr &= ~pack.mask(**ex)
    te = pack.mask(split="test")
    labels = np.stack([pack.person, pack.lamp, pack.session], 1)
    xtr = torch.from_numpy(counts_to_x(pack.frames[tr]))[:, None] * D.ZSCALE
    ctr = torch.from_numpy(labels[tr])
    rng = np.random.default_rng(123)
    te_idx = rng.choice(np.where(te)[0], 256, replace=False)
    xte = torch.from_numpy(counts_to_x(pack.frames[te_idx]))[:, None] * D.ZSCALE
    cte = torch.from_numpy(labels[te_idx])
    tr_idx = rng.choice(len(xtr), 256, replace=False)

    model = make_model(a.model, a.base).to(dev)  # channels_last measured slower on the Orin Nano
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    amp = not a.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    state = dict(step=0, train_seconds=0.0, chunks=0)
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); state = ck["state"]
    torch.manual_seed(a.seed + state["chunks"])
    print(f"{a.name}: {n_params(model)/1e6:.2f} M params, train frames {len(xtr)}, resume step {state['step']}", flush=True)

    def eval_loss(x, c, reps=4):
        g = torch.Generator(device=dev).manual_seed(7)
        tot = 0.0
        ema.eval()
        with torch.no_grad():
            for r in range(reps):
                for i in range(0, len(x), 64):
                    z0, cc = x[i:i + 64].to(dev), c[i:i + 64].to(dev)
                    if a.model == "ddpm":
                        t = torch.rand(len(z0), device=dev, generator=g)
                        n = torch.randn(z0.shape, device=dev, generator=g)
                        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                            p = ema(D.q_sample(z0, t, n), t * D.T_STEPS, cc)
                        tot += float(torch.sum(torch.mean((p.float() - D.v_target(z0, t, n)) ** 2, dim=(1, 2, 3))))
                    else:
                        mu, _ = ema.encode(z0, cc)
                        tot += float(torch.sum(torch.mean((ema.decode(mu, cc) - z0) ** 2, dim=(1, 2, 3))))
        return tot / (reps * len(x))

    budget = a.minutes * 60.0
    losses, n_bad = [], 0
    with PowerLog(f"{HOME}/logs/{a.name}_power.csv", tag=f"train_chunk{state['chunks']}") as pw:
        t0 = time.time()
        model.train()
        while time.time() - t0 < budget and state["step"] < a.max_steps:
            idx = torch.randint(0, len(xtr), (a.batch,))
            z0 = xtr[idx].to(dev, non_blocking=True)
            cond = ctr[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if a.model == "ddpm":
                    loss = D.loss_fn(model, z0, cond)
                else:
                    loss, _, _ = vae_loss(model, z0, cond, a.beta)
            if not torch.isfinite(loss):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                if n_bad > 20:
                    raise RuntimeError("loss not finite; rerun with --no-amp")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            state["step"] += 1
            decay = min(0.9995, (1 + state["step"]) / (10 + state["step"]))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
            losses.append(float(loss.detach()))
            if state["step"] % 200 == 0:
                rec = dict(step=state["step"], loss=float(np.mean(losses[-200:])),
                           train_seconds=state["train_seconds"] + time.time() - t0, chunk=state["chunks"])
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if state["step"] % 1000 == 0:
                    print(rec, flush=True)
        torch.cuda.synchronize()
        chunk_s = time.time() - t0
        train_w = pw.mean_w()
        steps_chunk = len(losses)
    state["train_seconds"] += chunk_s
    state["chunks"] += 1
    rec = dict(step=state["step"], chunk_end=True, chunk=state["chunks"] - 1, chunk_seconds=chunk_s,
               steps_in_chunk=steps_chunk, it_per_s=steps_chunk / max(chunk_s, 1e-9), mean_vdd_in_w=train_w,
               train_seconds=state["train_seconds"], nonfinite=n_bad,
               eval_train=eval_loss(xtr[tr_idx], ctr[tr_idx]), eval_test=eval_loss(xte, cte),
               peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20, batch=a.batch, amp=amp,
               exclude=a.exclude, wall_total_s=time.time() - t_start)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
    torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                    scaler=scaler.state_dict(), state=state, args=vars(a)), ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)


if __name__ == "__main__":
    main()
