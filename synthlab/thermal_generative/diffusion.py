"""Cosine-schedule v-prediction diffusion with DDIM sampling and per-condition
classifier-free guidance. Model space z = ZSCALE * x (x from mapping.counts_to_x)."""
import math

import torch

from .models import COND_SIZES

ZSCALE = 2.0
Z_MIN, Z_MAX = -3.0, 16.5  # hard clamp on predicted x0 (x in [-1.5, 8.25], i.e. 17.5 C .. ~170 C)
T_STEPS = 1000


def alpha_bar(t):
    """t in [0,1] -> cumulative alpha (cosine schedule, Nichol & Dhariwal 2021)."""
    s = 0.008
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
    return (f / f0).clamp(1e-5, 1.0)


def q_sample(z0, t, noise):
    ab = alpha_bar(t)[:, None, None, None]
    return ab.sqrt() * z0 + (1 - ab).sqrt() * noise


def v_target(z0, t, noise):
    ab = alpha_bar(t)[:, None, None, None]
    return ab.sqrt() * noise - (1 - ab).sqrt() * z0


def drop_cond(cond, p_each=0.15, p_all=0.1, generator=None):
    """Classifier-free-guidance dropout: null each condition independently, or all at once."""
    sizes = torch.tensor(COND_SIZES, device=cond.device)
    r = torch.rand(cond.shape, device=cond.device, generator=generator)
    ra = torch.rand(cond.shape[0], 1, device=cond.device, generator=generator)
    drop = (r < p_each) | (ra < p_all)
    return torch.where(drop, sizes.expand_as(cond), cond)


def loss_fn(model, z0, cond):
    t = torch.rand(z0.shape[0], device=z0.device)
    noise = torch.randn_like(z0)
    zt = q_sample(z0, t, noise)
    pred = model(zt, t * T_STEPS, drop_cond(cond))
    return torch.mean((pred.float() - v_target(z0, t, noise)) ** 2)


@torch.no_grad()
def ddim_sample(model, cond, steps=50, guidance=1.0, temperature=1.0, eta=0.0, generator=None,
                amp=True, shape=(120, 160)):
    """cond: LongTensor (B,3). guidance w: v = v_null + w (v_cond - v_null); w=1 is plain
    conditional sampling, w=0 unconditional. temperature scales the initial noise (and the
    eta noise), <1 gives smoother / more typical frames, >1 noisier ones."""
    dev = cond.device
    b = cond.shape[0]
    z = torch.randn(b, 1, *shape, device=dev, generator=generator) * temperature
    ts = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    null = torch.tensor(COND_SIZES, device=dev).expand_as(cond)
    for i in range(steps):
        t, tn = ts[i].expand(b), ts[i + 1].expand(b)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            if guidance == 1.0:
                v = model(z, t * T_STEPS, cond)
            else:
                vc, vu = model(torch.cat([z, z]), torch.cat([t, t]) * T_STEPS, torch.cat([cond, null])).chunk(2)
                v = vu + guidance * (vc - vu)
        v = v.float()
        ab, abn = alpha_bar(t)[:, None, None, None], alpha_bar(tn)[:, None, None, None]
        z0 = (ab.sqrt() * z - (1 - ab).sqrt() * v).clamp(Z_MIN, Z_MAX)
        eps = (z - ab.sqrt() * z0) / (1 - ab).sqrt()
        sig = eta * ((1 - abn) / (1 - ab)).sqrt() * (1 - ab / abn).sqrt()
        z = abn.sqrt() * z0 + (1 - abn - sig ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and i < steps - 1:
            z = z + sig * temperature * torch.randn(z.shape, device=dev, generator=generator)
    return z0
