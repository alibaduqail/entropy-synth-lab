"""Tiny conditional U-Net (for DDPM/DDIM) and a conditional VAE for 1x120x160 frames.

Conditions: person {0,1}, lamp {0,1}, session {0..S-1}; index n_classes = "null" token used
for classifier-free guidance (each condition is dropped independently during training).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COND_SIZES = (2, 2, 3)  # person, lamp, session


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class CondEmbed(nn.Module):
    def __init__(self, dim, sizes=COND_SIZES):
        super().__init__()
        self.sizes = sizes
        self.tabs = nn.ModuleList([nn.Embedding(n + 1, dim) for n in sizes])

    def forward(self, cond):
        """cond: LongTensor (B,3); value == size means null."""
        return sum(tab(cond[:, i]) for i, tab in enumerate(self.tabs))


def null_cond(batch, device, sizes=COND_SIZES):
    return torch.tensor(sizes, device=device).repeat(batch, 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.e(emb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.a = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.n(x).flatten(2).transpose(1, 2)
        y, _ = self.a(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """Patch-2 U-Net: 120x160 is pixel-unshuffled to 4x60x80, then 60x80 -> 30x40 -> 15x20
    (attention at 15x20 only), and pixel-shuffled back. The full-resolution variant
    (patch=1, mults (1,2,4,4)) ran at 0.75 it/s on the Orin Nano - too slow for tonight."""

    def __init__(self, base=32, mults=(1, 2, 4), emb=128, patch=2):
        super().__init__()
        self.emb_dim = emb
        self.patch = patch
        self.tmlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cond = CondEmbed(emb)
        chs = [base * m for m in mults]
        self.inc = nn.Conv2d(patch * patch, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = chs[0]
        for i, co in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, co, emb), ResBlock(co, co, emb)]))
            c = co
            if i < len(chs) - 1:
                self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid1 = ResBlock(c, c, emb)
        self.attn = Attn(c)
        self.mid2 = ResBlock(c, c, emb)
        self.up = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.up.append(nn.ModuleList([ResBlock(c + chs[i], chs[i], emb), ResBlock(chs[i], chs[i], emb)]))
            c = chs[i]
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, patch * patch, 3, padding=1))

    def forward(self, x, t, cond):
        e = self.tmlp(timestep_embedding(t, self.emb_dim)) + self.cond(cond)
        h = self.inc(F.pixel_unshuffle(x, self.patch))
        skips = []
        for i, (r1, r2) in enumerate(self.down):
            h = r2(r1(h, e), e)
            if i < len(self.pool):
                skips.append(h)
                h = self.pool[i](h)
        h = self.mid2(self.attn(self.mid1(h, e)), e)
        for r1, r2 in self.up:
            s = skips.pop()
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = r2(r1(torch.cat([h, s], 1), e), e)
        return F.pixel_shuffle(self.out(h), self.patch)


class CVAE(nn.Module):
    """Conditional VAE with a flat latent; decoder is conditioned, encoder sees the labels too."""

    def __init__(self, zdim=64, base=32, emb=64):
        super().__init__()
        self.zdim = zdim
        self.cond = CondEmbed(emb)
        c = [base, base * 2, base * 4, base * 4]
        enc, cin = [], 1
        for i, co in enumerate(c):
            enc += [nn.Conv2d(cin, co, 3, stride=2 if i < 3 else 1, padding=1), nn.GroupNorm(8, co), nn.SiLU()]
            cin = co
        self.enc = nn.Sequential(*enc)
        self.flat = c[-1] * 15 * 20
        self.fc_e = nn.Linear(self.flat + emb, 2 * zdim)
        self.fc_d = nn.Linear(zdim + emb, self.flat)
        self.dec = nn.Sequential(
            nn.Conv2d(c[3], c[2], 3, padding=1), nn.GroupNorm(8, c[2]), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(c[2], c[1], 3, padding=1), nn.GroupNorm(8, c[1]), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(c[1], c[0], 3, padding=1), nn.GroupNorm(8, c[0]), nn.SiLU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(c[0], c[0], 3, padding=1), nn.GroupNorm(8, c[0]), nn.SiLU(),
            nn.Conv2d(c[0], 1, 3, padding=1))

    def encode(self, x, cond):
        h = self.enc(x).flatten(1)
        mu, logvar = self.fc_e(torch.cat([h, self.cond(cond)], 1)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z, cond):
        h = self.fc_d(torch.cat([z, self.cond(cond)], 1)).view(-1, self.flat // 300, 15, 20)
        return self.dec(h)

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, cond), mu, logvar


def n_params(m):
    return sum(p.numel() for p in m.parameters())
