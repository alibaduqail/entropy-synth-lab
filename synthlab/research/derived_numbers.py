"""Small derived numbers quoted in docs/research/*.md. Run: python3 -m synthlab.research.derived_numbers"""
import json
import math
from pathlib import Path

import numpy as np

SEED = 20261123


def p840_kl(f_ghz, t_k):
    """ITU-R P.840-9 cloud/fog specific attenuation coefficient, (dB/km)/(g/m^3)."""
    th = 300.0 / t_k
    e0 = 77.66 + 103.3 * (th - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    fp = 20.20 - 146 * (th - 1) + 316 * (th - 1) ** 2
    fs = 39.8 * fp
    epp = f_ghz * (e0 - e1) / (fp * (1 + (f_ghz / fp) ** 2)) + f_ghz * (e1 - e2) / (fs * (1 + (f_ghz / fs) ** 2))
    ep = (e0 - e1) / (1 + (f_ghz / fp) ** 2) + (e1 - e2) / (1 + (f_ghz / fs) ** 2) + e2
    eta = (2 + ep) / epp
    return 0.819 * f_ghz / (epp * (1 + eta ** 2))


def shannon(block):
    hist = np.bincount(block.astype(np.int64).ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def bie(img, bs=15):
    """Same formula as the original project's experimentA.block_image_entropy (alpha=1)."""
    v = np.clip(img, 0, 255).astype(float)
    ents, sds, mods = [], [], []
    for r in range(0, v.shape[0], bs):
        for c in range(0, v.shape[1], bs):
            b = v[r:r + bs, c:c + bs]
            m = (b.max() - b.min()) / 255.0
            mods.append(-m * math.log(m) if 0 < m < 1 else 0.0)
            ents.append(shannon(np.rint(b)))
            sds.append(b.std())
    adp = 1 - abs(v.mean() - 127.5) / 127.5
    out = dict(adp=adp, mod=float(np.mean(mods)), block_entropy=float(np.mean(ents)),
               global_sd=float(v.std()), block_sd=float(np.mean(sds)))
    out["bie"] = adp * (out["mod"] / (1 + out["block_entropy"])) * (out["global_sd"] / (1 + out["block_sd"]))
    out["global_shannon"] = shannon(np.rint(v))
    return {k: round(float(x), 5) for k, x in out.items()}


def main():
    res = {"seed": SEED}
    res["p840_kl_dB_per_km_per_g_m3"] = {f"{f}GHz_{int(t - 273.15)}C": round(p840_kl(f, t), 3)
                                         for f in (60, 64, 77) for t in (293.15, 273.15)}
    kl = p840_kl(60, 293.15)
    res["radar_fog_loss_dB_8m_roundtrip"] = {f"M={m}g_m3": round(kl * m * 0.008, 4) for m in (0.05, 0.5, 5.0)}
    res["lidar_two_way_loss"] = {f"alpha={a}": {f"R={r}m": round(math.exp(-2 * a * r), 4) for r in (2, 4)}
                                 for a in (0.06, 0.15, 0.30, 0.60, 1.0)}
    res["alpha_from_MOR_ln20"] = {f"MOR={m}m": round(math.log(20) / m, 4) for m in (600, 300, 150, 100, 50, 20, 10, 5, 3)}
    # Lepton PSF: Gaussian sigma (px) giving MTF 0.51 at Nyquist/2 = 0.25 cycles/px
    s_lens = math.sqrt(-math.log(0.51) / (2 * math.pi ** 2 * 0.25 ** 2))
    sinc = math.sin(math.pi * 0.25) / (math.pi * 0.25)
    s_sys = math.sqrt(-math.log(0.51 * sinc) / (2 * math.pi ** 2 * 0.25 ** 2))
    res["lepton_psf"] = dict(sigma_lens_px=round(s_lens, 3), pixel_sinc_at_quarter=round(sinc, 3),
                             sigma_system_px=round(s_sys, 3))
    res["ina3221_vdd_in_step_mW"] = round(8 * 4.936, 1)
    # Fog extinction bridge: visible/NIR geometric-optics vs LWIR Chylek-type linear law (k assumed 0.13-0.15 m^2/g)
    res["fog_alpha_per_m_LWC_1g_m3"] = {f"r_eff={r}um": dict(nir=round(1.5 * 1.0 / (1e6 * r * 1e-6), 3), lwir_k0p13=0.13,
                                                           lwir_k0p15=0.15) for r in (2, 5, 10)}
    # BIE vs noise on a synthetic 120x160 scene (22 C background, one 32 C blob), 8-bit map of 15..40 C
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:120, 0:160]
    temp = 22.0 + 10.0 * np.exp(-(((xx - 80) / 12.0) ** 2 + ((yy - 60) / 25.0) ** 2))
    rows = []
    for sigma_c in (0.0, 0.035, 0.1, 0.3, 1.0, 3.0):
        t = temp + rng.normal(0, sigma_c, temp.shape) if sigma_c else temp
        img = np.rint(np.clip((t - 15.0) * 255.0 / 25.0, 0, 255))
        rows.append(dict(noise_sigma_C=sigma_c, **bie(img)))
    res["bie_vs_noise"] = rows
    out = Path(__file__).resolve().parents[2] / "results" / "research" / "derived_numbers.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
