"""Held-out evaluation of the generative models (run on the Jetson under `jetson.sh heavy`).

    python3 -m synthlab.thermal_generative.evaluate gen --ckpt ddpm_main      # sample eval sets
    python3 -m synthlab.thermal_generative.evaluate metrics --ckpt ddpm_main  # -> out/eval/<ckpt>_metrics.json
    python3 -m synthlab.thermal_generative.evaluate speed --ckpt ddpm_main    # sampling speed / power
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import metrics as Q
from .data import Pack
from .mapping import bie_image, counts_to_x, yolo_frame
from .power import PowerLog
from .sample import HOME, generate, load_model, make_cond

OUT = f"{HOME}/out/eval"
# (session, person, lamp)
REAL_CELLS = [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)]
NOVEL_CELLS = [(0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 1), (2, 1, 1)]


def cell_name(c):
    return f"s{c[0]}p{c[1]}l{c[2]}"


def gen_path(ckpt, c, tag):
    return f"{OUT}/{ckpt}_{cell_name(c)}_{tag}.npy"


def cmd_gen(a):
    model, kind, _ = load_model(a.ckpt)
    jobs = [(c, "A", 1000, a.n) for c in REAL_CELLS] + [(c, "B", 2000, a.n) for c in REAL_CELLS]
    jobs += [(c, "B", 3000, a.n // 2) for c in NOVEL_CELLS]
    t_end = time.time() + a.minutes * 60
    for c, tag, seed, n in jobs:
        p = gen_path(a.ckpt, c, tag)
        if os.path.exists(p):
            continue
        if time.time() > t_end:
            print("time budget reached; rerun to continue")
            return
        cond = make_cond(n, c[1], c[2], c[0])
        frames, secs = generate(model, kind, cond, a.steps, a.guidance, 1.0, 0.0, seed + 7 * c[0] + 3 * c[1] + c[2])
        np.save(p, frames)
        print(cell_name(c), tag, n, f"{n / secs:.2f} frames/s", flush=True)
    print("gen complete")


class SmallCNN(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                               nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                               nn.Linear(64, n_out))

    def forward(self, x):
        return self.f(x)


def feats(frames, view):
    """view 'u8' = what BIE/YOLO see (fixed 10-50 C, 8 bit); 'rad' = full radiometric x."""
    if view == "u8":
        return torch.from_numpy(Q.to_u8_batch(frames).astype(np.float32) / 255.0 * 8.0 - 3.0)[:, None]
    return torch.from_numpy(counts_to_x(frames))[:, None]


def fit_classifier(xtr, ytr, xte, yte, steps=400, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    net = SmallCNN(int(ytr.max()) + 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    xtr, ytr = xtr.to(dev), torch.as_tensor(ytr).long().to(dev)
    for _ in range(steps):
        i = torch.randint(0, len(xtr), (64,), device=dev)
        loss = nn.functional.cross_entropy(net(xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(xte[i:i + 256].to(dev)).argmax(1).cpu() for i in range(0, len(xte), 256)]).numpy()
    return float(np.mean(pred == np.asarray(yte))), pred


def yolo_stats(model, frames):
    n_person, n_det, hit = 0, 0, 0
    for f in frames:
        r = model.predict(source=yolo_frame(f), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.array([])
        n_det += len(cls)
        k = int(np.sum(cls == 0))
        n_person += k
        hit += k > 0
    n = len(frames)
    return dict(person_frame_rate=hit / n, persons_per_frame=n_person / n, detections_per_frame=n_det / n)


def set_stats(frames):
    ht = Q.ht_values(frames)
    return dict(n=int(len(frames)), ht_mean=float(ht.mean()), ht_sd=float(ht.std()),
                noise_mk_median=float(np.median(Q.highpass_noise_mk(frames))),
                lamp_hot_fraction=Q.lamp_hot_fraction(frames), warm_fraction=Q.warm_fraction(frames),
                mean_c=float(Q.counts_to_celsius(frames).mean()),
                pixel_sd_c_median=float(np.median(Q.pixel_sd_map_c(frames)))), ht


def sub(idx, n, seed=0):
    idx = np.where(idx)[0]
    if len(idx) <= n:
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def cmd_metrics(a):
    from ultralytics import YOLO
    pack = Pack(a.pack)
    yolo = YOLO(f"{HOME}/yolov8n.pt")
    tr_all = np.where(pack.mask(split="train"))[0]
    train_frames, train_trial = pack.frames[tr_all], pack.trial[tr_all]
    res = dict(ckpt=a.ckpt, cells={}, novel={})
    curves = {}
    real_tr_x, real_te_x, gen_a_x, gen_b_x = [], [], [], []
    for c in REAL_CELLS:
        name = cell_name(c)
        if not os.path.exists(gen_path(a.ckpt, c, "B")):
            continue
        gen = np.load(gen_path(a.ckpt, c, "B"))
        gen_a = np.load(gen_path(a.ckpt, c, "A"))
        te = pack.frames[sub(pack.mask("test", c[1], c[2], c[0]), 256)]
        tr_i = sub(pack.mask("train", c[1], c[2], c[0]), 256, 1)
        tr = pack.frames[tr_i]
        r = dict(session=c[0], person=c[1], lamp=c[2])
        for key, fr in (("real_test", te), ("real_train", tr), ("gen", gen)):
            s, ht = set_stats(fr)
            s.update(yolo_stats(yolo, fr[:128]))
            r[key] = s
            curves[f"{name}_{key}_ht"] = ht
            curves[f"{name}_{key}_hist"] = Q.temp_hist(fr)[0]
            curves[f"{name}_{key}_spec"] = Q.radial_spectrum(fr)[1]
            curves[f"{name}_{key}_noise"] = Q.highpass_noise_mk(fr)
        r["ht_ks_gen_vs_test"] = Q.ks_stat(curves[f"{name}_gen_ht"], curves[f"{name}_real_test_ht"])
        r["ht_ks_train_vs_test"] = Q.ks_stat(curves[f"{name}_real_train_ht"], curves[f"{name}_real_test_ht"])
        r["w1_c_gen_vs_test"] = Q.wasserstein_c(gen, te)
        r["w1_c_train_vs_test"] = Q.wasserstein_c(tr, te)
        r["spec_gen_vs_test"] = Q.spectrum_distance(curves[f"{name}_gen_spec"], curves[f"{name}_real_test_spec"])
        r["spec_train_vs_test"] = Q.spectrum_distance(curves[f"{name}_real_train_spec"], curves[f"{name}_real_test_spec"])
        # memorisation: nearest training frame (any cell), RMSE in deg C
        d_gen, i_gen = Q.nn_rmse_c(gen, train_frames)
        d_te, _ = Q.nn_rmse_c(te, train_frames)
        d_tr, _ = Q.nn_rmse_c(tr, train_frames, exclude_same=(pack.trial[tr_i], train_trial))
        curves[f"{name}_nn_gen"], curves[f"{name}_nn_test"], curves[f"{name}_nn_train"] = d_gen, d_te, d_tr
        curves[f"{name}_nn_gen_idx"] = tr_all[i_gen]
        r["nn_rmse_c_median"] = dict(gen_to_train=float(np.median(d_gen)), test_to_train=float(np.median(d_te)),
                                     train_to_other_train_trials=float(np.median(d_tr)))
        r["nn_gen_below_p5_of_test"] = float(np.mean(d_gen < np.percentile(d_te, 5)))
        r["nn_gen_below_median_of_test"] = float(np.mean(d_gen < np.median(d_te)))
        res["cells"][name] = r
        real_tr_x.append(tr); real_te_x.append(te); gen_a_x.append(gen_a); gen_b_x.append(gen)
        print(name, json.dumps({k: r[k] for k in ("nn_rmse_c_median", "ht_ks_gen_vs_test")}), flush=True)
    for c in NOVEL_CELLS:
        p = gen_path(a.ckpt, c, "B")
        if not os.path.exists(p):
            continue
        gen = np.load(p)
        s, ht = set_stats(gen)
        s.update(yolo_stats(yolo, gen))
        d, _ = Q.nn_rmse_c(gen, train_frames)
        s["nn_rmse_c_median_gen_to_train"] = float(np.median(d))
        res["novel"][cell_name(c)] = dict(session=c[0], person=c[1], lamp=c[2], gen=s)
        curves[f"{cell_name(c)}_gen_ht"] = ht
    # real-vs-generated classifier: train on real-train vs gen set A, test on held-out-trial real vs gen set B
    rtr, rte, ga, gb = map(np.concatenate, (real_tr_x, real_te_x, gen_a_x, gen_b_x))
    res["real_vs_gen_accuracy"] = {}
    for view in ("u8", "rad"):
        accs = []
        for seed in range(3):
            xtr = torch.cat([feats(rtr, view), feats(ga, view)])
            ytr = np.r_[np.zeros(len(rtr)), np.ones(len(ga))]
            xte = torch.cat([feats(rte, view), feats(gb, view)])
            yte = np.r_[np.zeros(len(rte)), np.ones(len(gb))]
            acc, pred = fit_classifier(xtr, ytr, xte, yte, seed=seed)
            accs.append(dict(acc=acc, real_recall=float(np.mean(pred[:len(rte)] == 0)),
                             gen_recall=float(np.mean(pred[len(rte):] == 1))))
        res["real_vs_gen_accuracy"][view] = accs
    # condition fidelity with a classifier trained on real frames only (person / session)
    all_tr = sub(pack.mask(split="train"), 3000)
    all_te = sub(pack.mask(split="test"), 1500)
    res["condition_classifier"] = {}
    for lab, arr in (("person", pack.person), ("session", pack.session)):
        acc_real, _ = fit_classifier(feats(pack.frames[all_tr], "u8"), arr[all_tr], feats(pack.frames[all_te], "u8"), arr[all_te], steps=600)
        y_gen = np.concatenate([np.full(len(g), c[1] if lab == "person" else c[0]) for c, g in zip(REAL_CELLS, gen_b_x)])
        acc_gen, _ = fit_classifier(feats(pack.frames[all_tr], "u8"), arr[all_tr], feats(gb, "u8"), y_gen, steps=600)
        res["condition_classifier"][lab] = dict(acc_on_real_heldout=acc_real, acc_on_generated=acc_gen)
    with open(f"{OUT}/{a.ckpt}_metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(f"{OUT}/{a.ckpt}_curves.npz", **curves)
    print(json.dumps(res["real_vs_gen_accuracy"]), json.dumps(res["condition_classifier"]))


def cmd_speed(a):
    model, kind, ck = load_model(a.ckpt)
    rows = []
    cond = make_cond(64, 1, 1, 0)
    generate(model, kind, cond[:32], steps=2)
    grid = [(s, g) for s in (10, 20, 50, 100) for g in (1.0, 2.0)] if kind == "ddpm" else [(1, 1.0)]
    for steps, g in grid:
        with PowerLog(None, interval_ms=500) as pw:
            _, secs = generate(model, kind, cond if steps <= 50 else cond[:32], steps=steps, guidance=g, seed=5)
            n = len(cond) if steps <= 50 else 32
            if secs < 3.0:  # make the window long enough for tegrastats
                reps = int(3.0 / max(secs, 1e-3)) + 1
                t = 0.0
                for _ in range(reps):
                    t += generate(model, kind, cond, steps=steps, guidance=g, seed=5)[1]
                secs, n = t, reps * len(cond)
            w = pw.mean_w()
        rows.append(dict(kind=kind, steps=steps, guidance=g, batch=32, frames=n, seconds=secs, frames_per_s=n / secs,
                         mean_vdd_in_w=w, joules_per_frame=w * secs / n))
        print(rows[-1], flush=True)
    with open(f"{OUT}/{a.ckpt}_speed.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "metrics", "speed"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pack", default=f"{HOME}/data/pack.npz")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--minutes", type=float, default=16.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(gen=cmd_gen, metrics=cmd_metrics, speed=cmd_speed)[a.cmd](a)
