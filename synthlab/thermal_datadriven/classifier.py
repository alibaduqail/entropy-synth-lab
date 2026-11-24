"""Real-vs-synthetic classifier on the validation sets (50 % accuracy = indistinguishable).

    python -m synthlab.thermal_datadriven.classifier

Per frame features: log radial power spectrum, median-removed temperature histogram, frame-to-frame
difference SD, row/column mean SD.  Gradient boosting, GROUPED cross-validation (all frames of a
clip stay in one fold).  Positive class = held-out real (real_test); negative = synth.
Floor: the same test with real_build (real trials the generator learned from) vs real_test.
Also reported: a 'noise-only' feature set (diff SD + high-frequency power) to see where the
residual difference lives.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from . import metrics as M
from .valsets import CONDS, OUT as SETS
from .validate import RES

EDGES = np.linspace(-3, 12, 31)


def feats(frames, nb=20):
    c = frames.astype(np.float32) * 0.01
    out = []
    for i in range(1, len(c)):
        x = c[i]
        r = x - np.median(x)
        f, p = M.radial_power_spectrum(x[None] - 273.15)
        d = x - c[i - 1]
        hist = np.histogram(np.clip(r, -3, 12), EDGES)[0] / r.size
        out.append(np.concatenate([np.log10(np.maximum(p[:nb], 1e-12)), np.log10(hist + 1e-6),
                                   [d.std(), np.diff(x, axis=1).std(), np.diff(x, axis=0).std(),
                                    x.mean(axis=1).std(), x.mean(axis=0).std()]]))
    return np.array(out)


NOISE_COLS = list(range(10, 20)) + [50, 51, 52]      # high-frequency PSD + diff SDs


def acc(A, B, ga, gb, cols=None, seed=0):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    g = np.r_[ga, gb + 100000]
    if cols is not None:
        X = X[:, cols]
    accs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=seed)
        clf.fit(X[tr], y[tr])
        pr = clf.predict(X[te])
        # balanced accuracy
        accs.append(0.5 * ((pr[y[te] == 0] == 0).mean() + (pr[y[te] == 1] == 1).mean()))
    return float(np.mean(accs))


def load(cond, kind):
    z = np.load(SETS / f"{cond}__{kind}.npz")
    f, g = z["frames"], z["group"]
    keep = np.concatenate([[False], g[1:] == g[:-1]])      # first frame of each clip has no predecessor
    F = np.vstack([feats(f[g == u]) for u in np.unique(g)])
    G = np.concatenate([np.full((g == u).sum() - 1, u) for u in np.unique(g)])
    return F, G


def main():
    res = {}
    for cond in CONDS:
        T, gt = load(cond, "real_test"); S, gs = load(cond, "synth"); Bd, gb = load(cond, "real_build")
        res[cond] = {"synth_vs_real_test": acc(S, T, gs, gt),
                     "real_build_vs_real_test": acc(Bd, T, gb, gt),
                     "synth_vs_real_test_noise_features_only": acc(S, T, gs, gt, NOISE_COLS),
                     "real_build_vs_real_test_noise_features_only": acc(Bd, T, gb, gt, NOISE_COLS),
                     "n_frames_real_test": int(len(T)), "n_frames_synth": int(len(S))}
        print(cond, json.dumps(res[cond]), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "classifier.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
