"""LiDAR geometric entropy HL, reimplemented from the method the LiDAR teammate uses
(dos Santos et al. 2019, Eqs. 6-9, adapted to time-windowed Mid-360 data):

  1 s window of frames -> ROI crop -> voxel-grid centroid downsample (leaf 5 cm)
  -> radius-limited kNN (r = 0.20 m, k_max 150, k_min 10, scipy cKDTree)
  -> local covariance -> eigenvalues l1 >= l2 >= l3
  -> L = (l1-l2)/l1, P = (l2-l3)/l1, S = l3/l1
  -> H_DS    = -sum x ln x over (L, P, S)      (0 for line, plane AND isotropic; max ln 3)
     H_eigen = -sum e ln e, e = l / sum(l)     (0 line, ln 2 plane, ln 3 isotropic)
Every stage is timed separately so the Jetson cost runner can attribute cost.
"""
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12
LN3 = math.log(3.0)


@dataclass
class HLParams:
    leaf: float = 0.05
    radius: float = 0.20
    k_max: int = 150
    k_min: int = 10
    s_thresh: float = 0.30
    p_thresh: float = 0.70
    # ROI in the SENSOR frame; z range cuts the ceiling like the teammate's lab ROI does
    roi: tuple = (-4.0, 4.0, -4.0, 4.0, -0.30, 1.30)
    min_voxels: int = 200
    workers: int = -1


def roi_crop(xyz, roi):
    m = ((xyz[:, 0] > roi[0]) & (xyz[:, 0] < roi[1]) & (xyz[:, 1] > roi[2]) & (xyz[:, 1] < roi[3])
         & (xyz[:, 2] > roi[4]) & (xyz[:, 2] < roi[5]))
    return m


def voxel_downsample(xyz, leaf):
    """Centroid of the points in each occupied leaf-sized voxel."""
    if xyz.shape[0] == 0:
        return xyz.astype(np.float64)
    keys = np.floor(xyz / leaf).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inv = np.unique(flat, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    out = np.empty((cnt.size, 3))
    for a in range(3):
        out[:, a] = np.bincount(inv, weights=xyz[:, a]) / cnt
    return out


def knn(P, radius, k_max, workers=-1):
    """Rectangular radius-limited kNN. Returns (valid mask (N,k), idx (N,k))."""
    tree = cKDTree(P)
    d, idx = tree.query(P, k=min(k_max, max(P.shape[0], 1)), distance_upper_bound=radius,
                        workers=workers)
    if d.ndim == 1:
        d, idx = d[:, None], idx[:, None]
    valid = np.isfinite(d)
    return valid, np.where(valid, idx, 0)


def eigenvalues(P, valid, idx, k_min, chunk=10000):
    """Covariance eigenvalues (descending) for points with >= k_min neighbours."""
    lam_all, nnb_all = [], []
    for s in range(0, P.shape[0], chunk):
        v = valid[s:s + chunk]
        w = v.astype(np.float64)
        n = w.sum(1)
        keep = n >= k_min
        if not keep.any():
            continue
        nb, w, n = P[idx[s:s + chunk][keep]], w[keep], n[keep]
        mean = (nb * w[..., None]).sum(1) / n[:, None]
        X = (nb - mean[:, None, :]) * w[..., None]
        cov = np.einsum("nki,nkj->nij", X, X) / (n[:, None, None] - 1.0)
        lam_all.append(np.linalg.eigvalsh(cov)[:, ::-1])
        nnb_all.append(n)
    if not lam_all:
        return np.zeros((0, 3)), np.zeros(0)
    return np.clip(np.concatenate(lam_all), 0.0, None), np.concatenate(nnb_all)


def _plogp(p):
    return np.where(p > EPS, p * np.log(np.clip(p, EPS, None)), 0.0)


def shape_entropies(lam):
    """(H_DS, H_eigen, L, P, S) per point from descending eigenvalues; degenerate rows dropped."""
    lam = np.asarray(lam, float)
    lam = lam[lam[:, 0] > EPS]
    if lam.shape[0] == 0:
        z = np.zeros(0)
        return z, z, z, z, z
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    L = np.clip((l1 - l2) / l1, 0, 1)
    Pl = np.clip((l2 - l3) / l1, 0, 1)
    S = np.clip(l3 / l1, 0, 1)
    h_ds = -(_plogp(L) + _plogp(Pl) + _plogp(S))
    e = lam / np.clip(lam.sum(1), EPS, None)[:, None]
    h_e = -(_plogp(e[:, 0]) + _plogp(e[:, 1]) + _plogp(e[:, 2]))
    return h_ds, h_e, L, Pl, S


def compute_hl(xyz, tag=None, params=None, match_nvoxel=None, rng=None, timing=True):
    """HL metrics for one accumulated window. `match_nvoxel`: control option - randomly
    subsample the voxelised cloud to exactly this many points before the kNN stage."""
    hp = params or HLParams()
    t = [time.perf_counter()]
    xyz = np.asarray(xyz, np.float64)
    n_raw = xyz.shape[0]
    m = roi_crop(xyz, hp.roi)
    xyz = xyz[m]
    n_roi = xyz.shape[0]
    t.append(time.perf_counter())
    P = voxel_downsample(xyz, hp.leaf)
    n_vox_full = P.shape[0]
    if match_nvoxel is not None and P.shape[0] > match_nvoxel:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(P.shape[0], match_nvoxel, replace=False)]
    t.append(time.perf_counter())
    out = dict(n_raw=n_raw, n_roi=n_roi, n_voxel=P.shape[0], n_voxel_full=n_vox_full,
               voxel_retention=(n_vox_full / n_roi) if n_roi else 0.0)
    if tag is not None:
        tg = np.asarray(tag)[m]
        out["frac_tag_particle"] = float((((tg >> 2) & 3) != 0).mean()) if tg.size else 0.0
        out["frac_tag_glue"] = float(((tg & 3) != 0).mean()) if tg.size else 0.0
    if P.shape[0] < hp.min_voxels:
        out["valid"] = False
        return out
    valid, idx = knn(P, hp.radius, hp.k_max, hp.workers)
    t.append(time.perf_counter())
    lam, n_nb = eigenvalues(P, valid, idx, hp.k_min)
    h_ds, h_e, L, Pl, S = shape_entropies(lam)
    t.append(time.perf_counter())
    if h_ds.size < 100:
        out["valid"] = False
        return out
    nn_all = valid.sum(1)
    out.update(valid=True, H_DS=float(h_ds.mean()), H_eigen=float(h_e.mean()),
               H_DS_std=float(h_ds.std()), H_eigen_std=float(h_e.std()),
               mean_L=float(L.mean()), mean_P=float(Pl.mean()), mean_S=float(S.mean()),
               frac_high_S=float((S > hp.s_thresh).mean()),
               frac_high_P=float((Pl > hp.p_thresh).mean()),
               n_valid=int(h_ds.size), mean_neighbors=float(n_nb.mean()),
               mean_neighbors_all=float(nn_all.mean()), sum_neighbors=int(nn_all.sum()),
               frac_capped=float((nn_all >= hp.k_max).mean()))
    if timing:
        out.update(ms_roi=(t[1] - t[0]) * 1e3, ms_voxel=(t[2] - t[1]) * 1e3,
                   ms_knn=(t[3] - t[2]) * 1e3, ms_eig=(t[4] - t[3]) * 1e3,
                   ms_total=(t[4] - t[0]) * 1e3)
    return out


def hl_params_dict(hp):
    return asdict(hp)
