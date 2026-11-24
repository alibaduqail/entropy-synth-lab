"""LiDAR window workloads, each stage timed separately.

Workload "entropy" re-implements the method of teammate Sam's compute_hl.py (his file is
not copied here): ROI crop -> 5 cm voxel-grid centroid downsample -> radius-limited kNN
(scipy cKDTree, k_max 150, radius 0.20 m, k_min 10) -> local covariance eigenvalues
l1>=l2>=l3 -> L,P,S -> H_DS (Shannon entropy of L,P,S; dos Santos 2019) and H_eigen
(normalised-eigenvalue entropy), averaged over the window.

Workload "cluster" is a second, more data-dependent job on the same voxelised cloud:
Euclidean clustering (all point pairs closer than a tolerance, connected components) and a
person-candidate size filter. Its cost follows the number of close point pairs, which is
what airborne clutter such as fog changes.
"""

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EPS = 1e-12


@dataclass
class Params:
    roi: tuple = (-6.0, 6.0, -6.0, 6.0, -2.0, 3.0)
    z_min: float = -0.85
    leaf: float = 0.05
    radius: float = 0.20
    k_max: int = 150
    k_min: int = 10
    min_voxels: int = 200
    s_thresh: float = 0.30
    cluster_tol: float = 0.10
    cluster_min_pts: int = 30
    person_height: tuple = (0.5, 2.2)
    person_width: tuple = (0.15, 1.2)
    chunk: int = 10000


def roi_mask(xyz, roi, z_min):
    return ((xyz[:, 0] > roi[0]) & (xyz[:, 0] < roi[1]) &
            (xyz[:, 1] > roi[2]) & (xyz[:, 1] < roi[3]) &
            (xyz[:, 2] > max(roi[4], z_min)) & (xyz[:, 2] < roi[5]))


def voxel_downsample(xyz, leaf):
    """Centroid of the points in each occupied leaf-sized voxel."""
    if xyz.shape[0] == 0:
        return np.zeros((0, 3))
    keys = np.floor(xyz / leaf).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inverse = np.unique(flat, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    out = np.empty((count.size, 3))
    for axis in range(3):
        out[:, axis] = np.bincount(inverse, weights=xyz[:, axis]) / count
    return out


def knn_query(points, radius, k_max, chunk):
    """Tree build + rectangular radius-limited kNN. Returns per-chunk (dist, idx)."""
    tree = cKDTree(points)
    out = []
    for start in range(0, points.shape[0], chunk):
        d, idx = tree.query(points[start:start + chunk], k=k_max,
                            distance_upper_bound=radius, workers=-1)
        if d.ndim == 1:
            d, idx = d[:, None], idx[:, None]
        out.append((d, idx))
    return tree, out


def eigenvalues(points, neighbours, k_max, k_min):
    """Covariance eigenvalues (descending) of each neighbourhood with >= k_min points."""
    lam_all, count_all, capped = [], [], 0
    for d, idx in neighbours:
        valid = np.isfinite(d)
        nb = points[np.where(valid, idx, 0)]
        w = valid.astype(np.float64)
        n = w.sum(1)
        capped += int((n >= k_max).sum())
        keep = n >= k_min
        if not keep.any():
            continue
        nb, w, n = nb[keep], w[keep], n[keep]
        mean = (nb * w[..., None]).sum(1) / n[:, None]
        centred = (nb - mean[:, None, :]) * w[..., None]
        cov = np.einsum("nki,nkj->nij", centred, centred) / (n[:, None, None] - 1.0)
        lam_all.append(np.linalg.eigvalsh(cov)[:, ::-1])
        count_all.append(n)
    if not lam_all:
        return np.zeros((0, 3)), np.zeros(0), 0.0
    lam = np.clip(np.concatenate(lam_all), 0.0, None)
    return lam, np.concatenate(count_all), capped / max(points.shape[0], 1)


def _plogp(p):
    return np.where(p > EPS, p * np.log(np.clip(p, EPS, None)), 0.0)


def entropies(lam, neg_tol=1e-6):
    """(H_DS, H_eigen, L, P, S) per accepted point from eigenvalues sorted l1>=l2>=l3."""
    lam = np.asarray(lam, dtype=np.float64)
    if lam.shape[0] == 0:
        z = np.zeros(0)
        return z, z, z, z, z
    l1 = lam[:, 0]
    good = ((lam[:, 0] >= lam[:, 1] - 1e-12) & (lam[:, 1] >= lam[:, 2] - 1e-12)
            & (lam[:, 2] >= -neg_tol * np.maximum(l1, EPS)) & (l1 > EPS))
    lam = np.clip(lam[good], 0.0, None)
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    L = np.clip((l1 - l2) / l1, 0, 1)
    P = np.clip((l2 - l3) / l1, 0, 1)
    S = np.clip(l3 / l1, 0, 1)
    h_ds = -(_plogp(L) + _plogp(P) + _plogp(S))
    e = lam / np.clip(lam.sum(1), EPS, None)[:, None]
    h_eigen = -(_plogp(e[:, 0]) + _plogp(e[:, 1]) + _plogp(e[:, 2]))
    return h_ds, h_eigen, L, P, S


def cluster(points, tree, p):
    """Euclidean clustering + person-candidate filter. Returns (results, stage times)."""
    t0 = time.perf_counter()
    if tree is None:
        tree = cKDTree(points)
    pairs = tree.query_pairs(r=p.cluster_tol, output_type="ndarray")
    t1 = time.perf_counter()
    n = points.shape[0]
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    t2 = time.perf_counter()
    sizes = np.bincount(labels, minlength=n_comp)
    big = np.flatnonzero(sizes >= p.cluster_min_pts)
    order = np.argsort(labels, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    sorted_pts = points[order]
    lo = np.minimum.reduceat(sorted_pts, starts, axis=0)[big]
    hi = np.maximum.reduceat(sorted_pts, starts, axis=0)[big]
    extent = hi - lo
    height, width = extent[:, 2], np.maximum(extent[:, 0], extent[:, 1])
    person = ((height >= p.person_height[0]) & (height <= p.person_height[1]) &
              (width >= p.person_width[0]) & (width <= p.person_width[1]))
    t3 = time.perf_counter()
    result = {"n_pairs": int(len(pairs)), "n_components": int(n_comp),
              "n_clusters": int(big.size), "n_person_candidates": int(person.sum()),
              "largest_cluster": int(sizes.max()) if sizes.size else 0}
    times = {"cl_pairs_ms": (t1 - t0) * 1e3, "cl_cc_ms": (t2 - t1) * 1e3,
             "cl_stats_ms": (t3 - t2) * 1e3}
    return result, times


ENTROPY_STAGES = ["accumulate_ms", "roi_ms", "voxel_ms", "knn_ms", "eig_ms", "entropy_ms"]
CLUSTER_STAGES = ["cl_pairs_ms", "cl_cc_ms", "cl_stats_ms"]


def process_window(frames, p, workload="entropy"):
    """frames: list of (xyz float32 (M,3), tag uint8 (M,)) 100 ms frames of one window.
    workload: entropy | cluster | both. Returns one flat dict of timings and results."""
    nan = float("nan")
    row = {k: 0.0 for k in ENTROPY_STAGES + CLUSTER_STAGES}
    row.update(H_DS=nan, H_eigen=nan, mean_L=nan, mean_P=nan, mean_S=nan, frac_high_S=nan,
               n_valid=0, mean_neighbors=nan, frac_capped=nan, n_pairs=-1, n_components=-1,
               n_clusters=-1, n_person_candidates=-1, largest_cluster=-1)
    t0 = time.perf_counter()
    xyz = np.concatenate([f[0] for f in frames]).astype(np.float64)
    tag = np.concatenate([f[1] for f in frames])
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-6)
    xyz, tag = xyz[ok], tag[ok]
    t1 = time.perf_counter()
    mask = roi_mask(xyz, p.roi, p.z_min)
    roi_xyz, roi_tag = xyz[mask], tag[mask]
    t2 = time.perf_counter()
    points = voxel_downsample(roi_xyz, p.leaf)
    t3 = time.perf_counter()
    row.update(accumulate_ms=(t1 - t0) * 1e3, roi_ms=(t2 - t1) * 1e3, voxel_ms=(t3 - t2) * 1e3,
               n_raw=int(xyz.shape[0]), n_roi=int(roi_xyz.shape[0]), n_voxel=int(points.shape[0]),
               frac_tag_particle=float((((roi_tag >> 2) & 3) != 0).mean()) if roi_tag.size else 0.0)
    tree = None
    if points.shape[0] >= p.min_voxels and workload in ("entropy", "both"):
        tree, neighbours = knn_query(points, p.radius, p.k_max, p.chunk)
        t4 = time.perf_counter()
        lam, counts, capped = eigenvalues(points, neighbours, p.k_max, p.k_min)
        t5 = time.perf_counter()
        h_ds, h_eigen, L, P, S = entropies(lam)
        if h_ds.size:
            row.update(H_DS=float(h_ds.mean()), H_eigen=float(h_eigen.mean()),
                       mean_L=float(L.mean()), mean_P=float(P.mean()), mean_S=float(S.mean()),
                       frac_high_S=float((S > p.s_thresh).mean()), n_valid=int(h_ds.size),
                       mean_neighbors=float(counts.mean()), frac_capped=float(capped))
        t6 = time.perf_counter()
        row.update(knn_ms=(t4 - t3) * 1e3, eig_ms=(t5 - t4) * 1e3, entropy_ms=(t6 - t5) * 1e3)
    if points.shape[0] >= 2 and workload in ("cluster", "both"):
        result, times = cluster(points, tree if workload == "cluster" else None, p)
        row.update(result)
        row.update(times)
    row["entropy_total_ms"] = sum(row[k] for k in ENTROPY_STAGES)
    row["cluster_total_ms"] = sum(row[k] for k in CLUSTER_STAGES)
    row["total_ms"] = (time.perf_counter() - t0) * 1e3
    return row
