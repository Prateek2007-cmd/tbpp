"""Best-view vertex texturing for a dense city mesh.

Each mesh vertex is painted from the registered keyframe that sees it most
fronto-parallelly. The multi-view depth map acts as an occlusion test, so a
facade vertex is not painted from a camera for which another building is in
front. This is intentionally simpler than a UV atlas: no seam packing, no
texture-bleed bugs, and the existing GLB writer already supports COLOR_0.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from uav3d.dense.cameras import CameraView


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(verts)
    if faces.size == 0:
        return normals.astype(np.float32)
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths < 1e-12] = 1.0
    return (normals / lengths[:, None]).astype(np.float32)


def bilinear_sample_rgb(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an ``(H, W, 3)`` uint8 image at floating point pixel positions."""
    image = np.asarray(image)
    height, width = image.shape[:2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.size == 0:
        return np.zeros((0, 3), np.uint8), np.zeros((0,), bool)

    inside = (u >= 0.0) & (v >= 0.0) & (u < width - 1) & (v < height - 1)
    uc = np.clip(u, 0.0, width - 1.001)
    vc = np.clip(v, 0.0, height - 1.001)
    x0 = np.floor(uc).astype(np.int64)
    y0 = np.floor(vc).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (uc - x0).astype(np.float32)[:, None]
    dy = (vc - y0).astype(np.float32)[:, None]

    flat = image.reshape(-1, 3)
    i00 = y0 * width + x0
    i10 = y0 * width + x1
    i01 = y1 * width + x0
    i11 = y1 * width + x1
    top = flat[i00].astype(np.float32) * (1.0 - dx) + flat[i10].astype(np.float32) * dx
    bottom = flat[i01].astype(np.float32) * (1.0 - dx) + flat[i11].astype(np.float32) * dx
    out = top * (1.0 - dy) + bottom * dy
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out[~inside] = 0
    return out, inside


def choose_vertex_views(
    vertices: np.ndarray,
    normals: np.ndarray,
    views: list[CameraView],
    depth_maps: dict[int, np.ndarray] | None = None,
    facing_min: float = 0.15,
    depth_tol: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the most fronto-parallel visible camera for every vertex.

    Returns:
        ``(best_view, best_score)``; ``best_view`` is -1 where no camera saw the
        vertex, ``best_score`` is the winning squared facing cosine.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    depths = depth_maps or {}
    best_score = np.zeros(len(verts), dtype=np.float32)
    best_view = np.full(len(verts), -1, dtype=np.int16)

    for view in views:
        intr = view.intrinsics
        pc = verts @ view.R.T + view.t
        z = pc[:, 2]
        u = intr.fx * pc[:, 0] / np.maximum(z, 1e-9) + intr.cx
        v = intr.fy * pc[:, 1] / np.maximum(z, 1e-9) + intr.cy
        inside = (
            (z > 1e-6)
            & (u >= 0.5)
            & (u < intr.width - 1.5)
            & (v >= 0.5)
            & (v < intr.height - 1.5)
        )
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue

        rays = view.center[None, :] - verts[idx]
        lengths = np.linalg.norm(rays, axis=1)
        lengths[lengths < 1e-9] = 1e-9
        facing = np.abs(np.sum(nrm[idx] * (rays / lengths[:, None]), axis=1))

        visible = np.ones(idx.size, dtype=bool)
        depth = depths.get(view.index)
        if depth is not None:
            dh, dw = depth.shape[:2]
            sx = dw / float(intr.width)
            sy = dh / float(intr.height)
            cols = np.clip((u[idx] * sx).astype(np.int64), 0, dw - 1)
            rows = np.clip((v[idx] * sy).astype(np.int64), 0, dh - 1)
            sampled = depth[rows, cols]
            tol = np.maximum(0.05, float(depth_tol) * z[idx])
            visible = (sampled <= 0) | (sampled >= z[idx] - tol)

        score = (facing ** 2) / (1.0 + 0.02 * z[idx])
        keep = (facing >= float(facing_min)) & visible
        if not np.any(keep):
            continue
        keep_idx = idx[keep]
        keep_score = score[keep].astype(np.float32)
        better = keep_score > best_score[keep_idx]
        if np.any(better):
            chosen = keep_idx[better]
            best_score[chosen] = keep_score[better]
            best_view[chosen] = view.index

    return best_view, best_score


def sample_vertex_colors(
    vertices: np.ndarray,
    best_view: np.ndarray,
    views: list[CameraView],
    load_image: Callable[[int], np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """Sample every vertex from its chosen view and fill misses from fallback."""
    verts = np.asarray(vertices, dtype=np.float64)
    colors = np.array(fallback, dtype=np.uint8, copy=True)
    lookup = {view.index: view for view in views}
    chosen = np.flatnonzero(best_view >= 0)
    for view_index in np.unique(best_view[chosen]):
        subset = chosen[best_view[chosen] == view_index]
        view = lookup.get(int(view_index))
        if view is None or subset.size == 0:
            continue
        image = np.asarray(load_image(int(view_index)))
        if image.ndim != 3 or image.shape[2] < 3:
            continue
        image = image[:, :, :3]
        pc = verts[subset] @ view.R.T + view.t
        z = pc[:, 2]
        u = view.intrinsics.fx * pc[:, 0] / np.maximum(z, 1e-9) + view.intrinsics.cx
        v = view.intrinsics.fy * pc[:, 1] / np.maximum(z, 1e-9) + view.intrinsics.cy
        u = np.clip(u, 0.0, view.intrinsics.width - 1.001)
        v = np.clip(v, 0.0, view.intrinsics.height - 1.001)
        sampled, inside = bilinear_sample_rgb(image, u, v)
        valid = inside & (z > 0)
        colors[subset[valid]] = sampled[valid]
    return colors


def gravity_transform(points: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA gravity for a city cloud: the smallest-variance axis is vertical.

    Returns:
        ``(R, offset)`` such that ``points @ R.T + [0, 0, offset]`` is Z-up with
        the ground near zero.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 300000:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), 300000, replace=False)]
    centered = pts - pts.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    up = vectors[:, 0]
    h = pts @ up
    p2, med, p98 = np.percentile(h, [2, 50, 98])
    if (p98 - med) < (med - p2):
        up = -up
        h = -h
        p2 = float(np.percentile(h, 2))
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    s = float(np.linalg.norm(axis))
    if s > 1e-6:
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - float(up @ target)) / (s * s + 1e-9))
    else:
        rotation = np.eye(3) if float(up @ target) > 0 else -np.eye(3)
    return rotation, -float(p2)
