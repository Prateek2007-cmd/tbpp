#!/usr/bin/env python
"""Render top-down and isometric preview images of a dense fused cloud.

PCA-aligns the cloud so "up" is +Z, removes gross outliers, subsamples, and
writes PNG previews (useful because a single-pass cloud has no reliable gravity
reference and may otherwise look tilted in a 3D viewer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def pca_align(p: np.ndarray) -> np.ndarray:
    centre = p.mean(axis=0)
    pc = p - centre
    _w, v = np.linalg.eigh(pc.T @ pc / max(len(pc), 1))
    up = v[:, 0]
    h = pc @ up
    p5, med, p95 = np.percentile(h, [5, 50, 95])
    if (p95 - med) < (med - p5):
        up = -up
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, target)
    s = np.linalg.norm(axis)
    if s > 1e-6:
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        R = np.eye(3) + skew + skew @ skew * ((1.0 - up @ target) / (s * s + 1e-9))
    else:
        R = np.eye(3) if up @ target > 0 else -np.eye(3)
    return pc @ R.T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=60000)
    args = ap.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.cloud))
    pts = np.asarray(cloud.points)
    cols = np.asarray(cloud.colors)
    down = cloud.voxel_down_sample(0.15)
    cl, ind = down.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)
    pts = np.asarray(cl.points)
    cols = np.asarray(cl.colors)
    if len(cols) != len(pts) or len(cols) == 0:
        cols = np.ones_like(pts) * 0.7

    pts_up = pca_align(pts)
    if len(pts_up) > args.samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pts_up), args.samples, replace=False)
        pts_up = pts_up[sel]
        cols = cols[sel]

    span = np.ptp(pts_up, axis=0)
    print("bbox min", np.round(pts_up.min(0), 2).tolist(), "max", np.round(pts_up.max(0), 2).tolist())
    print("span", np.round(span, 2).tolist())

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.view_init(elev=90, azim=-90)
    ax1.scatter(pts_up[:, 0], pts_up[:, 1], pts_up[:, 2], c=cols, s=1.0, depthshade=False)
    ax1.set_title("Top-down (XY)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.view_init(elev=25, azim=135)
    ax2.scatter(pts_up[:, 0], pts_up[:, 1], pts_up[:, 2], c=cols, s=1.0, depthshade=False)
    ax2.set_title("Isometric")
    for ax in (ax1, ax2):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((np.ptp(pts_up[:, 0]), np.ptp(pts_up[:, 1]), np.ptp(pts_up[:, 2])))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out), dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()