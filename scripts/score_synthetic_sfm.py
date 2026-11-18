"""Score a synthetic street-canyon SfM run against analytic ground truth.

Scores the RAW COLMAP sparse points (``colmap/sparse/0``) against GT
surface samples from ``synthetic_street.build_street`` — never the
``sparse_points.npz`` cache, which goes stale after an SfM re-run
(docs/HANDOFF_NEXT_SESSION.md §12, Bug B).

Alignment is orientation-aware: rotation from the frame-0 quaternions,
scale from centre spread. A plain centres-only Umeyama fit is degenerate
on a straight flight (roll about the flight axis is unconstrained).

Reports the headline accuracy metrics as percentages: precision@1m and
precision@0.5m (fraction of reconstructed points within 1 m / 0.5 m of a
GT surface sample), plus median error and camera-centre error.

Validation only; the production pipeline never imports this module.

Usage:
    python scripts/render_street_video.py --out data/benchmark/flyover.mp4
    python -m uav3d.cli keyframes --video data/benchmark/flyover.mp4 \\
        --config configs/video_flyover.yaml --focal-px 265
    python -m uav3d.cli sfm --run-dir data/outputs/run_<ts> \\
        --config configs/video_flyover.yaml
    python scripts/score_synthetic_sfm.py --run-dir data/outputs/run_<ts>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import build_street, flight


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (wxyz) to rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _sample_box_surfaces(boxes: list[dict], total: int, seed: int = 0) -> np.ndarray:
    """Uniform samples over every box face, counts proportional to face area."""
    rng = np.random.default_rng(seed)
    # Per-face areas: (box, axis, side) with area of the face.
    faces = []
    for box in boxes:
        lo, hi = box["lo"], box["hi"]
        size = hi - lo
        for axis in range(3):
            area = float(size[(axis + 1) % 3] * size[(axis + 2) % 3])
            faces.append((box, axis, 0, area))
            faces.append((box, axis, 1, area))
    total_area = sum(f[3] for f in faces)
    samples = []
    for box, axis, side, area in faces:
        n = max(1, int(total * area / total_area))
        lo, hi = box["lo"], box["hi"]
        pts = rng.uniform(lo, hi, size=(n, 3))
        pts[:, axis] = lo[axis] if side == 0 else hi[axis]
        samples.append(pts)
    return np.vstack(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--step", type=float, default=1.36)
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--gt-samples", type=int, default=100_000)
    args = parser.parse_args()

    import pycolmap
    from scipy.spatial import cKDTree

    run_dir = args.run_dir
    sparse_path = run_dir / "colmap" / "sparse" / "0"
    if not sparse_path.exists():
        raise SystemExit(f"No COLMAP sparse model at {sparse_path}")
    poses_path = run_dir / "poses.json"
    if not poses_path.exists():
        raise SystemExit(f"No poses.json at {poses_path}")

    rec = pycolmap.Reconstruction(str(sparse_path))
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    if len(points) == 0:
        raise SystemExit("COLMAP model has no 3D points")
    print(f"raw COLMAP sparse points: {len(points):,}")

    poses = json.loads(poses_path.read_text(encoding="utf-8"))["poses"]
    gt_poses = flight(args.frames, args.step, args.camera_height, args.pitch)
    gt_centres = {i: -r.T @ t for i, (r, t) in enumerate(gt_poses)}

    num_pat = re.compile(r"(\d+)$")
    est_centres, gt_matched = [], []
    r0_sfm = r0_gt = None
    for pose in poses:
        if not pose.get("registered", True):
            continue
        idx = int(num_pat.search(pose["frame_id"]).group(1))
        if pose.get("camera_center_world") is None:
            continue
        est_centres.append(pose["camera_center_world"])
        gt_matched.append(gt_centres[idx])
        if idx == 0:
            r0_sfm = _quat_to_rot(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
            r0_gt = gt_poses[0][0]
    est_centres = np.asarray(est_centres, dtype=np.float64)
    gt_matched = np.asarray(gt_matched, dtype=np.float64)
    print(f"matched camera centres: {len(est_centres)}")
    if r0_sfm is None or r0_gt is None:
        raise SystemExit("Frame 0 was not registered; cannot do orientation-aware alignment")

    # Orientation-aware Sim(3): R from frame-0 quaternions, scale from the
    # spread of the centres, translation from the centroids.
    # Both rotations are cam_from_world: the same physical camera sees the
    # same image, so R_gt @ (s*A@X + t - C_gt) = R_sfm @ (X - C_sfm) for all
    # X, which gives A = R_gt^T @ R_sfm (NOT R_gt @ R_sfm^T).
    rot = r0_gt.T @ r0_sfm
    est_c = est_centres - est_centres.mean(axis=0)
    gt_c = gt_matched - gt_matched.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(gt_c**2, axis=1))) / np.sqrt(np.mean(np.sum(est_c**2, axis=1))))
    trans = gt_matched.mean(axis=0) - scale * (rot @ est_centres.mean(axis=0))
    aligned_points = scale * (points @ rot.T) + trans
    aligned_centres = scale * (est_centres @ rot.T) + trans
    centre_err = np.linalg.norm(aligned_centres - gt_matched, axis=1)
    print(f"alignment: scale {scale:.4f} m/unit")
    print(f"camera centre error: median {np.median(centre_err):.3f} m, max {centre_err.max():.3f} m")

    gt_samples = _sample_box_surfaces(build_street(), args.gt_samples)
    tree = cKDTree(gt_samples)
    dists, _ = tree.query(aligned_points, workers=-1)
    p1m = float(np.mean(dists <= 1.0)) * 100.0
    p05m = float(np.mean(dists <= 0.5)) * 100.0
    print(f"sparse-point accuracy: median {np.median(dists):.3f} m")
    print(f"precision@1m:   {p1m:.1f}%")
    print(f"precision@0.5m: {p05m:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
