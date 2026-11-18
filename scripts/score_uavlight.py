#!/usr/bin/env python
"""Score a UAVLight slot run against its RTK-anchored ground truth.

Scores MY OWN reconstruction (raw COLMAP sparse points + fused dense cloud)
against the released RTK-anchored ``downsampled_points.ply``. The released
COLMAP poses are used ONLY for evaluation alignment (Sim3) and ATE — never
as SfM input (that would be circular). Follows the traps in
docs/HANDOFF_NEXT_SESSION.md §12:

* score RAW COLMAP points (``colmap/sparse/0``), never ``sparse_points.npz``
* align with ORIENTATIONS, not centres-only Umeyama — the UAVLight slot is a
  straight 336 m strip (PCA 98.1% / 1.9% / 0.0%), so a centres-only fit
  leaves roll about the flight axis unconstrained
* score the cloud, not the depth maps

Alignment (orientation-aware Sim3): per-frame relative rotations
``R_gt^T @ R_sfm`` averaged over all matched frames (chordal mean), scale
from the RMS spread of the camera centres, translation from the centroids.
A plain centres-only Umeyama fit is computed alongside for comparison, and
the per-frame camera-centre error (ATE) is reported for the primary
alignment.

Headline metrics are percentages: precision@1m / precision@0.5m (fraction
of reconstructed points within 1 m / 0.5 m of the GT surface), plus median
error and Chamfer distance. Completeness is measured against the GT cropped
to the flown footprint (the full scene covers ground the flight never saw,
so whole-scene completeness is low by construction).

Usage:
    python scripts/score_uavlight.py --run-dir data/outputs/run_<ts> \\
        --scene /home/hatch/workspace/tbpp-data/uavlight/1121211223101030/1121211223101030 \\
        --frame-order /home/hatch/workspace/tbpp-data/uavlight/1121211223101030/slot_2025-07-09_frame_order.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav3d.dense.evaluate import load_ply_points  # noqa: E402
from uav3d.validation.shape_eval import (  # noqa: E402
    Similarity,
    crop_from_alignment,
    subsample,
    umeyama_similarity,
)


def _nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Nearest-neighbour distance from each point in ``src`` to ``dst``."""
    d, _ = cKDTree(dst).query(src, k=1, workers=-1)
    return np.asarray(d, dtype=np.float64)


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (wxyz, COLMAP cam_from_world) to rotation matrix."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def chordal_mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Average rotation matrices (chordal mean via SVD)."""
    mean = np.mean(rotations, axis=0)
    U, _, Vt = np.linalg.svd(mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def load_rtk_poses(images_txt: Path, slot: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """RTK COLMAP poses: basename -> (rotation cam_from_world, centre in metres)."""
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with open(images_txt) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        name = parts[9]
        if not name.startswith(slot + "/"):
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        t = np.array(list(map(float, parts[5:8])))
        R = quat_to_rot(qw, qx, qy, qz)
        centre = -R.T @ t
        poses[name.split("/")[-1]] = (R, centre)
    return poses


def orientation_aware_similarity(
    est_centres: np.ndarray,
    gt_centres: np.ndarray,
    est_rots: list[np.ndarray],
    gt_rots: list[np.ndarray],
) -> Similarity:
    """Sim3 from orientations (rotation) + centre spread (scale).

    Both rotations are cam_from_world for the same physical camera, so the
    frame rotation is ``R_gt^T @ R_sfm`` (see score_synthetic_sfm.py).
    """
    rel = [gt_r.T @ est_r for est_r, gt_r in zip(est_rots, gt_rots)]
    rot = chordal_mean_rotation(rel)
    est_c = est_centres - est_centres.mean(axis=0)
    gt_c = gt_centres - gt_centres.mean(axis=0)
    scale = float(
        np.sqrt(np.mean(np.sum(gt_c**2, axis=1)))
        / np.sqrt(np.mean(np.sum(est_c**2, axis=1)))
    )
    trans = gt_centres.mean(axis=0) - scale * (rot @ est_centres.mean(axis=0))
    return Similarity(scale=scale, rotation=rot, translation=trans)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True,
                        help="Scene dir containing sparse/0 and downsampled_points.ply")
    parser.add_argument("--frame-order", type=Path, required=True,
                        help="JSON list: muxed MP4 frame index -> source JPG basename")
    parser.add_argument("--slot", type=str, default="2025-07-09-12-41-31",
                        help="Slot prefix used in images.txt names")
    parser.add_argument("--gt-max-points", type=int, default=200_000)
    parser.add_argument("--recon-max-points", type=int, default=200_000)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    frame_order = json.loads(args.frame_order.read_text())
    rtk = load_rtk_poses(args.scene / "sparse" / "0" / "images.txt", args.slot)
    print(f"RTK poses for slot {args.slot}: {len(rtk)}")

    poses_path = run_dir / "poses.json"
    if not poses_path.exists():
        raise SystemExit(f"No poses.json at {poses_path}")
    poses = json.loads(poses_path.read_text())["poses"]

    num_pat = re.compile(r"(\d+)$")
    est_centres, gt_centres, est_rots, gt_rots = [], [], [], []
    for pose in poses:
        if not pose.get("registered", True):
            continue
        idx = int(num_pat.search(pose["frame_id"]).group(1))
        name = frame_order[idx]
        if name not in rtk or pose.get("camera_center_world") is None:
            continue
        est_centres.append(pose["camera_center_world"])
        gt_centres.append(rtk[name][1])
        est_rots.append(quat_to_rot(pose["qw"], pose["qx"], pose["qy"], pose["qz"]))
        gt_rots.append(rtk[name][0])
    est_centres = np.asarray(est_centres, dtype=np.float64)
    gt_centres = np.asarray(gt_centres, dtype=np.float64)
    print(f"matched camera centres: {len(est_centres)} of {len(poses)} poses")
    if len(est_centres) < 3:
        raise SystemExit("Fewer than 3 matched cameras; cannot align")

    # Primary alignment: orientation-aware. Comparison: centres-only Umeyama.
    sim_orient = orientation_aware_similarity(est_centres, gt_centres, est_rots, gt_rots)
    sim_umeyama = umeyama_similarity(est_centres, gt_centres)
    for label, sim in (("orientation-aware", sim_orient), ("centres-Umeyama", sim_umeyama)):
        err = np.linalg.norm(sim.apply(est_centres) - gt_centres, axis=1)
        print(f"ATE [{label}]: scale {sim.scale:.4f} m/unit, "
              f"median {np.median(err):.3f} m, RMSE {float(np.sqrt((err**2).mean())):.3f} m, "
              f"max {err.max():.3f} m (n={len(err)})")

    # Ground truth cloud — load FULL (1.45M pts). Scoring against a 200k
    # subsample would floor every distance at the ~0.7 m subsample spacing
    # and hide the true accuracy; the KD-tree query is still seconds.
    gt_path = args.scene / "downsampled_points.ply"
    gt = load_ply_points(gt_path)
    print(f"GT cloud: {len(gt):,} points from {gt_path.name} (full cloud used as the NN target)")

    result: dict = {
        "run_dir": str(run_dir),
        "scene": str(args.scene),
        "slot": args.slot,
        "matched_cameras": len(est_centres),
        "alignment": {
            "method": "orientation-aware Sim3 (chordal-mean relative rotations, RMS-spread scale)",
            "scale_m_per_unit": sim_orient.scale,
            "rotation": sim_orient.rotation.tolist(),
            "translation": sim_orient.translation.tolist(),
        },
        "ate_m": {},
        "clouds": {},
    }
    for label, sim in (("orientation_aware", sim_orient), ("centres_umeyama", sim_umeyama)):
        err = np.linalg.norm(sim.apply(est_centres) - gt_centres, axis=1)
        result["ate_m"][label] = {
            "scale_m_per_unit": sim.scale,
            "median": float(np.median(err)),
            "rmse": float(np.sqrt((err**2).mean())),
            "max": float(err.max()),
            "n": len(err),
        }

    # Reconstruction clouds: raw COLMAP sparse points + fused dense cloud.
    clouds: dict[str, np.ndarray] = {}
    sparse_path = run_dir / "colmap" / "sparse" / "0"
    if sparse_path.exists():
        import pycolmap

        rec = pycolmap.Reconstruction(str(sparse_path))
        pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
        if len(pts):
            clouds["sparse_colmap_raw"] = pts
            print(f"raw COLMAP sparse points: {len(pts):,}")
    dense_cloud = run_dir / "dense" / "cloud.ply"
    if dense_cloud.exists():
        pts = load_ply_points(dense_cloud)
        if len(pts):
            clouds["dense_fused"] = pts
            print(f"dense fused cloud: {len(pts):,} points")
    if not clouds:
        raise SystemExit("No reconstruction cloud found (need colmap/sparse/0 or dense/cloud.ply)")

    for name, recon in clouds.items():
        recon_sub = subsample(recon, args.recon_max_points)
        # Precision direction (recon -> GT): query the recon subsample
        # against the FULL GT cloud. Points outside the GT scene's own
        # bounding box are unscorable, not wrong: this slot's reconstruction
        # extends to y=+228 m while the released GT cloud stops at y=+106 m
        # (the GT does not cover the northern end of the flight, although
        # the RTK cameras fly to y=+150 m and align there to ~0.17 m).
        # They are reported separately as out_of_gt_coverage_pct.
        # Completeness direction (GT -> recon): query a GT subsample cropped
        # to the flown footprint against the FULL recon cloud.
        gt_min = gt.min(axis=0) - 2.0
        gt_max = gt.max(axis=0) + 2.0
        mapped_sub = sim_orient.apply(recon_sub)
        in_box = np.all((mapped_sub >= gt_min) & (mapped_sub <= gt_max), axis=1)
        recon_in = recon_sub[in_box]
        out_pct = 100.0 * float((~in_box).mean())
        d_r2g = _nn_distances(sim_orient.apply(recon_in), gt)
        crop = crop_from_alignment(recon_sub, sim_orient, margin=0.10, percentile=1.0)
        gt_crop = subsample(gt[crop.contains(gt)], args.gt_max_points)
        d_g2r = _nn_distances(gt_crop, sim_orient.apply(recon))

        def _prf(d: np.ndarray, t: float) -> float:
            return 100.0 * float((d < t).mean())

        entry = {
            "num_points": len(recon),
            "scored_points": len(recon_sub),
            "scored_points_in_gt_coverage": len(recon_in),
            "out_of_gt_coverage_pct": out_pct,
            "precision_at_0.5m_pct": _prf(d_r2g, 0.5),
            "precision_at_1m_pct": _prf(d_r2g, 1.0),
            "median_error_m": float(np.median(d_r2g)),
            "p90_error_m": float(np.percentile(d_r2g, 90)),
            "mean_error_m": float(d_r2g.mean()),
            "chamfer_m": float(0.5 * (d_r2g.mean() + d_g2r.mean())),
            "chamfer_median_m": float(np.median(np.concatenate([d_r2g, d_g2r]))),
            "completeness_at_0.5m_pct_footprint_crop": _prf(d_g2r, 0.5),
            "completeness_at_1m_pct_footprint_crop": _prf(d_g2r, 1.0),
            "fscore_at_1m_pct_footprint_crop": (
                2.0 * _prf(d_r2g, 1.0) * _prf(d_g2r, 1.0)
                / max(_prf(d_r2g, 1.0) + _prf(d_g2r, 1.0), 1e-12)
            ),
            "gt_points_in_footprint_crop": len(gt_crop),
        }
        result["clouds"][name] = entry
        print(f"\n[{name}] ({len(recon):,} pts, scored {len(recon_sub):,}, "
              f"{len(recon_in):,} in GT coverage, {out_pct:.1f}% outside GT coverage)")
        print(f"  precision@1m {entry['precision_at_1m_pct']:.1f}%  "
              f"precision@0.5m {entry['precision_at_0.5m_pct']:.1f}%  "
              f"median error {entry['median_error_m']:.3f} m  "
              f"p90 {entry['p90_error_m']:.2f} m  mean {entry['mean_error_m']:.2f} m  "
              f"Chamfer {entry['chamfer_m']:.3f} m")
        print(f"  completeness@1m {entry['completeness_at_1m_pct_footprint_crop']:.1f}%  "
              f"@0.5m {entry['completeness_at_0.5m_pct_footprint_crop']:.1f}%  "
              f"(footprint crop, {len(gt_crop):,} GT pts)")

    out = args.json_out or (run_dir / "uavlight_score.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
