#!/usr/bin/env python
"""Shape-quality evaluation of a reconstruction against a ground-truth reference.

Two ways to register a reconstruction against its reference, in order of
preference:

1. **Camera trajectory** (``--run-dir``): recover the Sim(3) that maps the SfM
   frame onto the reference from matched camera positions. This is exact,
   physically grounded, and independent of the surface geometry being measured.
2. **Shape alignment** (fallback): a 24-candidate axis search plus robust
   Sim(3) refinement on the point clouds. This is a last resort: a city
   reference is mostly ground plane, so scale is weakly constrained, and a
   single-pass reconstruction covers one corridor of it, so its moments differ
   from the reference's. Prefer (1) whenever camera ground truth exists.

Scoring is then done against a *crop* of the reference: the region the flight
actually observed, derived from the aligned reconstruction footprint or the
camera track. Scoring against a whole 300 m reference measures how much of the
city the aircraft did not fly over, not how accurate the surface is.

Usage:
    # preferred: exact trajectory alignment, crop from the flown corridor
    python scripts/eval_city_model.py --recon city_textured.glb \
        --run-dir data/outputs/run_20261113_055317 --json-out docs/eval.json

    # explicit crop
    python scripts/eval_city_model.py --recon city_textured.glb \
        --gt city_flyover_ground_truth_reference.glb --gt-crop -20,-20,-5,20,20,40

    # no camera ground truth available: shape alignment, footprint crop
    python scripts/eval_city_model.py --recon city_textured_prior.glb --gt-crop-from-recon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav3d.exceptions import DataError
from uav3d.validation.shape_eval import (
    CropBox,
    Similarity,
    align_clouds,
    align_from_trajectory,
    compose_similarities,
    crop_from_alignment,
    score_clouds,
    subsample,
)

GT_UP = "neg_y"


def load_points(path: Path, max_points: int) -> np.ndarray:
    """Load vertices from a GLB/OBJ/PLY, mesh or point cloud.

    The reference sidecar is a *point cloud* PLY, so ``force="mesh"`` silently
    yields zero vertices. Both cases are handled explicitly.
    """
    obj = trimesh.load(str(path), force="scene", process=False)
    if isinstance(obj, trimesh.Scene):
        chunks: list[np.ndarray] = []
        for geom in obj.geometry.values():
            verts = getattr(geom, "vertices", None)
            if verts is not None and len(verts):
                chunks.append(np.asarray(verts, dtype=np.float64))
        if not chunks:
            sys.exit(f"No geometry or points in {path}")
        points = np.vstack(chunks)
    else:
        points = np.asarray(obj.vertices, dtype=np.float64)
    if len(points) == 0:
        sys.exit(f"No points loaded from {path}")
    return subsample(points, max_points)


def to_gltf_frame(points: np.ndarray, up: str) -> np.ndarray:
    """Map a Z-up reference (glTF convention) into the Y-up reconstruction frame.

    ``up="neg_y"`` marks references whose own metadata says the ground plane is
    ``y=0`` and up is ``-y`` (the city sidecar does); for those the Y-up frame is
    ``(x, z, -y)``. ``up="z"`` is the usual glTF Z-up -> Y-up ``(x, -z, y)``.
    """
    if up == "neg_y":
        return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)


def load_trajectory(run_dir: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Match reconstructed camera centres to reference positions by image id.

    Returns ``(estimate, reference)`` or ``None`` when the ground truth is not a
    camera trajectory (e.g. an evaluation-only reference mesh).
    """
    poses_path = run_dir / "poses.json"
    if not poses_path.exists():
        return None
    try:
        poses = json.loads(poses_path.read_text(encoding="utf-8")).get("poses", [])
    except (OSError, json.JSONDecodeError):
        return None
    if not poses:
        return None

    reference = load_reference_cameras(gt_path)
    if reference is None:
        return None

    est: list[list[float]] = []
    ref: list[list[float]] = []
    for pose in poses:
        center = pose.get("camera_center")
        source_id = pose.get("source_image_id")
        if source_id is None:
            frame_id = str(pose.get("frame_id", ""))
            digits = "".join(ch for ch in frame_id if ch.isdigit())
            source_id = int(digits) if digits else None
        if center is None or source_id is None or int(source_id) not in reference:
            continue
        est.append([float(v) for v in center])
        ref.append(reference[int(source_id)])
    if len(est) < 3:
        return None
    return np.asarray(est, dtype=np.float64), np.asarray(ref, dtype=np.float64)


def load_reference_cameras(gt_path: Path) -> dict[int, list[float]] | None:
    """Read ``image_id,x,y,z`` camera ground truth (the city sidecar format)."""
    if not gt_path.exists() or gt_path.suffix.lower() != ".csv":
        return None
    reference: dict[int, list[float]] = {}
    try:
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            reference[int(float(parts[0]))] = [float(p) for p in parts[1:4]]
    except (OSError, ValueError):
        return None
    return reference or None


def find_run_meshes(run_dir: Path) -> list[Path]:
    """All dense meshes a run produced, in the SfM frame.

    A run can hold several (``dense/``, ``dense_v2/``, ``dense_prior/``), and the
    exported model came from exactly one of them. They are returned together so
    the bridge can pick the right one by fit quality instead of guessing.
    """
    found: list[Path] = []
    for candidate in ("dense/mesh.obj", "dense_v2/mesh.obj", "dense_prior/mesh.obj", "map_3d.ply"):
        path = run_dir / candidate
        if path.exists():
            found.append(path)
    return found


def bridge_to_sfm_frame(
    recon: np.ndarray,
    run_meshes: list[Path],
    tolerance: float = 0.05,
) -> tuple[Similarity, Path]:
    """Map an exported model back into its run's SfM frame.

    A viewer-ready GLB is not in the SfM frame: the exporter applies a gravity
    rotation, recentres the scene and may rescale it. The camera trajectory is
    known in the SfM frame, so the exported model has to be brought back there
    before that transform can be applied.

    This alignment is well conditioned — unlike a reference alignment, both
    clouds describe the *same* extent of the same geometry — so a direct fit is
    enough. When a run has several candidate meshes, the one that the exported
    model actually fits is selected; a model built from a different mesh would
    not register tightly against it.

    Args:
        recon: exported model points.
        run_meshes: candidate run meshes, in the SfM frame.
        tolerance: fraction of the SfM extent allowed as median residual.

    Returns:
        ``(similarity, source_mesh)`` mapping ``recon`` into the SfM frame.
    """
    probe = subsample(recon, 40000)
    best: tuple[float, Similarity, Path] | None = None
    for path in run_meshes:
        sfm = subsample(load_points(path, 60000), 60000)
        similarity, _ = align_clouds(probe, sfm)
        mapped = similarity.apply(probe)
        tree = cKDTree(sfm)
        distances, _ = tree.query(mapped, k=1, workers=-1)
        residual = float(np.median(distances))
        if best is None or residual < best[0]:
            best = (residual, similarity, path)
    if best is None:
        raise DataError("no candidate run mesh could be loaded")
    residual, similarity, source = best
    sfm_extent = float(np.linalg.norm(np.ptp(similarity.apply(probe), axis=0)))
    if sfm_extent > 0 and residual > tolerance * sfm_extent:
        raise DataError(
            f"Could not bridge the exported model into the SfM frame: best candidate "
            f"{source.name} has median residual {residual:.3f} against a {sfm_extent:.1f} extent"
        )
    return similarity, source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gt", default=str(ROOT / "city_flyover_ground_truth_reference.glb"))
    ap.add_argument("--run-dir", default=None, help="Run directory with poses.json, for exact trajectory alignment")
    ap.add_argument(
        "--sfm-frame",
        action="store_true",
        help="The reconstruction is already in the run's SfM frame (e.g. dense/mesh.obj), so no bridge is needed",
    )
    ap.add_argument("--gt-cameras", default=None, help="CSV of image_id,x,y,z reference camera positions")
    ap.add_argument("--gt-up", choices=("neg_y", "z"), default=GT_UP, help="Reference up axis convention")
    ap.add_argument("--out", default=None, help="Write the metrics JSON here")
    ap.add_argument("--json-out", default=None, help="Alias of --out")
    ap.add_argument("--subsample", type=int, default=120000)
    ap.add_argument(
        "--gt-crop",
        default=None,
        help="Ground-truth crop in GT coordinates: xmin,ymin,zmin,xmax,ymax,zmax",
    )
    ap.add_argument(
        "--gt-crop-from-recon",
        action="store_true",
        help="Derive the crop from the aligned reconstruction footprint",
    )
    ap.add_argument(
        "--crop-margin",
        type=float,
        default=0.10,
        help="Padding for --gt-crop-from-recon, as a fraction of the recon extent",
    )
    ap.add_argument(
        "--crop-percentile",
        type=float,
        default=1.0,
        help="Use this percentile band for the footprint crop, so stray points cannot stretch it",
    )
    ap.add_argument(
        "--min-crop-fraction",
        type=float,
        default=0.05,
        help="Refuse a crop that keeps less than this fraction of the reference",
    )
    ap.add_argument(
        "--thresholds",
        default="0.05,0.1,0.2,0.5,1.0,2.0",
        help="Comma-separated distance thresholds in reference units for precision/completeness/F",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        thresholds = tuple(float(t.strip()) for t in args.thresholds.split(",") if t.strip())
    except ValueError:
        sys.exit(f"--thresholds must be comma-separated numbers, got {args.thresholds!r}")
    if not thresholds:
        sys.exit("--thresholds must contain at least one value")

    recon_path = Path(args.recon)
    gt_path = Path(args.gt)
    if not recon_path.exists():
        sys.exit(f"Reconstruction not found: {recon_path}")
    if not gt_path.exists():
        sys.exit(f"Ground truth not found: {gt_path}")

    recon = load_points(recon_path, args.subsample)
    gt_raw = load_points(gt_path, args.subsample)
    # The reference is remapped into the Y-up reconstruction frame only when the
    # registration is geometric. Trajectory alignment is solved in the reference's
    # *own* frame (the camera CSV uses it), so remapping the cloud would score the
    # model against a frame the cameras were never in.
    gt_all = gt_raw

    alignment_source = "shape"
    similarity: Similarity | None = None
    rotation_index = -1
    trajectory_error: float | None = None

    if args.run_dir:
        cameras = load_trajectory(Path(args.run_dir), Path(args.gt_cameras or ""))
        if cameras is not None:
            estimate, reference = cameras
            to_reference = align_from_trajectory(estimate, reference)
            trajectory_error = float(
                np.median(np.linalg.norm(to_reference.apply(estimate) - reference, axis=1))
            )
            alignment_source = "trajectory"
            print(
                f"trajectory alignment on {len(estimate)} cameras: "
                f"scale={to_reference.scale:.4f} median residual={trajectory_error:.4f}",
                file=sys.stderr,
            )
            if args.sfm_frame:
                similarity = to_reference
            else:
                run_meshes = find_run_meshes(Path(args.run_dir))
                if not run_meshes:
                    sys.exit(
                        f"--run-dir {args.run_dir} has no dense mesh to bridge the exported model "
                        f"into the SfM frame; pass --sfm-frame to evaluate the run's own mesh"
                    )
                try:
                    to_sfm, source_mesh = bridge_to_sfm_frame(recon, run_meshes)
                except DataError as exc:
                    sys.exit(str(exc))
                print(
                    f"bridged {recon_path.name} into the SfM frame via {source_mesh.name} "
                    f"(scale={to_sfm.scale:.4f})",
                    file=sys.stderr,
                )
                similarity = compose_similarities(to_reference, to_sfm)

    if similarity is None:
        gt_all = to_gltf_frame(gt_raw, args.gt_up)
        similarity, rotation_index = align_clouds(recon, gt_all)
    if similarity.is_degenerate:
        sys.exit("Alignment failed: the similarity fit is degenerate (check the inputs)")

    crop: CropBox | None = None
    if args.gt_crop:
        try:
            crop = CropBox.parse(args.gt_crop)
        except DataError as exc:
            sys.exit(str(exc))
    elif args.gt_crop_from_recon:
        crop = crop_from_alignment(recon, similarity, margin=args.crop_margin, percentile=args.crop_percentile)

    if crop is not None:
        keep = crop.contains(gt_all)
        fraction = float(keep.mean())
        if fraction < args.min_crop_fraction:
            sys.exit(
                f"Crop keeps only {fraction * 100:.1f}% of the reference "
                f"(minimum {args.min_crop_fraction * 100:.0f}%); refusing to report a meaningless score"
            )
        gt = gt_all[keep]
        if alignment_source == "shape":
            # Re-align on the crop: the coarse pass saw the whole reference.
            similarity, rotation_index = align_clouds(recon, gt)
        crop_payload = {
            "minimum": [round(float(v), 4) for v in crop.minimum],
            "maximum": [round(float(v), 4) for v in crop.maximum],
            "extent": [round(float(v), 4) for v in crop.extent],
            "fraction_kept": round(fraction, 4),
        }
    else:
        gt = gt_all
        crop_payload = None

    metrics = score_clouds(recon, gt, similarity, thresholds=thresholds)
    result = {
        "recon": str(recon_path),
        "gt": str(gt_path),
        "recon_points": len(recon),
        "gt_points_total": len(gt_all),
        "gt_points_scored": len(gt),
        "crop": crop_payload,
        "alignment": {
            "source": alignment_source,
            "scale_meters_per_unit": round(float(similarity.scale), 6),
            "rotation_index": int(rotation_index),
            "camera_median_residual_m": None if trajectory_error is None else round(trajectory_error, 6),
            "rotation": [[round(float(v), 6) for v in row] for row in similarity.rotation],
            "translation": [round(float(v), 4) for v in similarity.translation],
        },
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }

    print(json.dumps(result, indent=2))
    out = args.json_out or args.out
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
