#!/usr/bin/env python
"""Bake best-view keyframe colours onto a dense city mesh and export a GLB."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav3d.dense.cameras import Intrinsics, build_views, load_run_inputs  # noqa: E402
from uav3d.dense.glb_export import write_glb  # noqa: E402
from uav3d.modeling.texture_bake import (  # noqa: E402
    choose_vertex_views,
    gravity_transform,
    sample_vertex_colors,
    vertex_normals,
)


def _find_dense_dir(run: Path) -> Path:
    for name in ("dense_v2", "dense"):
        candidate = run / name
        if (candidate / "mesh.obj").exists():
            return candidate
    raise SystemExit(f"No mesh.obj under {run / 'dense_v2'} or {run / 'dense'}; run 'uav3d dense' first.")


def _load_views(run: Path, dense_dir: Path):
    poses, keyframes = load_run_inputs(run)
    views = build_views(poses, keyframes, run, 960)
    geometry = dense_dir / "rectified" / "geometry.json"
    if geometry.exists():
        data = json.loads(geometry.read_text(encoding="utf-8"))
        intr = Intrinsics(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
        for view in views:
            view.intrinsics = intr
    paths = {view.index: dense_dir / "rectified" / f"{view.index:05d}.png" for view in views}
    missing = [index for index, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing rectified images for views {missing[:5]}; rerun the dense stage.")
    return views, paths


def _load_depth_maps(dense_dir: Path, views):
    depth_dir = dense_dir / "depth_filtered"
    if not depth_dir.exists():
        depth_dir = dense_dir / "depth"
    depths = {}
    for view in views:
        path = depth_dir / f"{view.index:05d}.npy"
        if not path.exists():
            continue
        depth = np.load(path).astype(np.float32)
        if depth.shape == (view.intrinsics.height, view.intrinsics.width):
            depths[view.index] = depth
    return depths


def _fallback_colors(mesh: o3d.geometry.TriangleMesh, cloud_path: Path) -> np.ndarray:
    verts = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    if len(colors) == len(verts) and colors.size:
        return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    if cloud_path.exists():
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        points = np.asarray(cloud.points)
        point_colors = np.asarray(cloud.colors)
        if len(points) and len(point_colors) == len(points):
            from scipy.spatial import cKDTree

            _, index = cKDTree(points).query(verts, workers=-1)
            return np.clip(np.rint(point_colors[index] * 255.0), 0, 255).astype(np.uint8)
    return np.full((len(verts), 3), 150, np.uint8)


def _write_preview(verts: np.ndarray, colors: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    count = min(90000, len(verts))
    pick = rng.choice(len(verts), count, replace=False)
    sample = verts[pick]
    sample_colors = colors[pick].astype(np.float32) / 255.0
    theta = np.deg2rad(35.0)
    iso = np.stack(
        [
            np.cos(theta) * sample[:, 0] - np.sin(theta) * sample[:, 1],
            np.sin(theta) * sample[:, 0] + np.cos(theta) * sample[:, 1],
            sample[:, 2],
        ],
        axis=1,
    )
    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.view_init(elev=90, azim=-90)
    ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=sample_colors, s=0.5, depthshade=False)
    ax1.set_title("Top-down (coloured)")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.view_init(elev=24, azim=135)
    ax2.scatter(iso[:, 0], iso[:, 1], iso[:, 2], c=sample_colors, s=0.5, depthshade=False)
    ax2.set_title("Isometric (coloured)")
    for ax in (ax1, ax2):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((np.ptp(sample[:, 0]), np.ptp(sample[:, 1]), np.ptp(sample[:, 2])))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=115)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dense-dir", default=None)
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--out", default=str(ROOT / "city_textured.glb"))
    parser.add_argument("--max-faces", type=int, default=400000)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--min-component-faces", type=int, default=500)
    parser.add_argument("--preview", default=str(ROOT / "docs" / "preview_textured.png"))
    args = parser.parse_args()

    started = time.time()
    run = Path(args.run_dir).resolve()
    dense_dir = Path(args.dense_dir).resolve() if args.dense_dir else _find_dense_dir(run)
    mesh_path = Path(args.mesh) if args.mesh else dense_dir / "mesh.obj"
    cloud_path = dense_dir / "cloud.ply"

    views, image_paths = _load_views(run, dense_dir)
    depths = _load_depth_maps(dense_dir, views)
    print(f"views {len(views)}, depth maps {len(depths)}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise SystemExit(f"Empty mesh: {mesh_path}")
    if len(mesh.triangles) > args.max_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.max_faces)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.merge_close_vertices(0.004)

    clusters, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    if counts.size:
        keep = counts >= max(1, args.min_component_faces)
        drop = ~keep[np.asarray(clusters, dtype=np.int64)]
        mesh.remove_triangles_by_mask(drop)
        mesh.remove_unreferenced_vertices()
        print(f"kept {int(keep.sum())}/{len(counts)} components, {len(mesh.triangles)} triangles")

    if args.smooth > 0:
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=int(args.smooth),
            lambda_filter=0.5,
        )
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f"mesh {len(verts)} vertices, {len(faces)} triangles")

    normals = vertex_normals(verts, faces)
    best_view, best_score = choose_vertex_views(verts, normals, views, depths)
    assigned = int(np.count_nonzero(best_view >= 0))
    print(f"visible vertices {assigned}/{len(verts)} ({100.0 * assigned / len(verts):.1f}%)")

    fallback = _fallback_colors(mesh, cloud_path)

    def load_rgb(index: int) -> np.ndarray:
        image = cv2.imread(str(image_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Failed to read {image_paths[index]}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = sample_vertex_colors(verts, best_view, views, load_rgb, fallback)

    if cloud_path.exists():
        cloud = o3d.io.read_point_cloud(str(cloud_path))
        gravity_points = np.asarray(cloud.points)
    else:
        gravity_points = verts
    if len(gravity_points) == 0:
        gravity_points = verts
    rotation, offset = gravity_transform(gravity_points)
    verts_up = verts @ rotation.T
    verts_up[:, 2] += offset
    lo = np.percentile(verts_up, 2.0, axis=0)
    hi = np.percentile(verts_up, 98.0, axis=0)
    verts_up[:, 0] -= 0.5 * (lo[0] + hi[0])
    verts_up[:, 1] -= 0.5 * (lo[1] + hi[1])
    verts_up[:, 2] -= lo[2]

    out = Path(args.out)
    write_glb(out, verts_up, faces, colors=colors, name="city_textured")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    vertex_cloud = o3d.geometry.PointCloud()
    vertex_cloud.points = o3d.utility.Vector3dVector(verts_up)
    vertex_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    ply = out.with_suffix(".ply")
    o3d.io.write_point_cloud(str(ply), vertex_cloud)

    report = {
        "run_dir": str(run),
        "dense_dir": str(dense_dir),
        "mesh": str(mesh_path),
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "assigned_vertices": assigned,
        "assigned_fraction": round(assigned / max(len(verts), 1), 4),
        "unique_views_used": int(len(np.unique(best_view[best_view >= 0]))),
        "bbox_min": np.round(verts_up.min(0), 3).tolist(),
        "bbox_max": np.round(verts_up.max(0), 3).tolist(),
        "output": str(out),
    }
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    try:
        _write_preview(verts_up, colors, Path(args.preview))
        print(f"wrote {args.preview}")
    except Exception as exc:
        print(f"preview skipped: {exc}")

    print(f"done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

