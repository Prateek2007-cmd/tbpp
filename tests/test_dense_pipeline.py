"""End-to-end test of the CPU dense pipeline on an analytically rendered plane.

The scene is a slanted textured wall seen by a laterally translating camera, so
ground-truth depth is known in closed form and the recovered cloud can be scored
against it directly.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav3d.dense.cameras import Intrinsics
from uav3d.dense.config import DenseConfig
from uav3d.dense.evaluate import load_ply_points
from uav3d.dense.pipeline import run_dense

WIDTH, HEIGHT = 192, 144
FOCAL = 150.0
WALL_Z = 20.0
STEP_M = 1.0
FRAMES = 10
TEXTURE_METRES = (48.0, 36.0)
_TEXTURE_SHAPE = (720, 960)


def _texture_image() -> np.ndarray:
    """Band-limited random texture, so no view resamples fine detail differently."""
    rng = np.random.default_rng(5)
    coarse = rng.normal(size=(90, 120))
    texture = cv2.resize(coarse, (_TEXTURE_SHAPE[1], _TEXTURE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), 2.0)
    texture -= texture.min()
    texture /= max(texture.max(), 1e-9)
    return (texture * 255.0).astype(np.uint8)


_TEXTURE = _texture_image()


def _render(centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Render a fronto-parallel textured wall from a camera looking along +z."""
    u, v = np.meshgrid(
        np.arange(WIDTH, dtype=np.float32) + 0.5, np.arange(HEIGHT, dtype=np.float32) + 0.5
    )
    world_x = centre[0] + (u - WIDTH / 2.0) / FOCAL * WALL_Z
    world_y = centre[1] + (v - HEIGHT / 2.0) / FOCAL * WALL_Z

    # Texture spans TEXTURE_METRES metres, anchored at the origin.
    px = (world_x / TEXTURE_METRES[0] + 0.5) * (_TEXTURE_SHAPE[1] - 1)
    py = (world_y / TEXTURE_METRES[1] + 0.5) * (_TEXTURE_SHAPE[0] - 1)
    image = cv2.remap(
        _TEXTURE, px.astype(np.float32), py.astype(np.float32), interpolation=cv2.INTER_LINEAR
    )
    depth = np.full((HEIGHT, WIDTH), WALL_Z, dtype=np.float64)
    return image, depth


def _write_run(run_dir: Path) -> None:
    images = run_dir / "sfm" / "images"
    images.mkdir(parents=True, exist_ok=True)

    poses, keyframes = [], []
    for index in range(FRAMES):
        centre = np.array([index * STEP_M, 0.0, 0.0])
        image, _ = _render(centre)
        name = f"frame_{index:04d}.png"
        path = images / name
        assert cv2.imwrite(str(path), image)
        poses.append(
            {
                "frame_id": f"frame_{index:04d}",
                "timestamp_us": 1_000_000 + index * 100_000,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "tx": float(-centre[0]),
                "ty": 0.0,
                "tz": 0.0,
                "camera_center": centre.tolist(),
                "registered": True,
            }
        )
        keyframes.append(
            {
                "frame_id": f"frame_{index:04d}",
                "timestamp_us": 1_000_000 + index * 100_000,
                "file_path": f"sfm/images/{name}",
                "selected": True,
            }
        )

    (run_dir / "poses.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic dense pipeline test",
                "coordinate_frame": "cam_from_world",
                "camera": {
                    "camera_id": 1,
                    "model": "PINHOLE",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "params": [FOCAL, FOCAL, WIDTH / 2.0, HEIGHT / 2.0],
                    "scale_factor": 1.0,
                },
                "metrics": {
                    "num_images_total": FRAMES,
                    "num_images_registered": FRAMES,
                    "registration_rate": 1.0,
                    "mean_reprojection_error_px": 0.4,
                },
                "poses": poses,
                "unregistered_frames": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "keyframes.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic dense pipeline test",
                "keyframes": keyframes,
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(11)
    xs = rng.uniform(-3.0, 6.0, 400)
    ys = rng.uniform(-3.0, 3.0, 400)
    zs = np.full_like(xs, WALL_Z)
    np.savez(
        run_dir / "sparse_points.npz",
        points=np.stack([xs, ys, zs], axis=1),
        rgb=rng.integers(0, 255, (400, 3)),
    )


def test_dense_pipeline_recovers_plane_depth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)

    config = DenseConfig(
        max_width=WIDTH,
        num_sources=4,
        plane_step_px=2.0,
        min_planes=16,
        max_planes=64,
        cost_threshold=0.5,
        min_views=2,
        mesh=True,
    )
    result = run_dense(run_dir=run_dir, config=config)

    assert result.point_cloud_path.exists()
    points = load_ply_points(result.point_cloud_path)
    assert points.shape[0] > 500, f"only {points.shape[0]} points survived"

    # Every point should sit on the analytic wall.
    residual = np.abs(points[:, 2] - WALL_Z)
    assert float(np.median(residual)) < 0.6
    assert float(np.percentile(residual, 90)) < 2.0

    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"]
    assert isinstance(report["scale"]["status"], str) and report["scale"]["status"]
    assert report["fusion"]["num_points"] == points.shape[0]
    assert 0.0 <= report["depth"]["valid_fraction"] <= 1.0

    if result.mesh_path is not None:
        assert result.mesh_path.exists()
        assert report["mesh"]["num_faces"] > 0


def test_dense_pipeline_is_deterministic(tmp_path: Path) -> None:
    """Two runs on identical inputs must produce byte-identical clouds."""
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)
    config = DenseConfig(
        max_width=WIDTH,
        num_sources=3,
        plane_step_px=3.0,
        min_planes=8,
        max_planes=32,
        mesh=False,
    )
    first = run_dense(run_dir=run_dir, config=config, output_dir=tmp_path / "a")
    second = run_dense(run_dir=run_dir, config=config, output_dir=tmp_path / "b")
    assert first.point_cloud_path.read_bytes() == second.point_cloud_path.read_bytes()


def test_dense_pipeline_respects_manual_depth_range(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)
    config = DenseConfig(
        max_width=WIDTH,
        num_sources=3,
        plane_step_px=3.0,
        min_planes=8,
        max_planes=32,
        depth_range="5:60",
        mesh=False,
    )
    result = run_dense(run_dir=run_dir, config=config)
    assert tuple(result.report.geometry.depth_range) == pytest.approx((5.0, 60.0))
    assert result.point_cloud_path.exists()


def test_intrinsics_roundtrip() -> None:
    intrinsics = Intrinsics(fx=FOCAL, fy=FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0, width=WIDTH, height=HEIGHT)
    matrix = intrinsics.matrix
    assert matrix[0, 0] == FOCAL
    assert matrix[1, 2] == HEIGHT / 2.0
    scaled = intrinsics.scaled(0.5)
    assert scaled.fx == pytest.approx(FOCAL * 0.5)


def test_prior_sky_guard_defaults_on() -> None:
    """Phase 1 sky guard: prior fill must refuse sky pixels unless opted out."""
    cfg = DenseConfig()
    assert cfg.prior_sky_guard is True
    assert cfg.prior_sky_max_row_fraction == 0.35
    assert DenseConfig(prior_sky_guard=False).prior_sky_guard is False


def test_mesh_voxel_fraction_defaults_coarse() -> None:
    """Final polish: the old 1.5 default meshed the acceptance run at
    3.36M verts / 162 MB — too heavy to demo and an OOM risk on 4M-point
    clouds. The default is now 5.0, mirroring the proven UAVLight
    direct-mesh ratio (~3.3x coarser)."""
    assert DenseConfig().mesh_voxel_fraction == 5.0
    assert DenseConfig(mesh_voxel_fraction=1.5).mesh_voxel_fraction == 1.5


def test_load_sparse_points_prefers_fresh_colmap_over_stale_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: after an SfM re-run the npz cache is stale; the COLMAP
    model must win, otherwise dense depth ranges come from the old run."""
    from uav3d.dense.geometry import load_sparse_points

    run_dir = tmp_path / "run"
    (run_dir / "colmap" / "sparse" / "0").mkdir(parents=True)
    stale = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
    np.savez(run_dir / "sparse_points.npz", points=stale)

    fresh = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)

    class _FakePoint:
        def __init__(self, xyz: np.ndarray) -> None:
            self.xyz = xyz

    class _FakeReconstruction:
        def __init__(self, path: str) -> None:
            assert "colmap" in path
            self.points3D = {i: _FakePoint(p) for i, p in enumerate(fresh)}

    monkeypatch.setitem(sys.modules, "pycolmap", types.SimpleNamespace(Reconstruction=_FakeReconstruction))

    assert np.allclose(load_sparse_points(run_dir), fresh)

    # No COLMAP model: the npz cache is the fallback.
    (run_dir / "colmap").rename(run_dir / "colmap_gone")
    assert np.allclose(load_sparse_points(run_dir), stale)

    # Neither: None.
    (run_dir / "sparse_points.npz").unlink()
    assert load_sparse_points(run_dir) is None


def test_rectified_cache_rebuilds_when_focal_changes(tmp_path: Path) -> None:
    """Regression: a dense re-run after SfM solved a new focal must not
    reuse the previous run's rectified geometry (that ran the whole dense
    stage with the old focal while the summary reported the new one)."""
    from uav3d.dense.cameras import CameraView, Intrinsics
    from uav3d.dense.pipeline import _load_or_build_rectified

    img = np.zeros((270, 480, 3), dtype=np.uint8)
    img[100:170, 200:280] = 255
    src_dir = tmp_path / "images"
    src_dir.mkdir()
    paths = []
    for i in range(3):
        p = src_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)

    def _views(focal: float) -> list[CameraView]:
        intr = Intrinsics(fx=focal, fy=focal, cx=240.0, cy=135.0, width=480, height=270)
        views = [
            CameraView(
                index=i,
                frame_id=f"video_{i:06d}",
                image_path=paths[i],
                R=np.eye(3),
                t=np.zeros(3),
                intrinsics=intr,
            )
            for i in range(3)
        ]
        # build_views() normally stashes these on the view objects.
        for view in views:
            view._native_intrinsics = intr  # type: ignore[attr-defined]
            view._distortion = np.zeros(0)  # type: ignore[attr-defined]
        return views

    out_dir = tmp_path / "rectified"
    # First run with the old focal: builds the cache.
    _, intr_old = _load_or_build_rectified(_views(447.0), out_dir, target_width=640)
    assert intr_old.fx == 447.0

    # Second run after SfM re-solved the focal: the cache must rebuild.
    _, intr_new = _load_or_build_rectified(_views(265.0), out_dir, target_width=640)
    assert intr_new.fx == 265.0
    assert json.loads((out_dir / "geometry.json").read_text())["fx"] == 265.0

    # Third run with the same focal: the cache is reused, not rebuilt.
    _, intr_cached = _load_or_build_rectified(_views(265.0), out_dir, target_width=640)
    assert intr_cached.fx == 265.0
