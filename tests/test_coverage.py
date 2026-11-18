"""Unit tests for per-surface observation coverage (Phase 4).

The synthetic scene is three downward-looking cameras at height 10 over a
flat ground plane (z=0), with constant depth maps of 10.0 — so ground vertices
are observed by all three views, an elevated vertex is observed by none (its
depth disagrees with the maps), and a far-away vertex is observed by none
(it projects outside every frame).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav3d.coverage import (
    count_observations,
    coverage_report_for_run,
    human_summary,
    read_glb_positions,
    summarize_coverage,
)
from uav3d.dense.cameras import CameraView, Intrinsics
from uav3d.dense.glb_export import write_glb
from uav3d.exceptions import DataError

# 180-degree rotation about X: camera looks down (-Z world becomes +Z cam).
R_DOWN = np.diag([1.0, -1.0, -1.0])
INTR = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=64, height=48)


def _down_view(index: int, x0: float) -> CameraView:
    """Downward camera at world (x0, 0, 10). t = -R @ C."""
    centre = np.array([x0, 0.0, 10.0])
    return CameraView(
        index=index,
        frame_id=f"frame_{index:03d}",
        image_path=Path(f"img_{index}.png"),
        R=R_DOWN.copy(),
        t=-(R_DOWN @ centre),
        intrinsics=INTR,
    )


def _ground_grid() -> np.ndarray:
    xs = np.arange(-2.0, 3.0)
    ys = np.arange(-2.0, 3.0)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(25)], axis=1)


def test_count_observations_synthetic() -> None:
    views = [_down_view(0, -2.0), _down_view(1, 0.0), _down_view(2, 2.0)]
    depth_maps = {i: np.full((48, 64), 10.0) for i in range(3)}
    vertices = np.vstack([_ground_grid(), [[0.0, 0.0, 5.0], [50.0, 50.0, 0.0]]])

    counts = count_observations(vertices, views, depth_maps, tolerance=0.15)

    assert counts.shape == (27,)
    assert (counts[:25] == 3).all(), "ground grid must be observed by all 3 views"
    assert counts[25] == 0, "elevated vertex disagrees with every depth map"
    assert counts[26] == 0, "far vertex projects outside every frame"


def test_count_observations_partial_and_holes() -> None:
    views = [_down_view(0, -2.0), _down_view(1, 0.0), _down_view(2, 2.0)]
    depth_maps = {i: np.full((48, 64), 10.0) for i in range(3)}
    # View 1 has a hole (zeros) over the centre of the frame.
    depth_maps[1][20:28, 28:36] = 0.0
    # View 2 is entirely invalid.
    depth_maps[2][:] = 0.0
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])

    counts = count_observations(vertices, views, depth_maps, tolerance=0.15)

    # Centre vertex: view 0 yes, view 1 hole, view 2 invalid -> 1.
    assert counts[0] == 1
    # Corner vertex: view 0 yes, view 1 yes (outside the hole), view 2 no -> 2.
    assert counts[1] == 2


def test_count_observations_tolerance_boundary() -> None:
    views = [_down_view(0, 0.0)]
    depth_maps = {0: np.full((48, 64), 10.0)}
    vertices = np.array([[0.0, 0.0, 0.0]])  # true depth 10.0
    depth_maps[0][:] = 10.2  # 0.2 off

    assert count_observations(vertices, views, depth_maps, tolerance=0.15)[0] == 0
    assert count_observations(vertices, views, depth_maps, tolerance=0.25)[0] == 1


def test_summarize_coverage() -> None:
    counts = np.array([0, 0, 1, 2, 3, 5, 5], dtype=np.uint16)
    summary = summarize_coverage(counts)
    assert summary["num_vertices"] == 7
    assert summary["fraction_observed_ge_2"] == pytest.approx(4 / 7)
    assert summary["fraction_observed_ge_3"] == pytest.approx(3 / 7)
    assert summary["fraction_observed_ge_5"] == pytest.approx(2 / 7)
    assert summary["fraction_unobserved"] == pytest.approx(2 / 7)
    assert summary["mean_observations"] == pytest.approx(16 / 7)
    assert summary["median_observations"] == pytest.approx(2.0)


def test_summarize_coverage_empty() -> None:
    summary = summarize_coverage(np.zeros(0, dtype=np.uint16))
    assert summary["num_vertices"] == 0
    assert summary["fraction_observed_ge_2"] == 0.0


def test_human_summary_is_honest() -> None:
    summary = summarize_coverage(np.array([3, 3, 0], dtype=np.uint16))
    text = human_summary(summary)
    assert "66.7%" in text
    assert "no geometry is invented there" in text


def test_read_glb_positions_roundtrip(tmp_path: Path) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    colors = np.array([[1.0, 0.0, 0.0]] * 4)
    glb = write_glb(tmp_path / "tiny.glb", vertices, faces, colors)
    read_back = read_glb_positions(glb)
    # write_glb rotates Z-up -> Y-up on export: (x, y, z) -> (x, z, -y).
    # Order must be preserved: coverage arrays index vertices by position.
    assert read_back.shape == vertices.shape
    expected_gltf = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, -1.0]]
    )
    np.testing.assert_allclose(read_back, expected_gltf, rtol=0, atol=1e-6)
    # And the inverse rotation recovers the SfM-frame vertices exactly.
    from uav3d.coverage import gltf_positions_to_sfm

    np.testing.assert_allclose(gltf_positions_to_sfm(read_back), vertices, rtol=0, atol=1e-6)


def test_read_glb_positions_rejects_non_glb(tmp_path: Path) -> None:
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all")
    with pytest.raises(DataError, match="Not a GLB"):
        read_glb_positions(bad)


def _write_fixture_run(run_dir: Path, scale_value: float = 1.0) -> np.ndarray:
    """Build a minimal run dir: 3 poses, 3 images, depth maps, mesh GLB."""
    images_dir = run_dir / "sfm" / "images"
    images_dir.mkdir(parents=True)
    dense_dir = run_dir / "dense"
    (dense_dir / "depth_filtered").mkdir(parents=True)

    poses = []
    keyframes = []
    for i, x0 in enumerate((-2.0, 0.0, 2.0)):
        frame_id = f"frame_{i:03d}"
        img = images_dir / f"{frame_id}.png"
        cv2.imwrite(str(img), np.zeros((48, 64, 3), dtype=np.uint8))
        centre = np.array([x0, 0.0, 10.0])
        t = -(R_DOWN @ centre)
        poses.append(
            {
                "frame_id": frame_id,
                "qw": 0.0,
                "qx": 1.0,
                "qy": 0.0,
                "qz": 0.0,
                "tx": float(t[0]),
                "ty": float(t[1]),
                "tz": float(t[2]),
                "registered": True,
            }
        )
        keyframes.append({"frame_id": frame_id, "file_path": f"sfm/images/{frame_id}.png"})
        np.save(dense_dir / "depth_filtered" / f"{i:05d}.npy", np.full((48, 64), 10.0))

    (run_dir / "poses.json").write_text(
        json.dumps(
            {
                "camera": {
                    "model": "SIMPLE_PINHOLE",
                    "width": 64,
                    "height": 48,
                    "params": [50.0, 32.0, 24.0],
                },
                "metrics": {
                    "num_images_total": 3,
                    "num_images_registered": 3,
                    "registration_rate": 1.0,
                    "mean_reprojection_error_px": 0.3,
                },
                "poses": poses,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "keyframes.json").write_text(
        json.dumps({"keyframes": keyframes}), encoding="utf-8"
    )
    (dense_dir / "dense_report.json").write_text(
        json.dumps(
            {
                "geometry": {"working_width": 64, "median_depth": 10.0},
                "fusion": {"depth_sigma_units": 0.05},
                "scale": {"value": scale_value, "source": "gnss", "status": "gnss_aligned"},
            }
        ),
        encoding="utf-8",
    )

    vertices = np.vstack([_ground_grid(), [[0.0, 0.0, 5.0], [50.0, 50.0, 0.0]]])
    # Triangulate the 5x5 grid; the last two vertices stay unreferenced.
    faces = []
    for r in range(4):
        for c in range(4):
            a, b = r * 5 + c, r * 5 + c + 1
            d, e = (r + 1) * 5 + c, (r + 1) * 5 + c + 1
            faces.append([a, b, e])
            faces.append([a, e, d])
    mesh_vertices = vertices * scale_value  # mesh is stored in metres
    write_glb(
        dense_dir / "mesh.glb",
        mesh_vertices,
        np.array(faces, dtype=np.int64),
        np.ones((27, 3)) * 0.5,
    )
    return vertices


def test_coverage_report_for_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_fixture"
    run_dir.mkdir()
    _write_fixture_run(run_dir)

    report = coverage_report_for_run(run_dir)

    assert report["num_vertices"] == 27
    assert report["num_views_with_depth"] == 3
    assert report["fraction_observed_ge_2"] == pytest.approx(25 / 27)
    assert report["fraction_observed_ge_3"] == pytest.approx(25 / 27)
    assert report["fraction_observed_ge_5"] == pytest.approx(0.0)
    assert report["fraction_unobserved"] == pytest.approx(2 / 27)
    assert "no geometry is invented there" in report["summary"]
    assert (run_dir / "dense" / "coverage_report.json").exists()
    counts_file = json.loads((run_dir / "dense" / "coverage.json").read_text())
    assert counts_file["num_vertices"] == 27
    assert counts_file["counts"][:25] == [3] * 25
    assert counts_file["counts"][25:] == [0, 0]


def test_coverage_report_unscales_metre_mesh(tmp_path: Path) -> None:
    """Mesh vertices are in metres; depth maps are in SfM units — the driver
    must divide by the scale before projecting, or every vertex disagrees."""
    run_dir = tmp_path / "run_scaled"
    run_dir.mkdir()
    _write_fixture_run(run_dir, scale_value=2.0)

    report = coverage_report_for_run(run_dir)

    assert report["scale_m_per_unit"] == pytest.approx(2.0)
    assert report["fraction_observed_ge_3"] == pytest.approx(25 / 27)


def test_coverage_report_missing_mesh(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_nomesh"
    run_dir.mkdir()
    _write_fixture_run(run_dir)
    (run_dir / "dense" / "mesh.glb").unlink()
    with pytest.raises(DataError, match="No mesh"):
        coverage_report_for_run(run_dir)


def test_coverage_report_missing_dense(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    with pytest.raises(DataError, match="dense_report"):
        coverage_report_for_run(run_dir)
