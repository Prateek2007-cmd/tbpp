"""Tests for best-view vertex texturing."""

from pathlib import Path

import numpy as np
import pytest

from uav3d.dense.cameras import CameraView, Intrinsics
from uav3d.modeling.texture_bake import (
    bilinear_sample_rgb,
    choose_vertex_views,
    gravity_transform,
    sample_vertex_colors,
    vertex_normals,
)


def _view(index: int, r: np.ndarray | None = None, t: np.ndarray | None = None) -> CameraView:
    intr = Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0, width=64, height=64)
    return CameraView(
        index=index,
        frame_id=f"v{index}",
        image_path=Path(f"v{index}.jpg"),
        R=np.eye(3) if r is None else r,
        t=np.zeros(3) if t is None else t,
        intrinsics=intr,
    )


def test_vertex_normals_of_plane() -> None:
    vertices = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    normals = vertex_normals(vertices, faces)
    assert normals.shape == (4, 3)
    assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)


def test_bilinear_sample_hits_texture() -> None:
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    colors, inside = bilinear_sample_rgb(image, np.array([0.5]), np.array([0.5]))
    assert inside[0]
    assert np.allclose(colors[0], [64, 64, 64], atol=1)

    colors, inside = bilinear_sample_rgb(image, np.array([-1.0]), np.array([0.5]))
    assert not inside[0]
    assert colors[0].tolist() == [0, 0, 0]


def test_choose_views_rejects_occluded_camera() -> None:
    vertices = np.array([[0.0, 0.0, 1.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    good = _view(0)
    occluded = _view(1)
    depths = {0: np.ones((64, 64), np.float32), 1: np.full((64, 64), 0.5, np.float32)}

    best_view, best_score = choose_vertex_views(
        vertices,
        normals,
        [good, occluded],
        depths,
        facing_min=0.1,
    )
    assert np.all(best_view == 0)
    assert np.all(best_score > 0)


def test_sample_vertex_colors_uses_chosen_view() -> None:
    vertices = np.array([[0.0, 0.0, 1.0]])
    best = np.array([1], dtype=np.int16)
    views = [_view(0), _view(1)]
    red = np.zeros((64, 64, 3), np.uint8)
    red[:, :] = [255, 0, 0]
    green = np.zeros((64, 64, 3), np.uint8)
    green[:, :] = [0, 255, 0]
    fallback = np.array([[10, 10, 10]], dtype=np.uint8)
    colors = sample_vertex_colors(vertices, best, views, lambda i: red if i == 0 else green, fallback)
    assert colors[0].tolist() == [0, 255, 0]


def test_gravity_transform_lifts_a_tilted_plane() -> None:
    axis = np.array([0.3, 0.0, 0.954])
    u = np.array([0.954, 0.0, -0.3])
    v = np.array([0.0, 1.0, 0.0])
    grid = np.stack(
        np.meshgrid(np.linspace(-5.0, 5.0, 20), np.linspace(-5.0, 5.0, 20), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    points = grid[:, :1] * u + grid[:, 1:2] * v
    R, offset = gravity_transform(points)
    heights = points @ R.T
    heights = heights[:, 2] + offset
    assert np.ptp(heights) < 1e-6
    assert abs(float(np.min(heights))) < 1e-6


def test_gravity_transform_has_upward_sign() -> None:
    rng = np.random.default_rng(0)
    ground = rng.normal(0.0, 2.0, (500, 3))
    ground[:, 2] = 0.0
    towers = ground[:100] + np.array([0.0, 0.0, 5.0])
    R, offset = gravity_transform(np.vstack([ground, towers]))
    heights = towers @ R.T
    heights = heights[:, 2] + offset
    assert float(np.median(heights)) > 0.0
