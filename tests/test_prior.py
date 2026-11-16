"""Unit tests for the monocular depth prior alignment and gated fill."""

from pathlib import Path

import numpy as np

from uav3d.dense.cameras import CameraView, Intrinsics
from uav3d.dense.prior import depth_from_prior, fill_from_prior, fit_inverse_depth
from uav3d.dense.sweep import DepthEstimate

N = 400


def _depths() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(2.0, 20.0, size=N)


def test_fit_recovers_plain_affine_inverse_depth() -> None:
    depth = _depths()
    raw = 1.0 / depth
    result = fit_inverse_depth(raw, depth)
    assert result is not None
    scale, shift, corr, err, inverted = result
    assert inverted is False
    assert corr > 0.99
    assert err < 0.01
    assert np.allclose(scale, 1.0, rtol=1e-6)
    assert np.allclose(shift, 0.0, atol=1e-9)
    assert np.allclose(depth_from_prior(raw, scale, shift, inverted), depth, rtol=1e-4)


def test_fit_recovers_depth_ordered_map() -> None:
    depth = _depths()
    raw = 1.0 / (3.0 / depth + 1.5)
    result = fit_inverse_depth(raw, depth)
    assert result is not None
    scale, shift, corr, err, inverted = result
    assert inverted is True
    assert corr > 0.99
    assert err < 0.01
    assert np.allclose(depth_from_prior(raw, scale, shift, inverted), depth, rtol=1e-3)


def test_fit_rejects_uncorrelated_map() -> None:
    rng = np.random.default_rng(11)
    raw = rng.uniform(0.0, 1.0, size=N)
    depth = _depths()
    assert fit_inverse_depth(raw, depth, min_rank_corr=0.9) is None


def test_fit_rejects_tiny_sample() -> None:
    rng = np.random.default_rng(13)
    depth = rng.uniform(2.0, 20.0, size=10)
    assert fit_inverse_depth(1.0 / depth, depth) is None


def _view(index: int) -> CameraView:
    intr = Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=24.0, width=64, height=48)
    return CameraView(
        index=index,
        frame_id=f"v{index}",
        image_path=Path(f"v{index}.jpg"),
        R=np.eye(3),
        t=np.zeros(3),
        intrinsics=intr,
    )


def _estimate_with_one_hole() -> DepthEstimate:
    depth = np.full((48, 64), 4.0, dtype=np.float32)
    valid = np.ones((48, 64), dtype=bool)
    valid[24, 32] = False
    depth[24, 32] = 0.0
    return DepthEstimate(
        depth=depth,
        cost=np.zeros_like(depth),
        num_views=np.ones_like(depth, dtype=np.uint8),
        valid=valid,
    )


def test_fill_prior_requires_neighbour_agreement() -> None:
    ref = _view(0)
    neighbour = _view(1)
    estimate = _estimate_with_one_hole()
    prior = np.full((48, 64), 6.0, dtype=np.float32)

    filled, count = fill_from_prior(
        estimate,
        prior,
        ref,
        [neighbour],
        [np.full((48, 64), 6.0, dtype=np.float32)],
        near=2.0,
        far=12.0,
        min_support=1,
    )
    assert count == 1
    assert filled.valid[24, 32]
    assert filled.depth[24, 32] == 6.0
    measured = estimate.valid
    assert np.allclose(filled.depth[measured], estimate.depth[measured])

    rejected, count_rejected = fill_from_prior(
        estimate,
        prior,
        ref,
        [neighbour],
        [np.full((48, 64), 2.0, dtype=np.float32)],
        near=2.0,
        far=12.0,
        min_support=1,
    )
    assert count_rejected == 0
    assert not rejected.valid[24, 32]
    assert rejected.depth[24, 32] == 0.0
