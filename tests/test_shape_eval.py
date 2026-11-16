"""Unit tests for reconstruction-versus-ground-truth shape evaluation."""

import numpy as np
import pytest

from uav3d.exceptions import DataError
from uav3d.validation.shape_eval import (
    CropBox,
    Similarity,
    align_clouds,
    best_axis_alignment,
    crop_from_alignment,
    refine_similarity,
    score_clouds,
    subsample,
    umeyama_similarity,
)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _block(rng: np.random.Generator, n: int = 600) -> np.ndarray:
    """A box-ish cloud: enough structure to constrain a similarity."""
    base = rng.normal(0.0, 1.0, size=(n, 3))
    base[:, 2] = np.abs(base[:, 2])
    return base


def test_umeyama_recovers_known_similarity() -> None:
    rng = np.random.default_rng(5)
    src = _block(rng)
    scale_true = 3.25
    R_true = _rot_z(0.7)
    t_true = np.array([2.0, -1.0, 0.5])
    dst = scale_true * src @ R_true.T + t_true

    sim = umeyama_similarity(src, dst)
    assert sim.scale == pytest.approx(scale_true, rel=1e-9)
    np.testing.assert_allclose(sim.rotation, R_true, atol=1e-9)
    np.testing.assert_allclose(sim.translation, t_true, atol=1e-8)
    assert not sim.is_degenerate


def test_axis_search_finds_permuted_frame() -> None:
    rng = np.random.default_rng(9)
    src = _block(rng)
    # dst is src with axes swapped: Y-up reconstruction vs Z-up reference.
    dst = np.stack([src[:, 0], -src[:, 2], src[:, 1]], axis=1)

    coarse, index = best_axis_alignment(src, dst)
    mapped = coarse.apply(src)
    assert np.allclose(mapped, dst, atol=1e-6)
    assert 0 <= index < 24


def test_axis_search_initialises_scale_from_spread() -> None:
    rng = np.random.default_rng(37)
    src = _block(rng)
    dst = 4.0 * src
    coarse, _ = best_axis_alignment(src, dst)
    assert coarse.scale == pytest.approx(4.0, rel=1e-9)


def test_refine_recovers_scale_from_coarse_rotation() -> None:
    rng = np.random.default_rng(11)
    src = _block(rng)
    dst = 2.0 * src + np.array([5.0, 0.0, -3.0])

    coarse, _ = best_axis_alignment(src, dst)
    refined = refine_similarity(src, dst, coarse, iterations=2)
    assert refined.scale == pytest.approx(2.0, rel=1e-6)
    np.testing.assert_allclose(refined.apply(src), dst, atol=1e-6)


def _corner_scene(rng: np.random.Generator, n: int = 6000) -> np.ndarray:
    """Points on three orthogonal planes with *different* extents.

    A volume of random noise has no identifiable shape, so scale cannot be
    recovered from it; real reconstructions and references are both *surfaces*.
    The extents differ per plane so the scene is not symmetric under axis
    permutation — otherwise the correct rotation would be ambiguous and the test
    could not assert anything about which way the model points.
    """
    per_plane = n // 3
    xy = rng.uniform(0.0, 1.0, size=(per_plane, 2)) * np.array([6.0, 3.0])
    floor = np.column_stack([xy, np.zeros(per_plane)])
    xz = rng.uniform(0.0, 1.0, size=(per_plane, 2)) * np.array([6.0, 4.0])
    wall_x = np.column_stack([xz[:, 0], np.zeros(per_plane), xz[:, 1]])
    yz = rng.uniform(0.0, 1.0, size=(per_plane, 2)) * np.array([3.0, 4.0])
    wall_y = np.column_stack([np.zeros(per_plane), yz[:, 0], yz[:, 1]])
    return np.vstack([floor, wall_x, wall_y])


def test_refine_ignores_unmatched_reference_bulk() -> None:
    """Refinement must not be dragged outward by reference the recon never saw.

    The scene spans x 0..6; the recon is a half-scale copy of only the x < 3
    half, and the initial similarity is exactly correct. Scale is *not*
    recoverable from partial-overlap geometry alone (a shrunk copy of a half is
    indistinguishable from a full copy in several ways), so the contract tested
    here is narrower and is what matters in practice: given a correct start,
    the unobserved far half of the reference must not pull the fit away from it.
    """
    rng = np.random.default_rng(13)
    gt = _corner_scene(rng, 9000)
    subregion = gt[gt[:, 0] < 3.0]
    recon = 0.5 * subregion
    exact = Similarity(scale=2.0, rotation=np.eye(3), translation=np.zeros(3))

    refined = refine_similarity(recon, gt, exact, iterations=3)
    assert refined.scale == pytest.approx(2.0, rel=0.05)

    mapped = refined.apply(recon)
    # It must stay on the half it covers instead of spreading across 6 units.
    assert np.ptp(mapped[:, 0]) < 3.6
    assert np.ptp(mapped[:, 1]) < 3.6
    # And it must still sit on the reference surface.
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(gt).query(mapped, k=1, workers=-1)
    assert float(np.median(dist)) < 0.15


def test_score_clouds_perfect_overlap() -> None:
    rng = np.random.default_rng(17)
    cloud = _block(rng, 500)
    sim = umeyama_similarity(cloud, cloud)
    metrics = score_clouds(cloud, cloud, sim, thresholds=(0.05,))

    assert metrics["chamfer"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["precision_at_0.05"] == pytest.approx(1.0)
    assert metrics["completeness_at_0.05"] == pytest.approx(1.0)
    assert metrics["fscore_at_0.05"] == pytest.approx(1.0)


def test_score_clouds_penalises_displacement() -> None:
    """A displaced surface scores the displacement, not the point spacing.

    Chamfer is a surface metric: a *volume* cloud's nearest neighbour is inside
    the volume at roughly the point spacing regardless of how far the cloud
    moved, so the test uses a surface (a plane) where displacement is
    unambiguous.
    """
    rng = np.random.default_rng(41)
    surface = np.column_stack(
        [rng.uniform(-1.0, 1.0, 8000), rng.uniform(-1.0, 1.0, 8000), np.zeros(8000)]
    )
    shift = 0.5
    shifted = surface + np.array([0.0, 0.0, shift])
    sim = umeyama_similarity(surface, surface)
    metrics = score_clouds(surface, shifted, sim, thresholds=(0.05,))

    assert metrics["chamfer"] == pytest.approx(shift, rel=1e-6)
    assert metrics["precision_at_0.05"] == pytest.approx(0.0)
    assert metrics["completeness_at_0.05"] == pytest.approx(0.0)
    assert metrics["fscore_at_0.05"] == pytest.approx(0.0)


def test_score_reports_precision_and_completeness_separately() -> None:
    """Half the reference covered, nothing spurious -> high precision, low completeness."""
    rng = np.random.default_rng(23)
    gt = _block(rng, 2000)
    recon = gt[gt[:, 0] > 0.0]
    sim = umeyama_similarity(recon, recon)

    metrics = score_clouds(recon, gt, sim, thresholds=(0.05,))
    assert metrics["precision_at_0.05"] > 0.99
    assert metrics["completeness_at_0.05"] == pytest.approx(0.5, abs=0.05)


def test_crop_box_parse_and_contains() -> None:
    box = CropBox.parse("-1,-2,-3,1,2,3")
    assert np.allclose(box.minimum, [-1.0, -2.0, -3.0])
    assert np.allclose(box.maximum, [1.0, 2.0, 3.0])
    pts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert box.contains(pts).tolist() == [True, False, True]


def test_crop_box_rejects_bad_input() -> None:
    with pytest.raises(DataError):
        CropBox.parse("1,2,3")
    with pytest.raises(DataError):
        CropBox.parse("a,b,c,d,e,f")
    with pytest.raises(DataError):
        CropBox.parse("1,1,1,0,0,0")


def test_crop_from_alignment_brackets_the_reconstruction() -> None:
    rng = np.random.default_rng(29)
    recon = _block(rng, 400)
    gt = _block(rng, 2000)
    sim, _ = align_clouds(recon, gt)

    box = crop_from_alignment(recon, sim, margin=0.1)
    mapped = sim.apply(recon)
    assert np.all(box.contains(mapped))
    assert np.all(box.extent > np.ptp(mapped, axis=0))


def test_subsample_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(31)
    pts = _block(rng, 1000)
    a = subsample(pts, 100, seed=0)
    b = subsample(pts, 100, seed=0)
    assert len(a) == 100
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(subsample(pts, 5000, seed=0), pts)
