"""Reconstruction-versus-ground-truth shape evaluation.

Separated from ``scripts/eval_city_model.py`` so the alignment and cropping logic
is unit-testable without loading GLB files.

The alignment is deliberately *not* a global least-squares fit against the whole
ground-truth cloud: the reference can cover hundreds of metres while a single-pass
reconstruction covers one block, so a global fit aligns the wrong thing and
reports a Chamfer distance dominated by unobserved ground truth. The caller
supplies a crop of the reference that corresponds to what was actually flown.

Alignment is a 24-candidate axis search followed by two robust Sim(3)
refinements on mutual nearest neighbours, because the axis search alone is only
accurate to the voxel spacing of the sampled clouds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from uav3d.exceptions import DataError


@dataclass(frozen=True)
class Similarity:
    """``dst ~= scale * R @ src + t``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``src`` points into the destination frame."""
        return self.scale * points @ self.rotation.T + self.translation

    @property
    def is_degenerate(self) -> bool:
        """True when the fit cannot meaningfully constrain rotation."""
        return not np.isfinite(self.scale) or self.scale <= 0.0


@dataclass(frozen=True)
class CropBox:
    """Axis-aligned box in ground-truth coordinates (metres)."""

    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def parse(cls, text: str) -> CropBox:
        """Parse ``xmin,ymin,zmin,xmax,ymax,zmax``."""
        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) != 6:
            raise DataError(
                f"--gt-crop expects 6 comma-separated values (xmin,ymin,zmin,xmax,ymax,zmax), got {text!r}"
            )
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            raise DataError(f"--gt-crop contains a non-numeric value: {text!r}") from exc
        minimum = np.array(values[:3], dtype=np.float64)
        maximum = np.array(values[3:], dtype=np.float64)
        if np.any(maximum <= minimum):
            raise DataError(f"--gt-crop maximum must exceed minimum on every axis: {text!r}")
        return cls(minimum=minimum, maximum=maximum)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the box (inclusive)."""
        return np.all((points >= self.minimum) & (points <= self.maximum), axis=1)

    def padded(self, fraction: float) -> CropBox:
        """Return the box expanded by ``fraction`` of its extent on every side."""
        extent = self.maximum - self.minimum
        pad = extent * max(0.0, float(fraction))
        return CropBox(minimum=self.minimum - pad, maximum=self.maximum + pad)

    @property
    def extent(self) -> np.ndarray:
        """Box side lengths."""
        return self.maximum - self.minimum


def axis_rotations() -> list[np.ndarray]:
    """The 24 proper axis-aligned rotations (all permutations with det=+1)."""
    mats: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                j, s = perm[i], signs[i]
                R[i, j] = s
            if np.linalg.det(R) > 0:
                mats.append(R)
    return mats


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Least-squares similarity mapping ``src`` onto ``dst``."""
    if src.shape != dst.shape or src.shape[1] != 3:
        raise DataError("umeyama_similarity expects two (N, 3) arrays of equal length")
    if src.shape[0] < 3:
        raise DataError("umeyama_similarity needs at least 3 correspondences")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    s0 = src - mu_src
    d0 = dst - mu_dst
    cov = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = float((s0**2).sum() / src.shape[0])
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * R @ mu_src
    return Similarity(scale=scale, rotation=R, translation=t)


def _nn_distances(src: np.ndarray, dst: np.ndarray, tree: cKDTree | None = None) -> np.ndarray:
    tree = tree or cKDTree(dst)
    d, _ = tree.query(src, k=1, workers=-1)
    return np.asarray(d, dtype=np.float64)


def _symmetric_mean(src: np.ndarray, dst: np.ndarray) -> float:
    tree_dst = cKDTree(dst)
    tree_src = cKDTree(src)
    return float(_nn_distances(src, dst, tree_dst).mean() + _nn_distances(dst, src, tree_src).mean())


def fit_fixed_rotation(src: np.ndarray, dst: np.ndarray, rotation: np.ndarray) -> Similarity:
    """Optimal ``(scale, t)`` for a *given* rotation — closed-form least squares.

    Needed because a std-ratio scale estimate is invalid as soon as the
    translation differs per axis (three different offsets inflate the global
    spread). With the rotation fixed, the minimum of
    ``||s * R @ src + t - dst||^2`` has a closed form.
    """
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    s0 = src - mu_src
    d0 = dst - mu_dst
    rotated = s0 @ rotation.T
    denom = float((s0**2).sum())
    if denom <= 1e-12:
        raise DataError("fit_fixed_rotation needs a source cloud with non-zero spread")
    scale = float((d0 * rotated).sum() / denom)
    translation = mu_dst - scale * rotation @ mu_src
    return Similarity(scale=scale, rotation=rotation, translation=translation)


def best_axis_alignment(src: np.ndarray, dst: np.ndarray, max_points: int = 20000) -> tuple[Similarity, int]:
    """Coarse alignment: pick the axis rotation whose clouds overlap best.

    Rotations are scored on centred, unit-normalised clouds so the comparison is
    scale-free and identical across candidates. The returned similarity is then
    refitted in original units for the winning rotation, which is what lets the
    subsequent refinement pair points correctly instead of starting from a wrong
    scale.

    Args:
        src: reconstruction points.
        dst: reference points.
        max_points: cap for the scoring subsample, so 24 rotations stay cheap.

    Returns:
        ``(similarity, rotation_index)`` mapping ``src`` onto ``dst``.
    """
    if src.shape[0] < 3 or dst.shape[0] < 3:
        raise DataError("best_axis_alignment needs at least 3 points in each cloud")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_std = float(np.std(src - src_mean))
    dst_std = float(np.std(dst - dst_mean))
    if src_std <= 1e-9 or dst_std <= 1e-9:
        raise DataError("best_axis_alignment needs clouds with non-zero spread")

    src_sample = subsample(src - src_mean, max_points)
    dst_sample = subsample(dst - dst_mean, max_points)
    src_u = src_sample / src_std
    dst_u = dst_sample / dst_std

    best_score = np.inf
    best_R = np.eye(3)
    best_index = 0
    for index, R in enumerate(axis_rotations()):
        score = _symmetric_mean(src_u @ R.T, dst_u)
        if score < best_score:
            best_score = score
            best_R = R
            best_index = index

    # Build correspondences in the normalised frame, where rotation-only nearest
    # neighbours are already meaningful, then fit scale/translation on the
    # matching points in original units. The pairs must be selected with the
    # same sample indices on both sides, or they refer to different points.
    rotated = src_u @ best_R.T
    dist, idx = cKDTree(dst_u).query(rotated, k=1, workers=-1)
    # Partial coverage leaves the normalised clouds with different spreads, so
    # gate the correspondences to the nearest half before fitting.
    dist = np.asarray(dist, dtype=np.float64)
    cutoff = float(np.quantile(dist, 0.5))
    keep = dist <= max(cutoff, np.finfo(np.float64).eps)
    matched_src = src_sample[keep] + src_mean
    matched_dst = dst_sample[idx[keep]] + dst_mean
    if len(matched_src) < 3:
        matched_src, matched_dst = src, dst[: len(src)]
    similarity = umeyama_similarity(matched_src, matched_dst)
    return Similarity(
        scale=similarity.scale,
        rotation=best_R,
        translation=similarity.translation,
    ), best_index


def refine_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    initial: Similarity,
    iterations: int = 2,
    trim_fraction: float = 0.9,
    max_distance_fraction: float = 0.15,
) -> Similarity:
    """Refine a similarity on nearest neighbours, trimming far and worst pairs.

    Two trims, because they solve different problems:

    * ``max_distance_fraction`` drops pairs whose distance exceeds that fraction
      of the reference spread. A reconstruction that covers one block of a
      300 m reference would otherwise be pulled outward by the majority of the
      reference it never observed.
    * ``trim_fraction`` drops the worst remaining tail, so a handful of bad
      correspondences cannot dominate the least-squares fit.

    Pairs are required to be *mutual* nearest neighbours. Without that, points
    near the edge of a partially covered scene pair with reference points just
    across the boundary, which biases the fitted scale outward.
    """
    if not 0.0 < trim_fraction <= 1.0:
        raise DataError("trim_fraction must be in (0, 1]")
    if max_distance_fraction <= 0.0:
        raise DataError("max_distance_fraction must be positive")

    current = initial
    threshold = float(max_distance_fraction) * float(np.std(dst))
    for _ in range(max(0, int(iterations))):
        mapped = current.apply(src)
        tree_dst = cKDTree(dst)
        tree_src = cKDTree(mapped)
        dist, idx = tree_dst.query(mapped, k=1, workers=-1)
        dist = np.asarray(dist, dtype=np.float64)
        _, back = tree_src.query(dst, k=1, workers=-1)
        mutual = np.asarray(back, dtype=np.int64)[idx] == np.arange(len(mapped))

        eligible = np.flatnonzero(mutual & (dist <= threshold))
        if eligible.size < 3:
            eligible = np.flatnonzero(dist <= threshold)
        if eligible.size < 3:
            # The initial alignment is too far off for the gates; fall back to
            # all pairs so the fit can pull itself in.
            eligible = np.arange(dist.size)
        keep_n = max(3, int(eligible.size * trim_fraction))
        order = eligible[np.argsort(dist[eligible])[:keep_n]]

        pairs_src = src[order]
        pairs_dst = dst[idx[order]]
        if len(pairs_src) < 3:
            break
        current = umeyama_similarity(pairs_src, pairs_dst)
        if current.is_degenerate:
            break
    return current


def align_from_trajectory(
    estimate: np.ndarray,
    reference: np.ndarray,
) -> Similarity:
    """Exact Sim(3) from matched camera positions, instead of a shape search.

    Camera trajectories are the right registration handle for evaluating a
    reconstruction against a large reference: the cameras are *observed*, they
    pin down both scale and the region actually flown, and they are what the
    reconstruction is built from. Fitting the surface clouds directly is both
    ill-conditioned (a 300 m reference is mostly ground plane, so scale can
    slide along the plane) and biased (the reconstruction covers one corridor
    of it, so its moments are not the reference's moments).

    Args:
        estimate: ``(N, 3)`` reconstructed camera centres in the SfM frame.
        reference: ``(N, 3)`` the same cameras' reference positions, in order.

    Returns:
        The similarity mapping the SfM frame into the reference frame.
    """
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if est.shape != ref.shape or est.ndim != 2 or est.shape[1] != 3:
        raise DataError(
            f"align_from_trajectory expects two (N, 3) arrays of equal length, got {est.shape} and {ref.shape}"
        )
    if len(est) < 3:
        raise DataError("align_from_trajectory needs at least 3 matched camera positions")
    return umeyama_similarity(est, ref)


def align_clouds(src: np.ndarray, dst: np.ndarray, iterations: int = 2) -> tuple[Similarity, int]:
    """Full alignment: axis search then robust Sim(3) refinement."""
    coarse, index = best_axis_alignment(src, dst)
    return refine_similarity(src, dst, coarse, iterations=iterations), index


def compose_similarities(outer: Similarity, inner: Similarity) -> Similarity:
    """Compose two similarities so the result is ``outer(inner(x))``.

    Used to chain a model->SfM fit (to undo an exporter's gravity rotation and
    rescale) with an SfM->reference fit (from the camera trajectory).
    """
    return Similarity(
        scale=outer.scale * inner.scale,
        rotation=outer.rotation @ inner.rotation,
        translation=outer.scale * (outer.rotation @ inner.translation) + outer.translation,
    )



def score_clouds(
    recon: np.ndarray,
    gt: np.ndarray,
    similarity: Similarity,
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5),
) -> dict[str, float]:
    """Chamfer distance, precision, completeness and F-score in GT units.

    Precision is the fraction of reconstructed points within ``t`` of the
    reference (are the points where they should be); completeness is the
    fraction of reference points within ``t`` of the reconstruction (did we
    cover the scene). Units are whatever the ground truth is in — metres for the
    city reference.
    """
    mapped = similarity.apply(recon)
    d_recon_to_gt = _nn_distances(mapped, gt)
    d_gt_to_recon = _nn_distances(gt, mapped, cKDTree(mapped))

    combined = np.concatenate([d_recon_to_gt, d_gt_to_recon])
    result: dict[str, float] = {
        "chamfer": float(0.5 * (d_recon_to_gt.mean() + d_gt_to_recon.mean())),
        "chamfer_median": float(np.median(combined)),
        "chamfer_rmse": float(np.sqrt(np.mean(combined**2))),
        "accuracy_median": float(np.median(d_recon_to_gt)),
        "completeness_median": float(np.median(d_gt_to_recon)),
    }
    for t in thresholds:
        precision = float((d_recon_to_gt < t).mean())
        completeness = float((d_gt_to_recon < t).mean())
        fscore = 2.0 * precision * completeness / max(precision + completeness, 1e-12)
        label = f"{t:g}"
        result[f"precision_at_{label}"] = precision
        result[f"completeness_at_{label}"] = completeness
        result[f"fscore_at_{label}"] = fscore
    return result


def subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministically subsample a cloud to at most ``max_points``."""
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), int(max_points), replace=False)
    return points[np.sort(idx)]


def crop_from_alignment(
    recon: np.ndarray,
    similarity: Similarity,
    margin: float = 0.10,
    percentile: float = 0.0,
) -> CropBox:
    """Derive a GT crop from where the aligned reconstruction sits.

    Used when no explicit ``--gt-crop`` is given: the reconstruction is mapped
    into GT coordinates and its bounding box becomes the crop, padded by
    ``margin``. This is what makes the evaluation comparable between models of
    the same scene — both are scored against the reference region they cover.

    Args:
        recon: reconstruction points.
        similarity: mapping ``recon`` into GT coordinates.
        margin: padding as a fraction of the box extent.
        percentile: use this percentile band instead of min/max. A
            reconstruction can contain a few stray points far from the scene
            (spurious MVS blobs), which would stretch a min/max box across the
            whole reference and defeat the crop.
    """
    mapped = similarity.apply(recon)
    if percentile > 0.0:
        minimum = np.percentile(mapped, percentile, axis=0)
        maximum = np.percentile(mapped, 100.0 - percentile, axis=0)
    else:
        minimum = mapped.min(axis=0)
        maximum = mapped.max(axis=0)
    box = CropBox(minimum=minimum, maximum=maximum)
    return box.padded(margin)
