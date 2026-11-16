"""Monocular depth prior aligned to the SfM frame.

Plane-sweep MVS is photometric: flat, untextured walls and roofs have no cost
minimum and are rejected. A monocular model still predicts their shape. This
module fits an affine inverse-depth transform to the SfM points visible in each
view, then fills only those MVS holes whose prior depth is confirmed by the
prior maps of neighbouring views. Measured MVS depth always wins.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.stats import spearmanr

from uav3d.dense.cameras import CameraView
from uav3d.dense.sweep import DepthEstimate


def fit_inverse_depth(
    dispersion: np.ndarray,
    depth: np.ndarray,
    min_samples: int = 24,
    min_rank_corr: float = 0.35,
    max_relative_error: float = 0.35,
):
    """Fit ``1/z = scale * x + shift`` robustly, returning quality or None.

    ``x`` is the raw map, inverted first when the map is depth-ordered (larger
    values mean farther). Two MAD outlier-rejection passes keep a bad sparse
    track or two from tilting the whole view.
    """
    raw = np.asarray(dispersion, dtype=np.float64)
    z = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(raw) & np.isfinite(z) & (raw > 1e-6) & (z > 1e-6)
    raw, z = raw[ok], z[ok]
    if raw.size < max(4, int(min_samples)):
        return None

    med = float(np.median(z))
    keep = (z >= 0.1 * med) & (z <= 10.0 * med)
    if int(keep.sum()) >= min_samples:
        raw, z = raw[keep], z[keep]

    r_depth = float(spearmanr(raw, z).statistic)
    r_inv = float(spearmanr(raw, 1.0 / z).statistic)
    inverted = r_depth > r_inv
    x = 1.0 / np.maximum(raw, 1e-8) if inverted else raw
    inv_z = 1.0 / z

    inliers = np.ones(x.size, dtype=bool)
    scale = shift = 0.0
    for _ in range(2):
        if int(inliers.sum()) < 4:
            return None
        design = np.stack([x[inliers], np.ones(int(inliers.sum()))], axis=1)
        scale, shift = np.linalg.lstsq(design, inv_z[inliers], rcond=None)[0]
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        pred = 1.0 / np.maximum(scale * x + shift, 1e-8)
        rel = np.abs(pred - z) / np.maximum(z, 1e-8)
        med_err = float(np.median(rel[inliers]))
        mad = float(np.median(np.abs(rel[inliers] - med_err)))
        limit = med_err + 3.0 * 1.4826 * max(mad, 1e-6)
        new = rel <= max(limit, 0.05)
        if int(new.sum()) < min_samples or np.array_equal(new, inliers):
            break
        inliers = new

    corr = float(spearmanr(x[inliers], inv_z[inliers]).statistic)
    err = float(
        np.median(
            np.abs(1.0 / np.maximum(scale * x[inliers] + shift, 1e-8) - z[inliers])
            / z[inliers]
        )
    )
    if not np.isfinite(corr) or corr < float(min_rank_corr):
        return None
    if not np.isfinite(err) or err > float(max_relative_error):
        return None
    return scale, shift, corr, err, bool(inverted)


def depth_from_prior(
    raw: np.ndarray,
    scale: float,
    shift: float,
    inverted: bool,
) -> np.ndarray:
    """Apply the fitted affine map in inverse depth and return metres/units."""
    values = np.asarray(raw, dtype=np.float32)
    x = 1.0 / np.maximum(values, 1e-8) if inverted else values
    inv = scale * x + shift
    depth = np.zeros_like(inv, dtype=np.float32)
    ok = inv > 1e-6
    depth[ok] = 1.0 / inv[ok]
    return depth


class MonocularPriorModel:
    """CPU wrapper around the Depth Anything V2 ONNX model."""

    def __init__(self, model_path: str | Path, infer_height: int = 518, threads: int = 2) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Depth prior model not found: {self.model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.infer_height = int(infer_height)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return the raw monocular map at the input image resolution."""
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("monocular prior expects an HxWx3 image")
        image = image[:, :, :3]
        height, width = image.shape[:2]
        infer_h = max(14, int(round(self.infer_height / 14.0)) * 14)
        infer_w = max(14, int(round((width / height) * infer_h / 14.0)) * 14)
        resized = cv2.resize(image, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        raw = self.session.run(None, {self.input_name: tensor})[0]
        raw = np.asarray(raw, dtype=np.float32).squeeze()
        if raw.ndim != 2:
            raise ValueError(f"unexpected prior output shape {raw.shape}")
        return cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)


def _sample_nearest(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    cols = np.clip(np.floor(u).astype(np.int64), 0, width - 1)
    rows = np.clip(np.floor(v).astype(np.int64), 0, height - 1)
    return image[rows, cols]


def build_prior(
    model: MonocularPriorModel,
    view: CameraView,
    image_bgr: np.ndarray,
    sparse_points: np.ndarray,
    near: float,
    far: float,
    min_rank_corr: float = 0.35,
    max_relative_error: float = 0.35,
):
    """Predict and SfM-align one view's monocular depth."""
    raw = model.predict(image_bgr)
    uv, z = view.project(sparse_points)
    inside = (
        (z > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < view.intrinsics.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < view.intrinsics.height)
    )
    if int(inside.sum()) < 24:
        return np.zeros_like(raw), 0.0, 0.0, 0.0, float("inf"), False
    sampled = _sample_nearest(raw, uv[inside, 0], uv[inside, 1])
    good = sampled > 1e-6
    if int(good.sum()) < 24:
        return np.zeros_like(raw), 0.0, 0.0, 0.0, float("inf"), False
    fit = fit_inverse_depth(
        sampled[good],
        z[inside][good],
        min_rank_corr=min_rank_corr,
        max_relative_error=max_relative_error,
    )
    if fit is None:
        return np.zeros_like(raw), 0.0, 0.0, 0.0, float("inf"), False
    scale, shift, corr, err, inverted = fit
    depth = depth_from_prior(raw, scale, shift, inverted)
    valid = (depth >= 0.9 * near) & (depth <= 1.1 * far)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return depth, float(scale), float(shift), float(corr), float(err), bool(inverted)


def edge_mask(image_bgr: np.ndarray, radius: int = 1) -> np.ndarray:
    """Pixels close to strong image edges: depth-prior no-go zone."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    if radius <= 0:
        return edges > 0
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1) > 0


def fill_from_prior(
    estimate: DepthEstimate,
    prior: np.ndarray,
    ref_view: CameraView,
    neighbour_views: list[CameraView],
    neighbour_priors: list[np.ndarray],
    near: float,
    far: float,
    min_support: int = 2,
    tolerance: float = 0.06,
    avoid_edges: np.ndarray | None = None,
) -> tuple[DepthEstimate, int]:
    """Fill MVS-invalid pixels with prior depth confirmed by neighbouring priors."""
    candidates = (
        (~estimate.valid)
        & (prior > 0.0)
        & (prior >= 0.95 * near)
        & (prior <= 1.05 * far)
    )
    if avoid_edges is not None:
        candidates &= ~avoid_edges
    if not np.any(candidates) or not neighbour_views:
        return estimate, 0

    flat = np.flatnonzero(candidates.ravel())
    uv_all = ref_view.pixel_grid()
    uv = uv_all[flat]
    z = prior.ravel()[flat].astype(np.float64)
    points = ref_view.backproject(uv, z)
    votes = np.zeros(flat.size, np.float32)

    for view, neighbour_prior in zip(neighbour_views, neighbour_priors):
        uv_n, z_n = view.project(points)
        inside = (
            (z_n > 0)
            & (uv_n[:, 0] >= 0)
            & (uv_n[:, 0] < view.intrinsics.width)
            & (uv_n[:, 1] >= 0)
            & (uv_n[:, 1] < view.intrinsics.height)
        )
        if not np.any(inside):
            continue
        sample = _sample_nearest(neighbour_prior, uv_n[:, 0], uv_n[:, 1])
        agree = (
            inside
            & (sample > 0)
            & (np.abs(sample - z_n) / np.maximum(z_n, 1e-9) <= float(tolerance))
        )
        votes += agree.astype(np.float32)

    keep = votes >= float(min_support)
    filled = int(keep.sum())
    if filled == 0:
        return estimate, 0

    depth = estimate.depth.copy()
    valid = estimate.valid.copy()
    chosen = flat[keep]
    depth.ravel()[chosen] = prior.ravel()[chosen]
    valid.ravel()[chosen] = True
    return DepthEstimate(depth=depth, cost=estimate.cost, num_views=estimate.num_views, valid=valid), filled
