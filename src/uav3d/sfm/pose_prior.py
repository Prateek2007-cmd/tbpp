"""GPS-constrained (pose-prior) bundle adjustment for a finished SfM run.

Takes a run directory that already has ``colmap/sparse/0`` (incremental SfM)
and ``telemetry.json`` (Mode B sync), and re-solves the bundle adjustment
with GPS position priors on the camera centres, via pycolmap 3.12.6
``create_pose_prior_bundle_adjuster``.

Verified API facts (Stage 0 spike, ``~/workspace/tbpp-audit/grapeslam/
stage0-poseprior-spike.md`` — read it before changing anything here):

* The adjuster does NOT convert WGS84. It uses ``prior.position`` raw as
  Cartesian coordinates. The caller MUST convert lat/lon/alt to local ENU
  metres itself (``pycolmap.GPSTransform.ellipsoid_to_enu``) and set
  ``coordinate_system=CARTESIAN``. Passing raw [lat, lon, alt] runs WITHOUT
  error and produces a metrically garbage model (raw ENU residual 68.7 m,
  scale 196.2 on KABR) that the repo's Sim(3) GPS metric cannot detect.
* ``position_covariance`` is mandatory: a prior without a valid (all-finite)
  covariance is silently skipped, and with no valid covariances at all the
  adjuster silently falls back to plain BA with a gauge fix. This module
  asserts every prior ``is_covariance_valid()`` and asserts the Ceres
  residual count equals 2 x observations + 3 x n_priors.
* ``update_point_3d_errors()`` MUST be called before reading post-BA
  reprojection, or the AFTER number is just the stale BEFORE value.
* The focal MUST stay locked: freeing it collapsed KABR's focal 2443.6 ->
  1564.9 px and doubled GPS RMSE. This module hard-codes locked intrinsics
  (``refine_focal_length=False`` + ``set_constant_cam_intrinsics``) and
  asserts the focal is unchanged after the solve. There is no option to
  free it.
* After ``solve()`` the reconstruction is in the priors' metric frame
  (local ENU here): the constructor aligns + normalises, ``solve()`` undoes
  the normalisation but NOT the metric alignment. The scoring Sim(3) scale
  therefore comes out ~1.00 after a correct prior-BA — the metric scale
  comes from the pipeline, as the plan's Stage 5 requires.

The wrapper owns the downstream refreshes, so the run stays consistent:
the pre-BA model is backed up to ``colmap/sparse/0-pre-priorba``, the
adjusted model is written back to ``colmap/sparse/0`` (NOT a new subdir —
``dense/geometry.py::load_sparse_points`` takes the FIRST sorted subdir, so
dense would silently load the pre-BA model otherwise), ``poses.json`` is
re-exported from the adjusted reconstruction, the stale
``sparse_points.npz`` cache is deleted, and ``poses_enriched.json`` is
rebuilt from the new poses + the existing telemetry join (the frame<->fix
join does not change — timestamps did not move — so no re-matching is
needed or wanted).

Report: ``pose_prior_ba.json`` in the run dir, with before/after
reprojection (fresh), Sim(3) GPS median/RMS, RAW ENU median/RMS (no
re-alignment — the gate that catches a wrong output frame), scale factor,
held-out residuals, per-image residuals, sigma, holdout list, wall time
and Ceres termination type. All GPS numbers are scored in COLMAP's
ellipsoidal ENU frame (origin = first fix); the repo's spherical-ENU
numbers are reported alongside as a cross-check (they differ by up to
~0.3 m on KABR; ~0.25% of extent at GrapeSLAM's latitude).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from uav3d.exceptions import Uav3dError
from uav3d.models.frames import CameraCalibration
from uav3d.models.poses import PosesFile
from uav3d.models.telemetry import TelemetryFile
from uav3d.sfm.colmap_export import export_poses
from uav3d.validation.gps_residual import _latlon_to_enu, gps_trajectory_residual

logger = logging.getLogger("uav3d.sfm.pose_prior")

#: Default isotropic prior sigma (m). Consumer-GPS honesty, not a tuning
#: knob: the Stage 0 spike showed sigma barely matters on KABR (residual
#: 4.01-4.02 m for sigma 3 / 1 / 0.3), so pick the honest GPS accuracy.
DEFAULT_SIGMA_M = 3.0
#: Default fraction of fixes held out of the prior set (spread along the
#: strip, every 10th image) and scored afterwards (honest, non-circular).
DEFAULT_HOLDOUT_FRACTION = 0.10
#: Ceres iteration cap. Every Stage 0 spike run hit the 100-iteration
#: default ("No convergence"); 250 gives the 120-frame GrapeSLAM solve
#: room without an unbounded run.
DEFAULT_MAX_NUM_ITERATIONS = 250
#: Explicit RANSAC max error (m) for the initial Sim(3) alignment in the
#: adjuster constructor. The auto value (3 x mean max-stddev = 9 m at
#: sigma 3) is fine, but an explicit value documents the outlier policy.
RANSAC_MAX_ERROR_M = 10.0

SPARSE_DIRNAME = "0"
BACKUP_DIRNAME = "0-pre-priorba"
REPORT_FILENAME = "pose_prior_ba.json"


def select_holdout_indices(n_images: int, holdout_fraction: float) -> list[int]:
    """Pick held-out image indices, spread along the strip.

    Every ``stride``-th image (counting from 1), so the holdout is spread
    over the whole flight instead of clustered at one end. For the default
    10% holdout that is every 10th image (indices 9, 19, 29, ...).
    """
    if not 0.0 <= holdout_fraction < 0.5:
        raise Uav3dError(
            f"holdout_fraction must be in [0, 0.5), got {holdout_fraction}"
        )
    if holdout_fraction == 0.0 or n_images == 0:
        return []
    stride = max(1, int(round(1.0 / holdout_fraction)))
    return [i for i in range(n_images) if (i + 1) % stride == 0]


def latlon_to_enu_colmap(
    lla: np.ndarray, origin_lat: float, origin_lon: float
) -> np.ndarray:
    """WGS84 lat/lon/alt (N, 3) -> local ENU metres (N, 3), COLMAP frame.

    Uses pycolmap's ellipsoidal ``GPSTransform`` — the same conversion the
    ``pose_prior_mapper`` CLI path applies. The origin is the first fix,
    matching the repo's ENU-origin convention.
    """
    gps_tf = pycolmap.GPSTransform(pycolmap.GPSTransfromEllipsoid.WGS84)
    return np.asarray(
        gps_tf.ellipsoid_to_enu(
            np.asarray(lla, dtype=np.float64), float(origin_lat), float(origin_lon)
        ),
        dtype=np.float64,
    )


def build_pose_priors(
    enu_positions: np.ndarray,
    image_ids: list[int],
    sigma_m: float,
    holdout_indices: set[int] | None = None,
) -> dict[int, pycolmap.PosePrior]:
    """Build Cartesian ENU position priors for the given images.

    Args:
        enu_positions: (N, 3) ENU metres, same order as ``image_ids``.
        image_ids: COLMAP image ids, same order as ``enu_positions``.
        sigma_m: isotropic prior sigma in metres (covariance = sigma^2 * I).
        holdout_indices: positions (into ``image_ids``) held out of the
            prior set; they get no prior and are scored afterwards.

    Raises:
        Uav3dError: if sigma is not positive, or any prior fails the
            validity guards (a prior without a valid covariance is
            silently skipped by the adjuster — never let one through).
    """
    if sigma_m <= 0:
        raise Uav3dError(f"sigma_m must be positive, got {sigma_m}")
    positions = np.asarray(enu_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) != len(image_ids):
        raise Uav3dError(
            f"Shape mismatch: enu_positions={positions.shape}, "
            f"image_ids={len(image_ids)}. Expected (N, 3) and N ids."
        )
    holdout = holdout_indices or set()
    cov = np.eye(3) * float(sigma_m) ** 2
    priors: dict[int, pycolmap.PosePrior] = {}
    for k, image_id in enumerate(image_ids):
        if k in holdout:
            continue
        prior = pycolmap.PosePrior()
        prior.position = positions[k]
        prior.position_covariance = cov
        prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        # Both guards are load-bearing: the adjuster silently skips priors
        # that fail either check (LOG(ERROR), no exception), and with zero
        # valid priors it silently falls back to plain BA + gauge fix.
        if not prior.is_valid():
            raise Uav3dError(
                f"PosePrior for image_id {image_id} is invalid "
                f"(position {positions[k]} not all-finite)"
            )
        if not prior.is_covariance_valid():
            raise Uav3dError(
                f"PosePrior for image_id {image_id} has an invalid covariance "
                "(must be all-finite)"
            )
        priors[image_id] = prior
    if len(priors) < 3:
        raise Uav3dError(
            f"Need >= 3 priors for the initial Sim(3) alignment, got {len(priors)} "
            f"({len(image_ids)} images, {len(holdout)} held out)"
        )
    return priors


def load_run_telemetry(run_dir: Path) -> dict[str, tuple[float, float, float]]:
    """frame_id -> (lat, lon, alt) from the run's telemetry.json.

    Only records with a frame_id and a lat/lon fix are returned (unmatched
    frames have lat=None and cannot carry a prior).
    """
    telemetry_path = Path(run_dir) / "telemetry.json"
    if not telemetry_path.exists():
        raise Uav3dError(
            f"telemetry.json not found in {run_dir}; run 'uav3d sync' first "
            "(pose-prior BA needs synced GPS fixes)."
        )
    tel_file = TelemetryFile.model_validate_json(
        telemetry_path.read_text(encoding="utf-8")
    )
    out: dict[str, tuple[float, float, float]] = {}
    for rec in tel_file.records:
        if rec.frame_id and rec.lat is not None and rec.lon is not None:
            out[rec.frame_id] = (
                float(rec.lat),
                float(rec.lon),
                float(rec.alt or 0.0),
            )
    return out


def _residual_stats(errors_m: np.ndarray) -> dict[str, float]:
    """Median/RMSE/p95/max of a per-image residual array (metres)."""
    errs = np.asarray(errors_m, dtype=np.float64)
    return {
        "median_m": round(float(np.median(errs)), 4),
        "rmse_m": round(float(np.sqrt(np.mean(errs**2))), 4),
        "p95_m": round(float(np.percentile(errs, 95)), 4),
        "max_m": round(float(np.max(errs)), 4),
    }


def _sim3_stats(
    centres: np.ndarray, gps_enu: np.ndarray, min_frames: int = 10
) -> dict[str, Any]:
    """Sim(3) GPS residual stats via the repo's own metric."""
    metrics = gps_trajectory_residual(centres, gps_enu, min_frames=min_frames)
    return {
        "median_m": metrics.median_m,
        "rmse_m": metrics.rmse_m,
        "p95_m": metrics.p95_m,
        "max_m": metrics.max_m,
        "scale_factor": metrics.scale_factor,
        "num_matched_frames": metrics.num_matched_frames,
        "fit_condition": metrics.fit_condition,
    }


def _aligned_residuals(centres: np.ndarray, gps_enu: np.ndarray) -> np.ndarray:
    """Per-image residuals (m) after the best Sim(3) alignment."""
    from uav3d.validation.ate import umeyama_alignment

    _scale, _rot, _trans, aligned = umeyama_alignment(centres, gps_enu)
    return np.linalg.norm(gps_enu - aligned, axis=1)


@dataclass
class PosePriorBaReport:
    """Everything ``pose_prior_ba.json`` records for one prior-BA run."""

    run_dir: str
    sigma_m: float
    holdout_fraction: float
    holdout_frame_ids: list[str]
    n_images: int
    n_registered: int
    n_skipped: int
    skipped_frame_ids: list[str]
    n_priors: int
    n_holdout: int
    max_num_iterations: int
    ransac_max_error_m: float
    robust_loss_on_prior_position: bool
    enu_frame: str = "COLMAP ellipsoidal ENU (GPSTransform, origin = first fix)"
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    wall_time_s: dict[str, float] = field(default_factory=dict)
    focal_unchanged: bool = False
    backup_dir: str = ""
    refreshed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "sigma_m": self.sigma_m,
            "holdout_fraction": self.holdout_fraction,
            "holdout_frame_ids": self.holdout_frame_ids,
            "n_images": self.n_images,
            "n_registered": self.n_registered,
            "n_skipped": self.n_skipped,
            "skipped_frame_ids": self.skipped_frame_ids,
            "n_priors": self.n_priors,
            "n_holdout": self.n_holdout,
            "max_num_iterations": self.max_num_iterations,
            "ransac_max_error_m": self.ransac_max_error_m,
            "robust_loss_on_prior_position": self.robust_loss_on_prior_position,
            "enu_frame": self.enu_frame,
            "before": self.before,
            "after": self.after,
            "solver": self.solver,
            "wall_time_s": self.wall_time_s,
            "focal_unchanged": self.focal_unchanged,
            "backup_dir": self.backup_dir,
            "refreshed": self.refreshed,
        }


def _refresh_poses_enriched(
    run_dir: Path, poses_path: Path, telemetry: dict[str, tuple[float, float, float]]
) -> dict[str, Any]:
    """Rebuild poses_enriched.json from the (new) poses + the telemetry join.

    ``poses_enriched.json`` differs from ``poses.json`` only in the
    ``telemetry_recorded`` flag per pose (see
    ``TelemetrySynchronizer._sync_mode_b_video``). The frame<->fix join
    does not change under prior-BA — timestamps did not move — so the
    faithful refresh is a direct join, not a re-match (re-running the full
    synchronizer would re-estimate the clock offset, and the wrapper does
    not know the run's sync config). Returns a status dict for the report.
    """
    enriched_path = run_dir / "poses_enriched.json"
    if not enriched_path.exists():
        return {"poses_enriched_json": "absent (no pre-BA file to refresh)"}
    poses_file = PosesFile.model_validate_json(poses_path.read_text(encoding="utf-8"))
    for pose in poses_file.poses:
        pose.telemetry_recorded = pose.frame_id in telemetry
    enriched_path.write_text(
        poses_file.model_dump_json(indent=2), encoding="utf-8"
    )
    n_flagged = sum(1 for p in poses_file.poses if p.telemetry_recorded)
    return {
        "poses_enriched_json": "rebuilt from post-BA poses + telemetry join",
        "poses_enriched_flagged": f"{n_flagged}/{len(poses_file.poses)}",
    }


def apply_pose_prior_ba(
    run_dir: Path | str,
    sigma_m: float = DEFAULT_SIGMA_M,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    max_num_iterations: int = DEFAULT_MAX_NUM_ITERATIONS,
) -> PosePriorBaReport:
    """Run GPS-constrained bundle adjustment on a finished SfM run.

    Args:
        run_dir: Run directory with ``colmap/sparse/0`` and
            ``telemetry.json`` (run ``uav3d sfm`` then ``uav3d sync`` first).
        sigma_m: isotropic GPS prior sigma in metres (default 3.0).
        holdout_fraction: fraction of fixes held out of the prior set,
            spread along the strip (default 0.10 = every 10th image).
        max_num_iterations: Ceres iteration cap (default 250; the 100
            default was hit by every Stage 0 spike run).

    Returns:
        PosePriorBaReport; also written to ``<run_dir>/pose_prior_ba.json``.

    Side effects (all owned by this wrapper, so the run stays consistent):
    ``colmap/sparse/0`` is backed up to ``colmap/sparse/0-pre-priorba``
    and then overwritten with the adjusted model; ``poses.json`` is
    re-exported from the adjusted model; the stale ``sparse_points.npz``
    cache is deleted; ``poses_enriched.json`` is rebuilt from the new
    poses. Intrinsics are hard-coded locked — there is no option to free
    them (see the module docstring).
    """
    run_path = Path(run_dir).resolve()
    sparse_root = run_path / "colmap" / "sparse"
    sparse_dir = sparse_root / SPARSE_DIRNAME
    backup_dir = sparse_root / BACKUP_DIRNAME
    if not sparse_dir.exists():
        raise Uav3dError(
            f"Sparse model not found: {sparse_dir}; run 'uav3d sfm' first."
        )
    if backup_dir.exists():
        raise Uav3dError(
            f"Prior-BA backup already exists: {backup_dir}. This run has "
            "already been through pose-prior BA (or the backup was left "
            "behind). To re-run: restore colmap/sparse/0 from the backup "
            "first (rm -rf colmap/sparse/0 && cp -r "
            "colmap/sparse/0-pre-priorba colmap/sparse/0), then re-run. "
            "Refusing to overwrite the backup with an already-adjusted model."
        )
    keyframes_path = run_path / "keyframes.json"
    if not keyframes_path.exists():
        raise Uav3dError(f"keyframes.json not found in {run_path}")

    t_total0 = time.time()
    telemetry = load_run_telemetry(run_path)

    rec = pycolmap.Reconstruction(str(sparse_dir))
    reg_ids_all = sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name)
    names_all = [rec.images[i].name for i in reg_ids_all]
    frame_ids_all = [n.rsplit(".", 1)[0] for n in names_all]
    # Frames without a GPS fix (e.g. the first hover frames when the sync
    # offset pushes the first fix past them) carry no prior and are not
    # scored — they are skipped, not failed on. A low overall match rate
    # still fails loudly via the minimum-frame guard below.
    paired = [
        (i, f) for i, f in zip(reg_ids_all, frame_ids_all) if f in telemetry
    ]
    skipped_frame_ids = [f for f in frame_ids_all if f not in telemetry]
    if skipped_frame_ids:
        logger.warning(
            "%d registered frames lack a GPS fix in telemetry.json and "
            "are skipped (no prior, not scored): %s",
            len(skipped_frame_ids),
            skipped_frame_ids[:5],
        )
    if len(paired) < 10:
        raise Uav3dError(
            f"Only {len(paired)} registered frames have a GPS fix "
            f"(need >= 10 for the Sim(3) GPS scoring); check the sync "
            f"match rate. Skipped: {skipped_frame_ids[:5]}"
        )
    reg_ids = [i for i, _f in paired]
    frame_ids = [f for _i, f in paired]
    names = [rec.images[i].name for i in reg_ids]
    lla = np.array([telemetry[f] for f in frame_ids], dtype=np.float64)
    logger.info(
        "Pose-prior BA: %d registered images, all with telemetry; "
        "lat %.6f..%.6f, lon %.6f..%.6f, alt %.1f..%.1f m",
        len(reg_ids),
        lla[:, 0].min(),
        lla[:, 0].max(),
        lla[:, 1].min(),
        lla[:, 1].max(),
        lla[:, 2].min(),
        lla[:, 2].max(),
    )

    # GPS in the COLMAP ENU frame (primary scoring frame) and the repo's
    # spherical ENU frame (cross-check; differs by up to ~0.3 m on KABR).
    gps_enu_colmap = latlon_to_enu_colmap(lla, float(lla[0, 0]), float(lla[0, 1]))
    origin = (float(lla[0, 0]), float(lla[0, 1]), float(lla[0, 2]))
    gps_enu_repo = np.array(
        [_latlon_to_enu(a, b, c, origin) for a, b, c in lla], dtype=np.float64
    )
    logger.info(
        "COLMAP ENU vs repo ENU max diff: %.4f m",
        float(np.abs(gps_enu_colmap - gps_enu_repo).max()),
    )
    # Degeneracy early-warning: the adjuster's initial RANSAC Sim(3)
    # alignment samples 3-point sets, and COLMAP's Sim(3) estimator
    # returns NO model for a collinear sample (FullPivLU rank < 3). A
    # perfectly straight flight therefore fails the alignment and the
    # adjuster silently falls back to plain BA with a gauge fix (verified
    # on a synthetic collinear strip: residual count 2x obs, no priors).
    # Real flights always have metres of lateral/vertical extent; warn
    # early if this one does not, so the failure is not a surprise. The
    # residual-count assert after solve() remains the hard guard.
    centred_gps = gps_enu_colmap - gps_enu_colmap.mean(axis=0)
    gps_singular = np.linalg.svd(centred_gps, compute_uv=False)
    logger.info(
        "GPS track singular values (m): %.2f / %.2f / %.2f",
        gps_singular[0],
        gps_singular[1],
        gps_singular[2],
    )
    if len(gps_singular) > 1 and gps_singular[1] < 1.0:
        logger.warning(
            "GPS track is near-collinear (2nd singular value %.3f m): "
            "the initial Sim(3) alignment may fail and the adjuster "
            "would silently fall back to plain BA. The residual-count "
            "assert after solve() will catch that if it happens.",
            float(gps_singular[1]),
        )

    # ---- BEFORE metrics (free-SfM geometry, before any prior) -----------
    centres_before = np.array(
        [rec.images[i].projection_center() for i in reg_ids], dtype=np.float64
    )
    n_points_before = rec.num_points3D()
    n_obs_before = rec.compute_num_observations()
    reproj_before = float(rec.compute_mean_reprojection_error())
    focal_before = {
        str(cid): [float(p) for p in cam.params]
        for cid, cam in rec.cameras.items()
    }
    sim3_before_colmap = _sim3_stats(centres_before, gps_enu_colmap)
    sim3_before_repo = _sim3_stats(centres_before, gps_enu_repo)
    logger.info(
        "BEFORE: images=%d points=%d obs=%d reproj=%.4f px "
        "gps_sim3_median=%.4f m scale=%.4f",
        len(reg_ids),
        n_points_before,
        n_obs_before,
        reproj_before,
        sim3_before_colmap["median_m"],
        sim3_before_colmap["scale_factor"],
    )

    # ---- Priors ---------------------------------------------------------
    holdout_idx = select_holdout_indices(len(reg_ids), holdout_fraction)
    holdout_set = set(holdout_idx)
    holdout_frame_ids = [frame_ids[i] for i in holdout_idx]
    priors = build_pose_priors(
        gps_enu_colmap, reg_ids, sigma_m, holdout_indices=holdout_set
    )
    logger.info(
        "Built %d priors (sigma=%.2f m, Cartesian ENU, held out %d: %s)",
        len(priors),
        sigma_m,
        len(holdout_idx),
        holdout_frame_ids[:5],
    )

    # ---- Back up the pre-BA model (before the adjuster mutates anything) -
    shutil.copytree(sparse_dir, backup_dir)
    logger.info("Backed up pre-BA model: %s -> %s", sparse_dir, backup_dir)

    # ---- Bundle adjustment: everything variable, intrinsics LOCKED ------
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in reg_ids:
        config.add_image(image_id)
    for pid in rec.points3D:
        config.add_variable_point(pid)
    for cam_id in rec.cameras:
        config.set_constant_cam_intrinsics(cam_id)

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.solver_options.max_num_iterations = int(max_num_iterations)
    prior_options = pycolmap.PosePriorBundleAdjustmentOptions()
    # Cauchy loss on prior residuals: one bad fix must not drag the
    # solution (KABR's last frame sits 9.8 m off its GPS).
    prior_options.use_robust_loss_on_prior_position = True
    prior_options.ransac_max_error = float(RANSAC_MAX_ERROR_M)

    t0 = time.time()
    adjuster = pycolmap.create_pose_prior_bundle_adjuster(
        options, prior_options, config, priors, rec
    )
    t_construct = time.time() - t0
    logger.info("Adjuster constructed in %.1f s", t_construct)

    t0 = time.time()
    summary = adjuster.solve()
    t_solve = time.time() - t0
    logger.info("solve() returned in %.1f s", t_solve)

    # Guard: the priors must actually be in the Ceres problem. A missing
    # covariance would have been caught by build_pose_priors, but assert
    # the residual count anyway — it is the only direct evidence the
    # priors participated (2 reprojection residuals per observation +
    # 3 prior residuals per prior).
    expected_residuals = 2 * n_obs_before + 3 * len(priors)
    n_residuals = int(summary.num_residuals)
    if n_residuals != expected_residuals:
        raise Uav3dError(
            f"Ceres residual count mismatch: got {n_residuals}, expected "
            f"{expected_residuals} (2 x {n_obs_before} obs + 3 x "
            f"{len(priors)} priors). The priors are not (all) in the "
            "problem — most likely the initial Sim(3) alignment failed "
            "(check the log for 'Alignment w.r.t. prior positions "
            "failed'; a near-collinear trajectory is the usual cause) "
            "and the adjuster silently fell back to plain BA with a "
            "gauge fix. Do not trust this solution."
        )
    termination = str(summary.termination_type).replace(
        "TerminationType.", ""
    )

    # REQUIRED before reading post-BA reprojection: BA does not refresh the
    # stored per-point errors (verified on KABR: stale 1.1373 px vs fresh
    # 1.1265 px locked-focal, 0.9165 px refine-focal).
    rec.update_point_3d_errors()

    # Guard: the focal must not have moved. The locked-focal pipeline has
    # no business changing intrinsics; if it did, the model is suspect.
    focal_after = {
        str(cid): [float(p) for p in cam.params]
        for cid, cam in rec.cameras.items()
    }
    focal_unchanged = all(
        np.allclose(
            np.array(focal_before[cid], dtype=np.float64),
            np.array(focal_after[cid], dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )
        for cid in focal_before
    )
    if not focal_unchanged:
        raise Uav3dError(
            f"Focal changed during pose-prior BA despite locked intrinsics: "
            f"before={focal_before} after={focal_after}. Refusing to write "
            "the adjusted model."
        )

    # ---- Write the adjusted model back to sparse/0 ----------------------
    rec.write(str(sparse_dir))
    logger.info("Adjusted model written back to %s", sparse_dir)

    # Re-read from disk for the report: the post-BA reprojection in the
    # report must come from the re-written model on disk, not from memory.
    rec_after = pycolmap.Reconstruction(str(sparse_dir))
    rec_after.update_point_3d_errors()
    centres_after = np.array(
        [rec_after.images[i].projection_center() for i in reg_ids],
        dtype=np.float64,
    )
    n_points_after = rec_after.num_points3D()
    n_obs_after = rec_after.compute_num_observations()
    reproj_after = float(rec_after.compute_mean_reprojection_error())

    # ---- AFTER metrics --------------------------------------------------
    sim3_after_colmap = _sim3_stats(centres_after, gps_enu_colmap)
    sim3_after_repo = _sim3_stats(centres_after, gps_enu_repo)
    # Raw ENU residual: NO re-alignment — the true georeferencing error of
    # the prior-BA model, and the gate that catches a wrong output frame
    # (the Sim(3) metric cannot: the WGS84-raw negative control scored
    # 4.02 m Sim(3) while its raw ENU residual was 68.7 m).
    raw_colmap = np.linalg.norm(centres_after - gps_enu_colmap, axis=1)
    raw_repo = np.linalg.norm(centres_after - gps_enu_repo, axis=1)
    per_image_sim3 = _aligned_residuals(centres_after, gps_enu_colmap)

    held: dict[str, Any] | None = None
    if holdout_idx:
        h = np.array(holdout_idx, dtype=int)
        # Primary held-out number: residuals of the held-out images under
        # the GLOBAL Sim(3) fit (one alignment for all images). A separate
        # Sim(3) fit on the holdout alone (spike-style) is reported for
        # comparability — it can overfit 7 parameters to ~12 points, so it
        # is not the gate number.
        held_sim3_global = _residual_stats(per_image_sim3[h])
        held_raw = _residual_stats(raw_colmap[h])
        held_own = None
        if len(h) >= 3:
            try:
                held_own = _sim3_stats(
                    centres_after[h], gps_enu_colmap[h], min_frames=3
                )
            except Uav3dError:
                held_own = None
        held = {
            "n": len(h),
            "sim3_global_median_m": held_sim3_global["median_m"],
            "sim3_global_rmse_m": held_sim3_global["rmse_m"],
            "sim3_global_per_image_m": [
                round(float(x), 3) for x in per_image_sim3[h]
            ],
            "sim3_own_fit": held_own,
            "raw_median_m": held_raw["median_m"],
            "raw_rmse_m": held_raw["rmse_m"],
            "raw_p95_m": held_raw["p95_m"],
            "raw_max_m": held_raw["max_m"],
            "raw_per_image_m": [round(float(x), 3) for x in raw_colmap[h]],
        }

    logger.info(
        "AFTER: reproj=%.4f px gps_sim3_median=%.4f m "
        "raw_enu_median=%.4f m scale=%.4f",
        reproj_after,
        sim3_after_colmap["median_m"],
        float(np.median(raw_colmap)),
        sim3_after_colmap["scale_factor"],
    )

    # ---- Downstream refreshes (the run must stay consistent) ------------
    refreshed: dict[str, Any] = {}

    # 1. Re-export poses.json from the adjusted reconstruction (dense reads
    # poses.json for cameras). Camera block comes from the adjusted model.
    keyframes_data = json.loads(keyframes_path.read_text(encoding="utf-8"))
    all_image_names = [
        f"{kf['frame_id']}.jpg" for kf in keyframes_data["keyframes"]
    ]
    timestamps_map = {
        kf["frame_id"]: kf.get("timestamp_us") for kf in keyframes_data["keyframes"]
    }
    dataset_label = keyframes_data.get("dataset_label", "replayed public dataset")
    poses_path = run_path / "poses.json"
    if poses_path.exists():
        try:
            old_poses = PosesFile.model_validate_json(
                poses_path.read_text(encoding="utf-8")
            )
            dataset_label = old_poses.dataset_label
        except (ValueError, OSError):
            pass
    first_cam = next(iter(rec_after.cameras.values()))
    calibration = CameraCalibration(
        model=str(first_cam.model).replace("CameraModelId.", ""),
        width=int(first_cam.width),
        height=int(first_cam.height),
        params=[float(p) for p in first_cam.params],
        source="pose_prior_ba (adjusted reconstruction)",
        provenance="sensor_spec",
    )
    export_poses(
        reconstruction=rec_after,
        all_image_names=all_image_names,
        camera_calibration=calibration,
        output_path=poses_path,
        dataset_label=dataset_label,
        scale_factor=1.0,
        timestamps_map=timestamps_map,
    )
    refreshed["poses_json"] = "re-exported from adjusted model"

    # 2. The sparse_points.npz cache is stale now (dense prefers the model,
    # but the cache is read as a fallback — never leave a stale one).
    npz_path = run_path / "sparse_points.npz"
    if npz_path.exists():
        npz_path.unlink()
        refreshed["sparse_points_npz"] = "deleted (stale after prior-BA)"
    else:
        refreshed["sparse_points_npz"] = "absent (nothing to delete)"

    # 3. poses_enriched.json: rebuild from the new poses + telemetry join.
    try:
        refreshed.update(_refresh_poses_enriched(run_path, poses_path, telemetry))
    except Exception as exc:  # noqa: BLE001 - enrichment is advisory
        logger.warning("poses_enriched.json refresh failed: %s", exc)
        refreshed["poses_enriched_json"] = (
            f"PRE-BA (refresh failed: {exc}); poses_enriched.json still "
            "holds pre-BA camera centres — do not use it for scoring."
        )

    wall_total = time.time() - t_total0
    report = PosePriorBaReport(
        run_dir=str(run_path),
        sigma_m=float(sigma_m),
        holdout_fraction=float(holdout_fraction),
        holdout_frame_ids=holdout_frame_ids,
        n_images=len(reg_ids),
        n_registered=len(reg_ids_all),
        n_skipped=len(skipped_frame_ids),
        skipped_frame_ids=skipped_frame_ids,
        n_priors=len(priors),
        n_holdout=len(holdout_idx),
        max_num_iterations=int(max_num_iterations),
        ransac_max_error_m=float(RANSAC_MAX_ERROR_M),
        robust_loss_on_prior_position=True,
        before={
            "n_points": n_points_before,
            "n_obs": n_obs_before,
            "reproj_px": round(reproj_before, 4),
            "focal_params": focal_before,
            "gps_sim3_colmap_enu_m": sim3_before_colmap,
            "gps_sim3_repo_enu_m": sim3_before_repo,
        },
        after={
            "n_points": n_points_after,
            "n_obs": n_obs_after,
            "reproj_px": round(reproj_after, 4),
            "focal_params": focal_after,
            "gps_sim3_colmap_enu_m": sim3_after_colmap,
            "gps_sim3_repo_enu_m": sim3_after_repo,
            "raw_enu_colmap_m": _residual_stats(raw_colmap),
            "raw_enu_repo_m": _residual_stats(raw_repo),
            "held_out": held,
            "per_image_sim3_m": {
                frame_ids[i]: round(float(per_image_sim3[i]), 3)
                for i in range(len(frame_ids))
            },
            "per_image_raw_enu_m": {
                frame_ids[i]: round(float(raw_colmap[i]), 3)
                for i in range(len(frame_ids))
            },
        },
        solver={
            "termination_type": termination,
            "num_residuals": n_residuals,
            "expected_residuals": expected_residuals,
            "initial_cost": round(float(summary.initial_cost), 6),
            "final_cost": round(float(summary.final_cost), 6),
            "num_successful_steps": int(summary.num_successful_steps),
            "num_unsuccessful_steps": int(summary.num_unsuccessful_steps),
            "solver_total_time_s": round(float(summary.total_time_in_seconds), 1),
        },
        wall_time_s={
            "construct_s": round(t_construct, 1),
            "solve_s": round(t_solve, 1),
            "total_s": round(wall_total, 1),
        },
        focal_unchanged=focal_unchanged,
        backup_dir=str(backup_dir),
        refreshed=refreshed,
    )
    report_path = run_path / REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    logger.info("Pose-prior BA report written to %s", report_path)
    return report
