"""GPS-trajectory-residual check: the metric validation reproj error cannot do.

Reprojection error cannot catch the bas-relief trap: on forward flight a
wrong focal length is absorbed into depth, so the model can be metres wrong
while reprojection error stays at ~0.28 px (docs/HANDOFF_NEXT_SESSION.md
§12). The check that *can* catch it is metric: align the SfM camera centres
against independently measured GPS positions with a Sim(3) fit and look at
what is left over. A focal-induced depth stretch is anisotropic — no
similarity transform can absorb it — so it shows up as a large residual
even when every reprojection looks perfect.

This module takes telemetry as input (arrays of metric positions, or a
``PosesFile`` + ``TelemetryFile`` pair); it does NOT parse video GPS itself
— that is Phase 2's job. On the Zurich path it works now (telemetry.json
from ``uav3d sync``); for video it activates as soon as Phase 2 writes
telemetry.json for video runs.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from uav3d.exceptions import Uav3dError
from uav3d.models.poses import PosesFile
from uav3d.models.telemetry import TelemetryFile
from uav3d.models.validation import GpsResidualMetrics
from uav3d.validation.ate import check_flightline_geometry, umeyama_alignment

logger = logging.getLogger("uav3d.validation.gps_residual")

MIN_GPS_FRAMES = 10


def _latlon_to_enu(lat: float, lon: float, alt: float, origin: tuple[float, float, float]) -> np.ndarray:
    """Local east/north/up offsets in metres for a short-baseline site.

    Same short-baseline math as ``dense/pipeline.py::_latlon_to_enu``: valid
    for the few-hundred-metre baselines of a single drone pass, where earth
    curvature is negligible.
    """
    radius = 6378137.0
    lat0, lon0, alt0 = origin
    east = math.radians(lon - lon0) * radius * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * radius
    return np.array([east, north, alt - alt0], dtype=np.float64)


def gps_trajectory_residual(
    est_centres: np.ndarray,
    gps_positions_m: np.ndarray,
    frame_ids: list[str] | None = None,
    min_frames: int = MIN_GPS_FRAMES,
) -> GpsResidualMetrics:
    """Compare SfM camera centres against GPS positions, in metres.

    Args:
        est_centres: (N, 3) SfM camera centres in SfM units.
        gps_positions_m: (N, 3) GPS positions in a local metric frame
            (ENU metres), same order as ``est_centres``.
        frame_ids: optional frame ids, used only for error messages.
        min_frames: minimum matched frames; fewer raises Uav3dError.

    Returns:
        Residual statistics after the best Sim(3) alignment. A good run has
        a median residual at the GPS noise level (metre-level for plain GPS,
        centimetre-level for RTK); a focal-stretched run shows metres.
    """
    est = np.asarray(est_centres, dtype=np.float64)
    gps = np.asarray(gps_positions_m, dtype=np.float64)
    if est.shape != gps.shape or est.ndim != 2 or est.shape[1] != 3:
        raise Uav3dError(f"Shape mismatch: est={est.shape}, gps={gps.shape}. Expected (N, 3).")
    if len(est) < min_frames:
        raise Uav3dError(
            f"GPS-trajectory-residual check needs at least {min_frames} matched "
            f"frames, got {len(est)}."
        )

    fit_condition, _geom_stats = check_flightline_geometry(gps)
    if fit_condition == "weak_flightline_geometry":
        logger.warning(
            "GPS trajectory is near-collinear (straight flight); rotation about "
            "the flight axis is weakly constrained, so the residual is a "
            "lower bound on the true metric error."
        )

    scale, _rotation, _translation, aligned = umeyama_alignment(est, gps)
    errors_m = np.linalg.norm(gps - aligned, axis=1)
    return GpsResidualMetrics(
        rmse_m=round(float(np.sqrt(np.mean(errors_m**2))), 4),
        median_m=round(float(np.median(errors_m)), 4),
        p95_m=round(float(np.percentile(errors_m, 95)), 4),
        max_m=round(float(np.max(errors_m)), 4),
        scale_factor=round(float(scale), 4),
        num_matched_frames=len(est),
        fit_condition=fit_condition,
    )


def gps_residual_from_files(
    poses_file: PosesFile,
    telemetry_file: TelemetryFile,
    min_frames: int = MIN_GPS_FRAMES,
) -> GpsResidualMetrics | None:
    """GPS-trajectory-residual check from a run's poses.json + telemetry.json.

    Matches frames by frame_id, converts GPS lat/lon/alt to local ENU metres
    about the first fix, and runs the residual check. Returns ``None`` (with
    a logged warning) when too few frames have both a registered pose and a
    GPS fix — the check is advisory and must never fail a validation run
    for missing telemetry.
    """
    records = {
        rec.frame_id: rec
        for rec in telemetry_file.records
        if rec.frame_id is not None and rec.lat is not None and rec.lon is not None
    }
    est: list[list[float]] = []
    gps_latlon: list[tuple[float, float, float]] = []
    frame_ids: list[str] = []
    for pose in poses_file.poses:
        if not pose.registered or pose.camera_center_world is None:
            continue
        rec = records.get(pose.frame_id)
        if rec is None:
            continue
        est.append(list(pose.camera_center_world))
        gps_latlon.append((float(rec.lat), float(rec.lon), float(rec.alt or 0.0)))
        frame_ids.append(pose.frame_id)

    if len(est) < min_frames:
        logger.warning(
            "GPS-trajectory-residual check skipped: only %d frames have both a "
            "registered pose and a GPS fix (need >= %d).",
            len(est),
            min_frames,
        )
        return None

    origin = gps_latlon[0]
    gps_enu = np.asarray(
        [_latlon_to_enu(lat, lon, alt, origin) for lat, lon, alt in gps_latlon],
        dtype=np.float64,
    )
    return gps_trajectory_residual(np.asarray(est, dtype=np.float64), gps_enu, frame_ids, min_frames)


def load_gps_residual_for_run(run_dir: str | Path) -> GpsResidualMetrics | None:
    """Run the GPS check for a run directory, if it has poses + telemetry.

    Returns ``None`` when either file is missing or the check cannot run;
    the caller treats that as "no GPS check available", not an error.
    """
    run_path = Path(run_dir)
    poses_path = run_path / "poses.json"
    telemetry_path = run_path / "telemetry.json"
    if not poses_path.exists() or not telemetry_path.exists():
        return None
    try:
        poses = PosesFile.model_validate_json(poses_path.read_text(encoding="utf-8"))
        telemetry = TelemetryFile.model_validate_json(telemetry_path.read_text(encoding="utf-8"))
        return gps_residual_from_files(poses, telemetry)
    except (Uav3dError, ValueError, OSError) as exc:
        logger.warning("GPS-trajectory-residual check failed: %s", exc)
        return None
