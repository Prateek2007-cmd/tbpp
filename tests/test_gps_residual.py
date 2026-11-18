"""Tests for the GPS-trajectory-residual check (Phase 1).

Reprojection error cannot catch the bas-relief trap: a wrong focal length
is absorbed into depth, so the model can be metres wrong while reprojection
error stays at ~0.28 px (docs/HANDOFF_NEXT_SESSION.md §12). The GPS check
catches it because a focal-induced depth stretch is anisotropic — no Sim(3)
alignment can absorb it — so it survives as a large residual against
independently measured GPS positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav3d.exceptions import Uav3dError
from uav3d.models.poses import PoseRecord, PosesFile
from uav3d.models.telemetry import TelemetryFile, TelemetryRecord
from uav3d.validation.gps_residual import (
    gps_residual_from_files,
    gps_trajectory_residual,
)


def _straight_flight(n: int = 20, step: float = 1.36, height: float = 12.0) -> np.ndarray:
    """Straight-line GPS trajectory in local ENU metres (the hard case)."""
    x = np.arange(n, dtype=np.float64) * step
    return np.column_stack([x, np.zeros(n), np.full(n, height)])


def _sim3(points: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a random rotation + scale + translation (what SfM legitimately is)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    qw, qx, qy, qz = q
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    return scale * (points @ rot.T) + rng.normal(size=3) * 5.0


class TestGpsTrajectoryResidual:
    def test_clean_reconstruction_has_small_residual(self) -> None:
        # SfM centres are a Sim(3) transform of the GPS trajectory plus
        # 5 cm of noise: the alignment absorbs the transform, so the
        # residual is just the noise.
        rng = np.random.default_rng(42)
        gps = _straight_flight()
        est = _sim3(gps, scale=0.37, rng=rng) + rng.normal(scale=0.05, size=gps.shape)
        result = gps_trajectory_residual(est, gps)
        assert result.median_m < 0.2
        assert result.p95_m < 0.3
        assert result.num_matched_frames == 20
        assert result.scale_factor == pytest.approx(1.0 / 0.37, rel=0.05)
        assert result.fit_condition == "weak_flightline_geometry"

    def test_bas_relief_trap_is_caught(self) -> None:
        # THE case reprojection error misses: the SfM centres are the true
        # trajectory with a progressive along-track scale drift (spacing
        # grows 30% over the pass) — the signature of a focal-induced depth
        # stretch on forward flight. Reprojection error for such a
        # reconstruction stays ~0.28 px (the focal absorbs it), but no
        # Sim(3) alignment can undo a non-uniform stretch, so the GPS
        # residual must be large.
        rng = np.random.default_rng(42)
        # 60 frames x 1.36 m: an 81.6 m pass, like the synthetic benchmark.
        gps = _straight_flight(n=60)
        stretched = gps.copy()
        x_max = gps[-1, 0]
        stretched[:, 0] = gps[:, 0] * (1.0 + 0.30 * gps[:, 0] / x_max)
        est = _sim3(stretched, scale=0.37, rng=rng)
        result = gps_trajectory_residual(est, gps)
        assert result.median_m > 1.0
        assert result.p95_m > 1.0

    def test_small_stretch_is_visible_but_below_threshold(self) -> None:
        # A 5% progressive drift (the size of error a 5% focal error causes)
        # leaves a residual well above GPS noise but below 1 m on this short
        # pass — the check is sensitive, not binary.
        rng = np.random.default_rng(42)
        gps = _straight_flight()
        stretched = gps.copy()
        x_max = gps[-1, 0]
        stretched[:, 0] = gps[:, 0] * (1.0 + 0.05 * gps[:, 0] / x_max)
        est = _sim3(stretched, scale=0.37, rng=rng)
        result = gps_trajectory_residual(est, gps)
        assert 0.05 < result.median_m < 1.0

    def test_too_few_frames_raises(self) -> None:
        gps = _straight_flight(n=5)
        with pytest.raises(Uav3dError, match="at least 10 matched"):
            gps_trajectory_residual(gps, gps)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(Uav3dError, match="Shape mismatch"):
            gps_trajectory_residual(np.zeros((12, 3)), np.zeros((11, 3)))


class TestGpsResidualFromFiles:
    def _poses_and_telemetry(self, n: int = 12):
        """Poses + telemetry for a straight flight; GPS in lat/lon/alt."""
        poses = []
        records = []
        for i in range(n):
            poses.append(
                PoseRecord(
                    frame_id=f"zurich_{i:06d}",
                    qw=1.0,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                    tx=float(i),
                    ty=0.0,
                    tz=0.0,
                    camera_center_world=[i * 1.36, 0.0, 12.0],
                    registered=True,
                )
            )
            # 1.36 m north per frame: 1 degree of latitude is ~111320 m.
            records.append(
                TelemetryRecord(
                    frame_id=f"zurich_{i:06d}",
                    lat=47.376900 + i * 1.36 / 111320.0,
                    lon=8.541700,
                    alt=450.0,
                )
            )
        poses_file = PosesFile(
            schema_version="1.0.0",
            dataset_label="test",
            poses=poses,
        )
        telemetry_file = TelemetryFile(
            schema_version="1.0.0",
            dataset_label="test",
            records=records,
        )
        return poses_file, telemetry_file

    def test_matching_by_frame_id_and_enu_conversion(self) -> None:
        poses_file, telemetry_file = self._poses_and_telemetry()
        result = gps_residual_from_files(poses_file, telemetry_file)
        assert result is not None
        # SfM centres are already in metres here (scale ~1), and the ENU
        # conversion of the lat/lon fixes reproduces the 1.36 m steps, so
        # the residual is at the centimetre level of the lat/lon rounding.
        assert result.median_m < 0.05
        assert result.num_matched_frames == 12

    def test_unmatched_frames_are_skipped_not_fatal(self) -> None:
        poses_file, telemetry_file = self._poses_and_telemetry()
        # Drop the telemetry for 2 frames and un-register 1 pose: 9 matched
        # frames remain, below the 10-frame minimum -> None, not an error.
        telemetry_file.records = telemetry_file.records[:-2]
        poses_file.poses[0].registered = False
        assert gps_residual_from_files(poses_file, telemetry_file) is None

    def test_missing_gps_fix_is_skipped(self) -> None:
        poses_file, telemetry_file = self._poses_and_telemetry()
        telemetry_file.records[3].lat = None
        telemetry_file.records[3].lon = None
        result = gps_residual_from_files(poses_file, telemetry_file)
        assert result is not None
        assert result.num_matched_frames == 11
