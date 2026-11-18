"""Tests for the robust GNSS metric-scale fit (Phase 2).

The fit aligns SfM camera centres onto GPS positions (ENU metres about
the first fix) with Umeyama Sim(3), rejects outlier fixes by
median + k*MAD, and re-fits on the inliers. The rejected count is part
of the result — the fit never silently runs through bad fixes.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from uav3d.dense.cameras import CameraView, Intrinsics
from uav3d.dense.config import DenseConfig
from uav3d.dense.pipeline import _resolve_metric_scale, resolve_gnss_alignment
from uav3d.exceptions import DataError
from uav3d.telemetry.georef import GeorefFrame

ORIGIN = (41.7716476, -0.7458537, 324.72)  # PinPoint flight01 first fix
TRUE_SCALE = 3.5  # metres per SfM unit


def _synthetic_run(tmp_path: Path, n: int = 24, outlier_idx: tuple[int, ...] = (),
                   noise_m: float = 0.3, seed: int = 0):
    """Build views + telemetry.json for a known Sim(3) ground truth.

    ENU positions follow a gently zigzagging flight; SfM centres are the
    same points pushed through the inverse of a known Sim(3)
    (scale TRUE_SCALE, 20-degree yaw, translation (50, -30, 10)).
    Outlier fixes are shifted +60 m north before conversion to lat/lon.
    """
    rng = np.random.default_rng(seed)
    frame = GeorefFrame.from_first_fix(*ORIGIN)

    enu = np.array([
        [i * 8.0, 5.0 if i % 2 == 0 else -5.0, 100.0 + 0.5 * i]
        for i in range(n)
    ], dtype=np.float64)
    enu_noisy = enu + rng.normal(0.0, noise_m, size=enu.shape)
    for i in outlier_idx:
        enu_noisy[i] = enu_noisy[i] + np.array([0.0, 60.0, 0.0])

    theta = np.deg2rad(20.0)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    trans = np.array([50.0, -30.0, 10.0])
    sfm_centres = (enu - trans) @ rot / TRUE_SCALE  # inverse Sim(3), noise-free

    intr = Intrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, width=1280, height=720)
    views = [
        CameraView(
            index=i,
            frame_id=f"video_{i:06d}",
            image_path=tmp_path / f"frame_{i}.jpg",
            R=np.eye(3),
            t=-sfm_centres[i],  # centre = -R.T @ t = sfm_centres[i]
            intrinsics=intr,
        )
        for i in range(n)
    ]

    records = []
    for i in range(n):
        lat, lon, alt = frame.from_enu(*enu_noisy[i])
        records.append({
            "timestamp_us": 1_200_000 + i * 200_000,
            "frame_id": f"video_{i:06d}",
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "source": "test.BIN",
        })
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.json").write_text(json.dumps({"records": records}))
    return run_dir, views


class TestResolveGnssAlignment:
    def test_recovers_known_scale(self, tmp_path):
        run_dir, views = _synthetic_run(tmp_path)
        alignment = resolve_gnss_alignment(run_dir, views)
        assert alignment.scale == pytest.approx(TRUE_SCALE, rel=0.01)
        assert alignment.alignment_rms_m < 1.0
        assert len(alignment.inlier_frame_ids) == 24
        assert alignment.rejected_frame_ids == []
        # Origin = the FIRST GPS fix (the first telemetry record), not the
        # ENU frame's reference point — the flight starts ~5 m north of it.
        first_rec = json.loads((run_dir / "telemetry.json").read_text())["records"][0]
        assert alignment.origin[0] == pytest.approx(first_rec["lat"])
        assert alignment.origin[1] == pytest.approx(first_rec["lon"])
        assert alignment.origin[2] == pytest.approx(first_rec["alt"])
        assert alignment.frame.utm_epsg == 32630

    def test_outliers_rejected_and_reported(self, tmp_path):
        run_dir, views = _synthetic_run(tmp_path, outlier_idx=(5, 17))
        alignment = resolve_gnss_alignment(run_dir, views)
        # Both wild fixes rejected, scale still recovered.
        assert sorted(alignment.rejected_frame_ids) == ["video_000005", "video_000017"]
        assert len(alignment.inlier_frame_ids) == 22
        assert alignment.scale == pytest.approx(TRUE_SCALE, rel=0.02)
        assert alignment.alignment_rms_m < 1.0

    def test_outliers_would_corrupt_a_plain_fit(self, tmp_path):
        # Control: without rejection the same data fits with a 16+ m RMS —
        # the rejection is doing real work, not decoration. (The scale
        # itself barely moves here because a pure north shift is absorbed
        # into rotation/translation; the RMS is where the damage shows.)
        run_dir, views = _synthetic_run(tmp_path, outlier_idx=(5, 17))
        alignment = resolve_gnss_alignment(run_dir, views, reject_k_mad=1e9)
        assert alignment.rejected_frame_ids == []
        assert alignment.alignment_rms_m > 5.0

    def test_missing_telemetry_raises(self, tmp_path):
        _, views = _synthetic_run(tmp_path)
        empty_run = tmp_path / "empty_run"
        empty_run.mkdir()
        with pytest.raises(DataError, match="telemetry.json"):
            resolve_gnss_alignment(empty_run, views)

    def test_too_few_fixes_raises(self, tmp_path):
        run_dir, views = _synthetic_run(tmp_path, n=6)
        with pytest.raises(DataError, match="at least 10"):
            resolve_gnss_alignment(run_dir, views)


class TestResolveMetricScale:
    def test_gnss_branch_returns_alignment(self, tmp_path):
        run_dir, views = _synthetic_run(tmp_path)
        config = DenseConfig(scale_source="gnss")
        scale, source, status, extra, alignment = _resolve_metric_scale(config, run_dir, views)
        assert scale == pytest.approx(TRUE_SCALE, rel=0.01)
        assert source == "gnss"
        assert status == "gnss_aligned"
        assert extra["gnss_frames"] == 24.0
        assert extra["gnss_rejected"] == 0.0
        assert alignment is not None
        assert alignment.scale == pytest.approx(TRUE_SCALE, rel=0.01)

    def test_fixed_branch(self, tmp_path):
        config = DenseConfig(scale_source="fixed", scale_factor=6.76)
        scale, source, status, extra, alignment = _resolve_metric_scale(
            config, tmp_path, []
        )
        assert scale == 6.76
        assert source == "fixed"
        assert status == "fixed_by_user"
        assert alignment is None

    def test_fixed_without_factor_raises(self, tmp_path):
        config = DenseConfig(scale_source="fixed")
        with pytest.raises(DataError, match="scale_factor"):
            _resolve_metric_scale(config, tmp_path, [])

    def test_none_branch_is_unscaled(self, tmp_path):
        config = DenseConfig(scale_source="none")
        scale, source, status, extra, alignment = _resolve_metric_scale(
            config, tmp_path, []
        )
        assert scale == 1.0
        assert source == "none"
        assert status == "unverified_monocular_scale"
        assert alignment is None

    def test_gnss_default_k_mad(self):
        assert DenseConfig().gnss_reject_k_mad == 5.0
