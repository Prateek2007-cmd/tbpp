"""Tests for video telemetry sync (Phase 2): Mode B with .BIN logs."""

from pathlib import Path

import numpy as np
import pytest

from uav3d.config import SyncConfig
from uav3d.exceptions import DataError
from uav3d.models.poses import PoseRecord, PosesFile, SfmCameraBlock, SfmMetrics
from uav3d.models.telemetry import TelemetryRecord
from uav3d.telemetry.dataflash_parser import DataFlashParseResult
from uav3d.telemetry.synchronizer import TelemetrySynchronizer

N_FRAMES = 12


@pytest.fixture
def video_poses_file() -> PosesFile:
    """Poses for a video run: video_ frame ids, PTS timestamps from t=1.2 s."""
    poses = [
        PoseRecord(
            frame_id=f"video_{i:06d}",
            timestamp_us=1_200_000 + i * 200_000,  # 5 Hz keyframe cadence
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            tx=float(i),
            ty=0.0,
            tz=5.0,
            camera_center_world=[float(-i), 0.0, -5.0],
            num_observations=150,
            registered=True,
        )
        for i in range(N_FRAMES)
    ]
    return PosesFile(
        schema_version="1.0.0",
        dataset_label="test video",
        coordinate_frame="cam_from_world",
        camera=SfmCameraBlock(
            camera_id=1,
            model="SIMPLE_PINHOLE",
            width=1280,
            height=720,
            params=[876.7, 640.0, 360.0],
        ),
        metrics=SfmMetrics(
            num_images_total=N_FRAMES,
            num_images_registered=N_FRAMES,
            registration_rate=1.0,
            mean_reprojection_error_px=0.5,
        ),
        poses=poses,
        unregistered_frames=[],
    )


@pytest.fixture
def video_telemetry() -> list[TelemetryRecord]:
    """GPS fixes at the same timestamps (already in the video-PTS basis)."""
    return [
        TelemetryRecord(
            timestamp_us=1_200_000 + i * 200_000,
            frame_id=None,
            lat=41.7716476 + i * 0.0001,
            lon=-0.7458537 + i * 0.0001,
            alt=324.72 + i,
            source="test_flight01.BIN",
            raw_fields={"Status": "4", "HDop": "0.65", "NSats": "10"},
        )
        for i in range(N_FRAMES)
    ]


class TestVideoModeAutoDetection:
    def test_video_frame_ids_select_mode_b(self, video_poses_file, video_telemetry):
        # Regression: video_000123 ends in digits, which used to
        # misclassify video runs as Zurich (Mode A).
        sync = TelemetrySynchronizer(config=SyncConfig())
        tel_file, enriched = sync.sync(
            poses=video_poses_file,
            raw_telemetry=video_telemetry,
            mode="auto",
        )
        assert "Mode B" in (tel_file.summary or {}).get("mode", "")
        assert tel_file.summary["match_rate"] == 1.0
        assert tel_file.sync_status == "timestamp_nearest_offset_estimated"
        assert all(p.telemetry_recorded for p in enriched.poses)
        assert tel_file.records[0].lat == pytest.approx(41.7716476)

    def test_explicit_video_mode(self, video_poses_file, video_telemetry):
        sync = TelemetrySynchronizer(config=SyncConfig())
        tel_file, _ = sync.sync(
            poses=video_poses_file,
            raw_telemetry=video_telemetry,
            mode="video",
        )
        assert tel_file.summary["matched_telemetry_frames"] == N_FRAMES


class TestBinPathDispatch:
    def test_bin_path_is_parsed_with_config_offset(
        self, monkeypatch, video_poses_file, video_telemetry
    ):
        seen: dict = {}

        def fake_parse(path, sync_offset_us):
            seen["path"] = str(path)
            seen["offset"] = sync_offset_us
            return DataFlashParseResult(records=list(video_telemetry))

        monkeypatch.setattr(
            "uav3d.telemetry.dataflash_parser.parse_dataflash", fake_parse
        )
        config = SyncConfig(dataflash_sync_offset_us=1_200_000)
        sync = TelemetrySynchronizer(config=config)
        tel_file, _ = sync.sync(
            poses=video_poses_file,
            raw_telemetry=Path("test_flight01.BIN"),
            mode="video",
        )
        assert seen["path"] == "test_flight01.BIN"
        assert seen["offset"] == 1_200_000
        assert tel_file.summary["match_rate"] == 1.0

    def test_unsupported_telemetry_extension_raises(self, video_poses_file):
        sync = TelemetrySynchronizer(config=SyncConfig())
        with pytest.raises(DataError, match="Unsupported telemetry file"):
            sync.sync(
                poses=video_poses_file,
                raw_telemetry=Path("telemetry.xyz"),
                mode="video",
            )


class TestSrtPathDispatch:
    def test_srt_path_is_parsed_with_zero_offset(
        self, monkeypatch, video_poses_file, video_telemetry
    ):
        from uav3d.telemetry.srt_parser import SrtParseResult

        def fake_parse(path):
            assert str(path) == "DJI_0679.SRT"
            return SrtParseResult(records=list(video_telemetry))

        monkeypatch.setattr("uav3d.telemetry.srt_parser.parse_dji_srt", fake_parse)
        sync = TelemetrySynchronizer(config=SyncConfig())
        tel_file, _ = sync.sync(
            poses=video_poses_file,
            raw_telemetry=Path("DJI_0679.SRT"),
            mode="video",
        )
        assert tel_file.summary["match_rate"] == 1.0
        # SRT times are video times: no sync offset is applied.
        assert tel_file.summary["telemetry_sync_offset_us"] == 0
        assert "srt_video_time" in tel_file.summary["telemetry_sync_offset_source"]


class TestXlsxPathDispatch:
    def test_xlsx_path_is_parsed_with_config_offset(
        self, monkeypatch, video_poses_file, video_telemetry
    ):
        from uav3d.telemetry.xlsx_parser import XlsxParseResult

        seen: dict = {}

        def fake_parse(path, sync_offset_us):
            seen["path"] = str(path)
            seen["offset"] = sync_offset_us
            return XlsxParseResult(
                records=list(video_telemetry),
                first_time_ms=84400,
                sync_offset_us=sync_offset_us,
                sync_offset_source="explicit",
            )

        monkeypatch.setattr("uav3d.telemetry.xlsx_parser.parse_dji_xlsx", fake_parse)
        config = SyncConfig(xlsx_sync_offset_us=50_000)
        sync = TelemetrySynchronizer(config=config)
        tel_file, _ = sync.sync(
            poses=video_poses_file,
            raw_telemetry=Path("DJI0016.xlsx"),
            mode="video",
        )
        assert seen["path"] == "DJI0016.xlsx"
        assert seen["offset"] == 50_000
        assert tel_file.summary["match_rate"] == 1.0
        assert tel_file.summary["telemetry_sync_offset_us"] == 50_000
        assert "xlsx_coarse" in tel_file.summary["telemetry_sync_offset_source"]
        assert "84400" in tel_file.summary["telemetry_sync_offset_source"]

    def test_xlsx_path_defaults_to_zero_offset(
        self, monkeypatch, video_poses_file, video_telemetry
    ):
        from uav3d.telemetry.xlsx_parser import XlsxParseResult

        seen: dict = {}

        def fake_parse(path, sync_offset_us):
            seen["offset"] = sync_offset_us
            return XlsxParseResult(
                records=list(video_telemetry),
                first_time_ms=84400,
                sync_offset_us=sync_offset_us,
                sync_offset_source="default (first fix = video t=0)",
            )

        monkeypatch.setattr("uav3d.telemetry.xlsx_parser.parse_dji_xlsx", fake_parse)
        # No xlsx_sync_offset_us configured: the parser anchors the first
        # fix to video t=0 with no residual correction.
        sync = TelemetrySynchronizer(config=SyncConfig())
        tel_file, _ = sync.sync(
            poses=video_poses_file,
            raw_telemetry=Path("DJI0016.xlsx"),
            mode="video",
        )
        assert seen["offset"] == 0
        assert tel_file.summary["telemetry_sync_offset_us"] == 0


class TestSyncConfigDefaults:
    def test_dataflash_offset_default_is_none_not_silent_1_2s(self):
        # The offset is per-flight (flight01 1.2 s, flight02 49.4 s): no
        # silent default. None means "read the flight's campaign.json".
        assert SyncConfig().dataflash_sync_offset_us is None

    def test_xlsx_offset_default_is_none_not_silent(self):
        # The xlsx parser already anchors the first fix to video t=0; the
        # config value is only the fine-scan residual (None = 0).
        assert SyncConfig().xlsx_sync_offset_us is None


def _make_poses(frame_times_us: list[int]) -> PosesFile:
    poses = [
        PoseRecord(
            frame_id=f"video_{i:06d}",
            timestamp_us=t,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            tx=float(i),
            ty=0.0,
            tz=5.0,
            camera_center_world=[float(-i), 0.0, -5.0],
            num_observations=150,
            registered=True,
        )
        for i, t in enumerate(frame_times_us)
    ]
    return PosesFile(
        schema_version="1.0.0",
        dataset_label="test video",
        coordinate_frame="cam_from_world",
        camera=SfmCameraBlock(
            camera_id=1, model="SIMPLE_PINHOLE", width=1280, height=720,
            params=[876.7, 640.0, 360.0],
        ),
        metrics=SfmMetrics(
            num_images_total=len(poses),
            num_images_registered=len(poses),
            registration_rate=1.0,
            mean_reprojection_error_px=0.5,
        ),
        poses=poses,
        unregistered_frames=[],
    )


def _make_fixes(fix_times_us: list[int]) -> list[TelemetryRecord]:
    return [
        TelemetryRecord(
            timestamp_us=t,
            frame_id=None,
            lat=41.7716476 + i * 0.0001,
            lon=-0.7458537 + i * 0.0001,
            alt=324.72,
            source="test.BIN",
            raw_fields={"Status": "4"},
        )
        for i, t in enumerate(fix_times_us)
    ]


class TestModeBOffsetEstimationSharpness:
    """The offset estimator must abstain on sparse telemetry (2026-11-17).

    PinPoint .BIN fixes are 5 Hz (200 ms apart); with max_delta 150 ms the
    frame-to-fix diffs spread over +-150 ms and the histogram "peak" is
    noise. The old code accepted any non-empty cluster and injected a
    spurious -147 ms offset on the real PinPoint window run, shifting
    matches by a fix interval. Now the peak needs >= 25% frame support.
    """

    def test_sparse_5hz_telemetry_abstains(self):
        # 250 frames at 60 fps (16.67 ms), fixes at 5 Hz, true offset 0.
        frame_times = [1_000_000 + int(i * 1_000_000 / 60) for i in range(250)]
        fix_times = [1_000_000 + i * 200_000 for i in range(30)]
        sync = TelemetrySynchronizer(config=SyncConfig(max_delta_us=150_000))
        tel_file, _ = sync.sync(
            poses=_make_poses(frame_times),
            raw_telemetry=_make_fixes(fix_times),
            mode="video",
        )
        assert tel_file.summary["match_rate"] == 1.0  # matching still works
        assert tel_file.offset_us == 0
        assert tel_file.offset_estimation == "abstained_no_sharp_peak"
        assert tel_file.summary["offset_support_fraction"] < 0.25

    def test_dense_telemetry_with_true_offset_is_estimated(self):
        # One fix per frame, all shifted by a constant +80 ms: a real
        # clock offset the estimator must find (the Zurich / SRT case).
        frame_times = [1_000_000 + i * 200_000 for i in range(50)]
        fix_times = [t + 80_000 for t in frame_times]
        sync = TelemetrySynchronizer(config=SyncConfig(max_delta_us=150_000))
        tel_file, _ = sync.sync(
            poses=_make_poses(frame_times),
            raw_telemetry=_make_fixes(fix_times),
            mode="video",
        )
        assert tel_file.offset_us == pytest.approx(80_000, abs=2_000)
        assert tel_file.offset_estimation == "median_offset"
        assert tel_file.summary["offset_support_fraction"] >= 0.9
