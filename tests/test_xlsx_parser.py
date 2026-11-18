"""Tests for the DJI AirData (.xlsx) telemetry parser (GrapeSLAM Stage 0).

Covers header-whitespace stripping, feet->metre conversion, the first-fix
time anchor (xlsx t=84400 ms <-> video t=0 on DJI0016), malformed-row
handling, and the real DJI0016.xlsx export (543/543 fixes).
"""

from pathlib import Path

import openpyxl
import pytest

from uav3d.exceptions import DataError
from uav3d.telemetry.xlsx_parser import (
    FEET_TO_M,
    normalise_xlsx_timestamps,
    parse_dji_xlsx,
)

DJI0016_XLSX = Path(
    "/home/hatch/workspace/tbpp-data/more-videos/grapeslam-dji0016/DJI0016.xlsx"
)

# Header row mirroring the real AirData export, including the leading
# whitespace on some columns.
HEADERS = [
    "time(millisecond)",
    "latitude",
    "longitude",
    "height_above_takeoff(feet)",
    "altitude_above_seaLevel(feet)",
    "speed(mph)",
    "satellites",
    "gpslevel",
    " compass_heading(degrees)",
    "gimbal_pitch(degrees)",
]


def _write_workbook(path: Path, headers: list[str], rows: list[list[object]]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


@pytest.fixture
def small_xlsx(tmp_path: Path) -> Path:
    """A 3-fix synthetic AirData export (first fix at 84400 ms, like DJI0016)."""
    return _write_workbook(
        tmp_path / "synthetic.xlsx",
        HEADERS,
        [
            [84400, 41.93783433, -8.792473334, 4.92126, 171.5626482, 0.0, 18, 5, 261.2, 0.0],
            [84500, 41.93783433, -8.792473353, 4.92126, 171.5626482, 0.5, 18, 5, 261.2, -1.0],
            [84600, 41.93783400, -8.792473500, 5.0, 172.0, 1.0, 17, 5, 261.0, -2.0],
        ],
    )


class TestHeaderStripping:
    def test_leading_whitespace_headers_are_stripped(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        assert result.kept == 3
        # compass_heading has a leading space in the header; it must still
        # land in raw_fields under its stripped name.
        assert result.records[0].raw_fields["compass_heading(degrees)"] == "261.2"

    def test_missing_required_column_raises(self, tmp_path: Path):
        path = _write_workbook(
            tmp_path / "no_alt.xlsx",
            ["time(millisecond)", "latitude", "longitude"],
            [[84400, 41.9, -8.79]],
        )
        with pytest.raises(DataError, match="missing required columns"):
            parse_dji_xlsx(path)

    def test_no_log_sheet_raises(self, tmp_path: Path):
        path = _write_workbook(tmp_path / "empty.xlsx", ["foo", "bar"], [[1, 2]])
        with pytest.raises(DataError, match="No AirData log sheet"):
            parse_dji_xlsx(path)

    def test_log_sheet_found_when_not_first(self, tmp_path: Path):
        # AirData puts the log on Sheet1 and frame indices on Sheet2; the
        # parser must find the log sheet by its headers, not its position.
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet2"
        ws.append(["000000"])
        ws.append(["000001"])
        log = wb.create_sheet("Sheet1")
        log.append(HEADERS)
        log.append([84400, 41.93783433, -8.792473334, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0])
        path = tmp_path / "two_sheets.xlsx"
        wb.save(path)
        result = parse_dji_xlsx(path)
        assert result.sheet_name == "Sheet1"
        assert result.kept == 1


class TestTimeBase:
    def test_first_fix_anchors_video_t0(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        assert result.first_time_ms == 84400
        assert result.last_time_ms == 84600
        assert result.span_s == pytest.approx(0.2)
        assert [r.timestamp_us for r in result.records] == [0, 100_000, 200_000]
        assert result.video_time_range_s == pytest.approx((0.0, 0.2))

    def test_sync_offset_shifts_timestamps(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx, sync_offset_us=50_000)
        assert [r.timestamp_us for r in result.records] == [-50_000, 50_000, 150_000]
        assert result.sync_offset_us == 50_000
        assert result.sync_offset_source == "explicit"

    def test_normalise_xlsx_timestamps_pure_function(self):
        timestamps, first = normalise_xlsx_timestamps([84400, 84500, 138600])
        assert first == 84400
        assert timestamps == [0, 100_000, 54_200_000]
        timestamps, _ = normalise_xlsx_timestamps([84400, 84500], sync_offset_us=150_000)
        assert timestamps == [-150_000, -50_000]

    def test_normalise_empty_raises(self):
        with pytest.raises(DataError, match="no GPS fixes"):
            normalise_xlsx_timestamps([])


class TestUnitsAndFields:
    def test_feet_to_metres_conversion(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        # altitude_above_seaLevel 171.5626482 ft -> metres MSL
        assert result.records[0].alt == pytest.approx(171.5626482 * FEET_TO_M, abs=1e-6)
        assert result.records[0].raw_fields["alt_kind"] == "abs_msl"
        # height above takeoff 4.92126 ft -> 1.5 m, kept in raw_fields
        assert float(result.records[0].raw_fields["height_above_takeoff_m"]) == pytest.approx(1.5, abs=1e-3)

    def test_speed_mph_kept_and_converted(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        assert float(result.records[2].raw_fields["speed_mph"]) == pytest.approx(1.0)
        assert float(result.records[2].raw_fields["speed_ms"]) == pytest.approx(0.44704, abs=1e-3)

    def test_gpslevel_and_satellites_in_raw_fields(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        assert result.records[0].raw_fields["satellites"] == "18"
        assert result.records[0].raw_fields["gpslevel"] == "5"
        assert float(result.records[0].raw_fields["gimbal_pitch(degrees)"]) == pytest.approx(0.0)

    def test_lat_lon_sanity(self, small_xlsx: Path):
        result = parse_dji_xlsx(small_xlsx)
        for rec in result.records:
            assert -90.0 <= rec.lat <= 90.0
            assert -180.0 <= rec.lon <= 180.0
            assert rec.source == "synthetic.xlsx"


class TestRowHandling:
    def test_rows_without_gps_skipped_and_counted(self, tmp_path: Path):
        path = _write_workbook(
            tmp_path / "gaps.xlsx",
            HEADERS,
            [
                [84400, 41.93783433, -8.792473334, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0],
                [84500, None, None, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0],
                [84600, 41.93783400, -8.792473500, 5.0, 172.0, 1.0, 17, 5, 261.0, -2.0],
            ],
        )
        result = parse_dji_xlsx(path)
        assert result.kept == 2
        assert result.skipped_no_gps == 1
        assert result.total_rows == 3

    def test_malformed_time_and_out_of_range_lat_skipped(self, tmp_path: Path):
        path = _write_workbook(
            tmp_path / "bad.xlsx",
            HEADERS,
            [
                [84400, 41.93783433, -8.792473334, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0],
                ["not-a-time", 41.93783433, -8.792473334, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0],
                [84600, 999.0, -8.792473500, 5.0, 172.0, 1.0, 17, 5, 261.0, -2.0],
            ],
        )
        result = parse_dji_xlsx(path)
        assert result.kept == 1
        assert result.skipped_malformed == 2

    def test_no_fixes_raises(self, tmp_path: Path):
        path = _write_workbook(
            tmp_path / "nofix.xlsx",
            HEADERS,
            [[84400, None, None, 4.92126, 171.5626482, 0, 18, 5, 261.2, 0]],
        )
        with pytest.raises(DataError, match="No GPS fixes found"):
            parse_dji_xlsx(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(DataError, match="not found"):
            parse_dji_xlsx(tmp_path / "does_not_exist.xlsx")


@pytest.mark.skipif(not DJI0016_XLSX.exists(), reason="GrapeSLAM DJI0016.xlsx not downloaded")
class TestRealDji0016:
    def test_all_543_fixes_parsed(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        assert result.total_rows == 543
        assert result.kept == 543
        assert result.skipped_no_gps == 0
        assert result.skipped_malformed == 0
        assert len(result.records) == 543

    def test_time_base_and_span(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        assert result.first_time_ms == 84400
        assert result.last_time_ms == 138600
        assert result.span_s == pytest.approx(54.2)
        # First fix anchors video t=0; last fix at 54.2 s (video is 54.054 s).
        assert result.records[0].timestamp_us == 0
        assert result.records[-1].timestamp_us == 54_200_000
        # 10 Hz cadence: consecutive fixes 100 ms apart.
        gaps = [
            b.timestamp_us - a.timestamp_us
            for a, b in zip(result.records, result.records[1:])
        ]
        assert all(g == 100_000 for g in gaps)

    def test_lat_lon_span_sane(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        lats = [r.lat for r in result.records]
        lons = [r.lon for r in result.records]
        # GrapeSLAM vineyard, northern Portugal.
        assert min(lats) == pytest.approx(41.9377, abs=0.001)
        assert max(lats) == pytest.approx(41.9378, abs=0.001)
        assert min(lons) == pytest.approx(-8.7932, abs=0.001)
        assert max(lons) == pytest.approx(-8.7925, abs=0.001)
        # Track span is tens of metres, not kilometres.
        assert (max(lats) - min(lats)) * 111_000 < 100.0
        assert (max(lons) - min(lons)) * 111_000 < 100.0

    def test_altitude_msl_range(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        alts = [r.alt for r in result.records]
        # Plan expectation: ~51.5-53.9 m MSL, varying (not constant).
        assert min(alts) == pytest.approx(51.5, abs=1.0)
        assert max(alts) == pytest.approx(53.9, abs=1.0)
        assert max(alts) - min(alts) > 1.0

    def test_height_above_takeoff_range(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        heights = [
            float(r.raw_fields["height_above_takeoff_m"]) for r in result.records
        ]
        # Plan expectation: ~0.7-3.1 m above takeoff (low flight).
        assert min(heights) == pytest.approx(0.7, abs=0.5)
        assert max(heights) == pytest.approx(3.1, abs=0.5)

    def test_gpslevel_and_satellites(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        levels = {r.raw_fields["gpslevel"] for r in result.records}
        sats = [int(r.raw_fields["satellites"]) for r in result.records]
        assert levels == {"5"}
        assert min(sats) >= 15
        assert max(sats) <= 18

    def test_speed_column_present(self):
        result = parse_dji_xlsx(DJI0016_XLSX)
        speeds = [float(r.raw_fields["speed_mph"]) for r in result.records]
        assert max(speeds) > 0.0
        assert max(speeds) < 15.0  # slow vineyard flight, not a fast transit

    def test_mode_b_sync_end_to_end_with_real_xlsx(self):
        # Proves `--telemetry DJI0016.xlsx` works through the real Mode B
        # dispatch: synthetic 5 Hz poses spanning the flight all match.
        from uav3d.config import SyncConfig
        from uav3d.models.poses import (
            PoseRecord,
            PosesFile,
            SfmCameraBlock,
            SfmMetrics,
        )
        from uav3d.telemetry.synchronizer import TelemetrySynchronizer

        n_frames = 270  # 5 Hz over 54 s
        poses = [
            PoseRecord(
                frame_id=f"video_{i:06d}",
                timestamp_us=i * 200_000,
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
            for i in range(n_frames)
        ]
        poses_file = PosesFile(
            schema_version="1.0.0",
            dataset_label="test grapeslam",
            coordinate_frame="cam_from_world",
            camera=SfmCameraBlock(
                camera_id=1,
                model="SIMPLE_PINHOLE",
                width=3840,
                height=2160,
                params=[2560.0, 1920.0, 1080.0],
            ),
            metrics=SfmMetrics(
                num_images_total=n_frames,
                num_images_registered=n_frames,
                registration_rate=1.0,
                mean_reprojection_error_px=0.5,
            ),
            poses=poses,
            unregistered_frames=[],
        )
        sync = TelemetrySynchronizer(config=SyncConfig())
        tel_file, enriched = sync.sync(
            poses=poses_file,
            raw_telemetry=DJI0016_XLSX,
            mode="video",
        )
        assert tel_file.summary["match_rate"] == 1.0
        assert tel_file.sync_status == "timestamp_nearest_offset_estimated"
        assert tel_file.summary["telemetry_sync_offset_us"] == 0
        assert "xlsx_coarse" in tel_file.summary["telemetry_sync_offset_source"]
        assert all(p.telemetry_recorded for p in enriched.poses)
        # First frame matches the first fix (video t=0 anchor).
        assert tel_file.records[0].lat == pytest.approx(41.93783433)
