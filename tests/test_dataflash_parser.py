"""Tests for the ArduPilot DataFlash (.BIN) telemetry parser (Phase 2)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uav3d.exceptions import DataError
from uav3d.telemetry.dataflash_parser import (
    DataFlashParseResult,
    normalise_dataflash_timestamps,
    parse_dataflash,
    read_campaign_sync_offset,
    resolve_sync_offset,
)

# PinPoint flight01 ground truth (~/workspace/tbpp-data/DATASET_REPORT.md §1).
PINPOINT_BIN = Path("/home/hatch/workspace/tbpp-data/pinpoint/flight01/test_flight01.BIN")
PINPOINT_CAMPAIGN = Path("/home/hatch/workspace/tbpp-data/pinpoint/flight01/campaign.json")
PINPOINT_FIRST_FIX_TIMEUS = 87_782_991
PINPOINT_SYNC_OFFSET_US = 1_200_000
PINPOINT_GPS_MESSAGES = 4688
PINPOINT_SPAN_S = 938.666335

# PinPoint flight02 ground truth (~/workspace/tbpp-data/more-videos/).
FLIGHT02_BIN = Path("/home/hatch/workspace/tbpp-data/more-videos/pinpoint-flight02/test_flight02.BIN")
FLIGHT02_CAMPAIGN = Path("/home/hatch/workspace/tbpp-data/more-videos/pinpoint-flight02/campaign.json")
FLIGHT02_SYNC_OFFSET_US = 49_400_000
FLIGHT02_GPS_MESSAGES = 5882
FLIGHT02_SPAN_S = 1208.719


class TestTimestampNormalisation:
    """The boot-TimeUS -> video-PTS arithmetic, against the known 1.2 s offset."""

    def test_pinpoint_ground_truth_offset(self):
        raw = [
            PINPOINT_FIRST_FIX_TIMEUS,
            PINPOINT_FIRST_FIX_TIMEUS + 200_000,  # 5 Hz: +0.2 s
            PINPOINT_FIRST_FIX_TIMEUS + 10_000_000,  # +10 s
        ]
        normalised, first_fix = normalise_dataflash_timestamps(raw, PINPOINT_SYNC_OFFSET_US)
        assert first_fix == PINPOINT_FIRST_FIX_TIMEUS
        # The video starts 1.2 s AFTER the first fix, so the first fix
        # predates the video: it lands at video time MINUS the offset.
        # (The plus sign was tried first and failed the GCP cross-check:
        # every GCP frame missed by ~40 m. See TestGcpSyncCrossCheck.)
        assert normalised[0] == -PINPOINT_SYNC_OFFSET_US
        assert normalised[1] == -PINPOINT_SYNC_OFFSET_US + 200_000
        assert normalised[2] == -PINPOINT_SYNC_OFFSET_US + 10_000_000

    def test_zero_offset_keeps_relative_times(self):
        normalised, first_fix = normalise_dataflash_timestamps([500, 1500, 2500], 0)
        assert normalised == [0, 1000, 2000]
        assert first_fix == 500

    def test_empty_input_raises(self):
        with pytest.raises(DataError):
            normalise_dataflash_timestamps([], PINPOINT_SYNC_OFFSET_US)


def _fake_gps_message(timeus, status, lat, lon, alt, hdop=0.7, nsats=10):
    return SimpleNamespace(
        TimeUS=timeus, Status=status, Lat=lat, Lng=lon, Alt=alt, HDop=hdop, NSats=nsats
    )


class _FakeConnection:
    """Stand-in for pymavlink's connection: serves canned GPS messages."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.closed = False

    def recv_match(self, type=None, blocking=False):  # noqa: A002 - pymavlink's kwarg name
        if not self._messages:
            return None
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def _install_fake_connection(monkeypatch, tmp_path, messages):
    import pymavlink.mavutil as mavutil

    fake_bin = tmp_path / "fake.BIN"
    fake_bin.write_bytes(b"fake log")
    conn = _FakeConnection(messages)
    monkeypatch.setattr(mavutil, "mavlink_connection", lambda *a, **k: conn)
    return fake_bin, conn


class TestParseDataflashWithFakeLog:
    def test_status_filtering_field_mapping_and_timestamps(self, monkeypatch, tmp_path):
        messages = [
            _fake_gps_message(1_000_000, 2, 41.0, -0.7, 300.0),  # 2D fix: rejected
            _fake_gps_message(1_100_000, 3, 41.7716476, -0.7458537, 324.72, hdop=0.85, nsats=10),
            _fake_gps_message(1_300_000, 4, 41.7717000, -0.7458000, 330.5, hdop=0.6, nsats=12),
        ]
        fake_bin, _conn = _install_fake_connection(monkeypatch, tmp_path, messages)
        result = parse_dataflash(fake_bin, sync_offset_us=1_200_000)

        assert isinstance(result, DataFlashParseResult)
        assert result.total_gps_messages == 3
        assert result.kept_fixes == 2
        assert result.rejected_status == 1
        assert result.rejected_malformed == 0
        assert result.first_fix_timeus == 1_100_000
        assert result.source_name == "fake.BIN"
        assert result.span_s == pytest.approx(0.2)

        # Field mapping: lat/lon decimal degrees, alt metres MSL.
        rec0, rec1 = result.records
        assert rec0.lat == pytest.approx(41.7716476)
        assert rec0.lon == pytest.approx(-0.7458537)
        assert rec0.alt == pytest.approx(324.72)
        assert rec0.source == "fake.BIN"
        assert rec0.raw_fields["Status"] == "3"
        assert rec0.raw_fields["HDop"] == "0.85"
        assert rec0.raw_fields["NSats"] == "10"
        assert rec0.raw_fields["TimeUS"] == "1100000"
        # Timestamp normalisation: first fix -> -offset (it predates the
        # video), second -> -offset + 0.2 s.
        assert rec0.timestamp_us == -1_200_000
        assert rec1.timestamp_us == -1_000_000

    def test_malformed_messages_counted_not_fatal(self, monkeypatch, tmp_path):
        bad = SimpleNamespace(TimeUS=1_000_000, Status=4)  # no Lat/Lng/Alt
        good = _fake_gps_message(1_200_000, 4, 41.0, -0.7, 300.0)
        fake_bin, _conn = _install_fake_connection(monkeypatch, tmp_path, [bad, good])
        result = parse_dataflash(fake_bin, sync_offset_us=PINPOINT_SYNC_OFFSET_US)
        assert result.total_gps_messages == 2
        assert result.kept_fixes == 1
        assert result.rejected_malformed == 1

    def test_no_usable_fixes_raises(self, monkeypatch, tmp_path):
        messages = [_fake_gps_message(1_000_000, 1, 41.0, -0.7, 300.0)]
        fake_bin, _conn = _install_fake_connection(monkeypatch, tmp_path, messages)
        with pytest.raises(DataError, match="No GPS fixes"):
            parse_dataflash(fake_bin, sync_offset_us=PINPOINT_SYNC_OFFSET_US)


class TestPerFlightSyncOffset:
    """The offset is per-flight: campaign.json first, explicit flag next,
    fail loudly — never a silent 1.2 s default."""

    @pytest.mark.skipif(not PINPOINT_CAMPAIGN.exists(), reason="PinPoint flight01 not downloaded")
    def test_flight01_campaign_offset_is_1_2s(self):
        assert read_campaign_sync_offset(PINPOINT_CAMPAIGN) == 1_200_000

    @pytest.mark.skipif(not FLIGHT02_CAMPAIGN.exists(), reason="PinPoint flight02 not downloaded")
    def test_flight02_campaign_offset_is_49_4s(self):
        assert read_campaign_sync_offset(FLIGHT02_CAMPAIGN) == 49_400_000

    def test_campaign_offset_missing_key_is_none(self, tmp_path):
        campaign = tmp_path / "campaign.json"
        campaign.write_text('{"config": {}}')
        assert read_campaign_sync_offset(campaign) is None

    def test_campaign_offset_missing_file_is_none(self, tmp_path):
        assert read_campaign_sync_offset(tmp_path / "nope.json") is None

    def test_campaign_offset_non_numeric_raises(self, tmp_path):
        campaign = tmp_path / "campaign.json"
        campaign.write_text('{"config": {"sync": {"offset_s": "soon"}}}')
        with pytest.raises(DataError, match="non-numeric"):
            read_campaign_sync_offset(campaign)

    def test_explicit_offset_beats_campaign(self, tmp_path):
        campaign = tmp_path / "campaign.json"
        campaign.write_text('{"config": {"sync": {"offset_s": 49.4}}}')
        offset, source = resolve_sync_offset(
            tmp_path / "flight.BIN", sync_offset_us=2_000_000, campaign_json=campaign
        )
        assert offset == 2_000_000
        assert source == "explicit"

    def test_campaign_discovered_next_to_bin(self, tmp_path):
        campaign = tmp_path / "campaign.json"
        campaign.write_text('{"config": {"sync": {"offset_s": 7.5}}}')
        offset, source = resolve_sync_offset(tmp_path / "flight.BIN")
        assert offset == 7_500_000
        assert "campaign.json" in source

    def test_no_offset_anywhere_fails_loudly(self, tmp_path):
        with pytest.raises(DataError, match="No sync offset"):
            resolve_sync_offset(tmp_path / "flight.BIN")

    def test_parse_without_offset_fails_loudly(self, monkeypatch, tmp_path):
        messages = [_fake_gps_message(1_000_000, 4, 41.0, -0.7, 300.0)]
        fake_bin, _conn = _install_fake_connection(monkeypatch, tmp_path, messages)
        # No campaign.json next to the fake .BIN and no explicit offset.
        with pytest.raises(DataError, match="No sync offset"):
            parse_dataflash(fake_bin)

    def test_parse_uses_campaign_next_to_bin(self, monkeypatch, tmp_path):
        messages = [_fake_gps_message(1_000_000, 4, 41.0, -0.7, 300.0)]
        fake_bin, _conn = _install_fake_connection(monkeypatch, tmp_path, messages)
        (tmp_path / "campaign.json").write_text('{"config": {"sync": {"offset_s": 3.0}}}')
        result = parse_dataflash(fake_bin)
        assert result.sync_offset_us == 3_000_000
        assert "campaign.json" in result.sync_offset_source
        assert result.records[0].timestamp_us == -3_000_000

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataError, match="not found"):
            parse_dataflash(tmp_path / "nope.BIN")


@pytest.mark.skipif(not PINPOINT_BIN.exists(), reason="PinPoint dataset not downloaded")
class TestParseRealPinpointBin:
    """The real flight01 log: counts must match the dataset report exactly."""

    def test_real_log_counts_and_ranges(self):
        result = parse_dataflash(PINPOINT_BIN, sync_offset_us=PINPOINT_SYNC_OFFSET_US)

        assert result.total_gps_messages == PINPOINT_GPS_MESSAGES
        assert result.kept_fixes == PINPOINT_GPS_MESSAGES
        assert result.rejected_status == 0
        assert result.rejected_malformed == 0
        assert result.first_fix_timeus == PINPOINT_FIRST_FIX_TIMEUS
        assert result.span_s == pytest.approx(PINPOINT_SPAN_S, abs=0.01)

        # Video-PTS basis: first fix at -1.2 s (it predates the video),
        # last at 938.7 - 1.2 s.
        first_s, last_s = result.video_time_range_s
        assert first_s == pytest.approx(-1.2)
        assert last_s == pytest.approx(PINPOINT_SPAN_S - 1.2, abs=0.01)

        lats = np.array([r.lat for r in result.records])
        lons = np.array([r.lon for r in result.records])
        alts = np.array([r.alt for r in result.records])
        hdops = np.array([float(r.raw_fields["HDop"]) for r in result.records])
        assert lats.min() == pytest.approx(41.770531, abs=1e-5)
        assert lats.max() == pytest.approx(41.777462, abs=1e-5)
        assert lons.min() == pytest.approx(-0.750130, abs=1e-5)
        assert lons.max() == pytest.approx(-0.740388, abs=1e-5)
        assert alts.min() == pytest.approx(319.8, abs=0.5)
        assert alts.max() == pytest.approx(439.9, abs=0.5)
        assert hdops.mean() == pytest.approx(0.65, abs=0.02)

        # Status mix from the report: 476x Status 3, 4212x Status 4.
        statuses = [int(r.raw_fields["Status"]) for r in result.records]
        assert statuses.count(3) == 476
        assert statuses.count(4) == 4212

        # Timestamps strictly increasing, 5 Hz cadence (~200 ms steps).
        stamps = np.array([r.timestamp_us for r in result.records])
        assert bool(np.all(np.diff(stamps) > 0))
        assert np.median(np.diff(stamps)) == pytest.approx(200_000, abs=5_000)


@pytest.mark.skipif(not FLIGHT02_BIN.exists(), reason="PinPoint flight02 not downloaded")
class TestParseRealFlight02Bin:
    """Flight02's 49.4 s offset must be honoured — applying flight01's 1.2 s
    would shift every fix by 48.2 s (~850 m of flight)."""

    def test_flight02_parse_uses_its_own_offset(self):
        # No explicit offset: the parser must discover campaign.json next
        # to the .BIN and use flight02's 49.4 s, not flight01's 1.2 s.
        result = parse_dataflash(FLIGHT02_BIN)

        assert result.sync_offset_us == FLIGHT02_SYNC_OFFSET_US
        assert "campaign.json" in result.sync_offset_source
        assert result.total_gps_messages == FLIGHT02_GPS_MESSAGES
        assert result.kept_fixes == FLIGHT02_GPS_MESSAGES
        assert result.rejected_status == 0
        assert result.span_s == pytest.approx(FLIGHT02_SPAN_S, abs=0.01)

        # First fix lands at -49.4 s in the video-PTS basis (it predates
        # the video by the offset) — NOT at -1.2 s.
        first_s, last_s = result.video_time_range_s
        assert first_s == pytest.approx(-49.4)
        assert last_s == pytest.approx(FLIGHT02_SPAN_S - 49.4, abs=0.01)

        lats = np.array([r.lat for r in result.records])
        alts = np.array([r.alt for r in result.records])
        assert lats.min() == pytest.approx(42.468645, abs=1e-5)
        assert lats.max() == pytest.approx(42.475512, abs=1e-5)
        assert alts.min() == pytest.approx(678.3, abs=0.5)
        assert alts.max() == pytest.approx(812.4, abs=0.5)


class TestGcpSyncCrossCheck:
    """The sign of the sync offset, pinned against independent ground truth.

    PinPoint's points.csv / campaign.json tie video frames to drone
    lat/lon/alt. The .BIN fix nearest to a GCP frame's video time must be
    within a few metres of the published drone position. With the offset
    sign flipped (``+`` instead of ``-``) every frame misses by ~40 m —
    this test is what caught that bug (2026-11-17: the first sync run
    showed a 76 m median GPS residual, traced to the sign via these GCPs).
    """

    @pytest.mark.skipif(not PINPOINT_BIN.exists(), reason="PinPoint flight01 not downloaded")
    def test_flight01_fixes_match_gcp_drone_positions(self):
        import csv

        from uav3d.telemetry.synchronizer import haversine_distance_m

        result = parse_dataflash(PINPOINT_BIN)
        stamps = np.array([r.timestamp_us for r in result.records], dtype=np.float64)

        points_path = PINPOINT_BIN.parent / "points.csv"
        rows = list(csv.DictReader(open(points_path)))
        checked = 0
        worst_m = 0.0
        seen: set[int] = set()
        for row in rows:
            frame_index = int(row["frame_index"])
            if frame_index in seen:
                continue
            seen.add(frame_index)
            tv_us = float(row["tv_s"]) * 1e6
            j = int(np.argmin(np.abs(stamps - tv_us)))
            fix = result.records[j]
            dist = haversine_distance_m(
                fix.lat, fix.lon, fix.alt,
                float(row["drone_lat"]), float(row["drone_lng"]),
                float(row["alt_msl_m"]),
            )
            worst_m = max(worst_m, dist)
            checked += 1
        # 38 GCP frames; every one must agree to GPS-noise level (< 5 m).
        # A flipped sign gives ~40 m on every frame.
        assert checked == 38
        assert worst_m < 5.0

    @pytest.mark.skipif(not FLIGHT02_BIN.exists(), reason="PinPoint flight02 not downloaded")
    def test_flight02_fixes_match_campaign_drone_positions(self):
        import json

        from uav3d.telemetry.synchronizer import haversine_distance_m

        result = parse_dataflash(FLIGHT02_BIN)
        stamps = np.array([r.timestamp_us for r in result.records], dtype=np.float64)

        campaign = json.loads((FLIGHT02_BIN.parent / "campaign.json").read_text())
        worst_m = 0.0
        for frame in campaign["frames"]:
            tel = frame["telemetry"]
            tv_us = float(tel["tv"]) * 1e6
            j = int(np.argmin(np.abs(stamps - tv_us)))
            fix = result.records[j]
            dist = haversine_distance_m(
                fix.lat, fix.lon, fix.alt,
                float(tel["drone"]["lat"]), float(tel["drone"]["lng"]),
                float(tel["alt_msl"]),
            )
            worst_m = max(worst_m, dist)
        assert len(campaign["frames"]) == 51
        assert worst_m < 5.0
