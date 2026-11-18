"""Tests for the DJI .SRT telemetry parser (Phase 2).

Covers both KABR bracketed variants (the ``longtitude`` typo and the
correct ``longitude`` spelling) plus the GPS() family from the reference
samples. SRT times are video times — no sync offset is involved.
"""

from pathlib import Path

import numpy as np
import pytest

from uav3d.exceptions import DataError
from uav3d.telemetry.srt_parser import _parse_gps_from_text, parse_dji_srt

KABR_0679 = Path("/home/hatch/workspace/tbpp-data/more-videos/kabr-dji0679/DJI_0679.SRT")
KABR_0005 = Path("/home/hatch/workspace/tbpp-data/more-videos/kabr-dji0005/flight_1-DJI_0005.SRT")
SAMPLES = Path("/home/hatch/workspace/tbpp-data/dji-srt-samples")


class TestBracketedVariants:
    def test_longtitude_typo_variant_prefers_abs_alt(self):
        text = (
            "[iso : 110] [focal_len : 240],"
            "[latitude: 0.331223] [longtitude: 36.868989] "
            "[rel_alt: 44.260 abs_alt: 1830.628]"
        )
        lat, lon, alt, kind, raw = _parse_gps_from_text(text)
        assert lat == pytest.approx(0.331223)
        assert lon == pytest.approx(36.868989)
        assert alt == pytest.approx(1830.628)  # abs_alt (MSL), not rel_alt
        assert kind == "abs_msl"
        assert raw["rel_alt"] == "44.260"
        assert raw["abs_alt"] == "1830.628"

    def test_longitude_variant_uses_lone_altitude_as_rel_home(self):
        text = (
            "[iso : 100] [focal_len : 224],"
            "[latitude: 0.395644] [longitude: 36.886641] [altitude: 54.100000]"
        )
        lat, lon, alt, kind, raw = _parse_gps_from_text(text)
        assert lat == pytest.approx(0.395644)
        assert lon == pytest.approx(36.886641)
        assert alt == pytest.approx(54.1)
        assert kind == "rel_home"

    def test_focal_len_is_never_read(self):
        # focal_len varies 224 vs 240 between videos of the same drone and
        # is not a calibrated focal — the parser must not surface it.
        text = "[latitude: 1.0] [longitude: 2.0] [altitude: 10.0] [focal_len : 240]"
        _lat, _lon, _alt, _kind, raw = _parse_gps_from_text(text)
        assert "focal_len" not in raw

    def test_no_gps_returns_none(self):
        assert _parse_gps_from_text("[iso : 100] [shutter : 1/2000.0]") is None


class TestGpsParenFamily:
    def test_gps_no_space_with_m_suffix(self):
        lat, lon, alt, kind, _ = _parse_gps_from_text("GPS(36.6146,-6.1120,0.0M) BAROMETER:0.3M")
        assert (lat, lon, alt) == pytest.approx((36.6146, -6.1120, 0.0))
        assert kind == "gps_alt"

    def test_gps_with_space_no_suffix(self):
        lat, lon, alt, kind, _ = _parse_gps_from_text(
            "GPS (-58.851745, -34.237922, 15), HOME (-58.847509, -34.232707, -57.98m)"
        )
        assert (lat, lon, alt) == pytest.approx((-58.851745, -34.237922, 15.0))
        assert kind == "gps_alt"


class TestParseSyntheticSrt:
    def test_timestamps_are_video_pts_and_blocks_skipped(self, tmp_path):
        srt = tmp_path / "test.SRT"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n"
            "GPS(10.0, 20.0, 100.0)\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\n"
            "[iso : 100] no gps here\n\n"
            "3\nnot a timestamp\n"
            "GPS(11.0, 21.0, 101.0)\n\n"
            "4\n00:00:03,500 --> 00:00:04,500\n"
            "GPS(12.0, 22.0, 102.0)\n"
        )
        result = parse_dji_srt(srt)
        assert result.total_blocks == 4
        assert result.kept == 2
        assert result.skipped_no_gps == 1
        assert result.skipped_malformed == 1
        assert result.records[0].timestamp_us == 0
        assert result.records[1].timestamp_us == 3_500_000
        assert result.records[1].lat == pytest.approx(12.0)
        assert result.video_time_range_s == pytest.approx((0.0, 3.5))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataError, match="not found"):
            parse_dji_srt(tmp_path / "nope.SRT")

    def test_no_gps_anywhere_raises(self, tmp_path):
        srt = tmp_path / "empty.SRT"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
        with pytest.raises(DataError, match="No GPS fixes"):
            parse_dji_srt(srt)


@pytest.mark.skipif(not KABR_0679.exists(), reason="KABR DJI_0679 not downloaded")
class TestRealKabr0679:
    def test_full_file(self):
        result = parse_dji_srt(KABR_0679)
        assert result.total_blocks == 4304
        assert result.kept == 4304
        assert result.skipped_no_gps == 0
        assert result.formats_seen == ["bracketed_longtitude"]
        first, last = result.records[0], result.records[-1]
        assert first.timestamp_us == 0
        assert first.lat == pytest.approx(0.331223)
        assert first.lon == pytest.approx(36.868989)
        assert first.alt == pytest.approx(1830.628)
        assert first.raw_fields["alt_kind"] == "abs_msl"
        assert last.timestamp_us == 143_561_000
        stamps = np.array([r.timestamp_us for r in result.records])
        assert bool(np.all(np.diff(stamps) > 0))
        # Per-frame SRT at ~30 fps: median step ~33 ms.
        assert np.median(np.diff(stamps)) == pytest.approx(33_000, abs=2_000)


@pytest.mark.skipif(not KABR_0005.exists(), reason="KABR DJI_0005 not downloaded")
class TestRealKabr0005:
    def test_full_file(self):
        result = parse_dji_srt(KABR_0005)
        assert result.total_blocks == 496
        assert result.kept == 496
        assert result.formats_seen == ["bracketed_longitude"]
        first = result.records[0]
        assert first.lat == pytest.approx(0.395644)
        assert first.lon == pytest.approx(36.886641)
        assert first.alt == pytest.approx(54.1)
        assert first.raw_fields["alt_kind"] == "rel_home"


@pytest.mark.skipif(not (SAMPLES / "matrice_300.srt").exists(), reason="DJI SRT samples not downloaded")
class TestReferenceSamples:
    def test_matrice_300(self):
        result = parse_dji_srt(SAMPLES / "matrice_300.srt")
        assert result.kept == 381
        assert result.formats_seen == ["gps_paren"]
        assert result.records[0].lat == pytest.approx(36.6146)
        assert result.records[0].lon == pytest.approx(-6.1120)
        assert result.records[0].alt == pytest.approx(0.0)
        assert result.video_time_range_s == pytest.approx((0.0, 380.0))

    def test_p4_rtk(self):
        result = parse_dji_srt(SAMPLES / "p4_rtk.SRT")
        assert result.kept == 55
        assert result.records[0].lat == pytest.approx(-58.851745)
        assert result.records[0].lon == pytest.approx(-34.237922)
        assert result.records[0].alt == pytest.approx(15.0)

    def test_p4p(self):
        result = parse_dji_srt(SAMPLES / "p4p_sample.SRT")
        assert result.kept == 5
        assert result.records[0].lat == pytest.approx(8.2090)
        assert result.records[0].lon == pytest.approx(47.4692)
        assert result.records[0].alt == pytest.approx(18.0)
