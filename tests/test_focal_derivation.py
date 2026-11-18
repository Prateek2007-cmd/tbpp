"""Tests for focal-length derivation (Phase 1).

The focal length is the accuracy-limiting parameter on single-pass forward
flight: the Nov 16 experiment (docs/HANDOFF_NEXT_SESSION.md §12) showed a
384 px prior (the 0.8*width guess) against a true 265 px focal gives a
3.84 m median error while reprojection error stays ~0.28 px. These tests
lock in the derivation paths, the provenance ladder, and the traps the
derivation must refuse: the 35mm-equivalent focal, an fx/fy mismatch, and
implausible focal / field-of-view values.
"""

from __future__ import annotations

import json
import math

import pytest
from PIL import Image

from uav3d.calibration.focal import (
    SENSOR_FORMATS,
    FocalSpec,
    focal_from_exif,
    focal_from_measured_px,
    focal_from_mm,
    guessed_focal_px,
    resolve_focal,
)
from uav3d.config import SfmConfig
from uav3d.exceptions import Uav3dError
from uav3d.models.calibration_io import read_calibration, write_calibration
from uav3d.sources.video import VideoFileSource


class TestFocalFromMm:
    def test_mavic_3_main_camera(self) -> None:
        # DJI Mavic 3 Hasselblad: 12.29 mm lens, 4/3 sensor 17.3x13.0 mm,
        # 5280x3956 stills. fx = 12.29/17.3*5280, fy = 12.29/13.0*3956.
        derived = focal_from_mm(12.29, 17.3, 5280, 3956, sensor_height_mm=13.0)
        assert derived.fx == pytest.approx(3751.4, abs=0.5)
        assert derived.fy == pytest.approx(3739.9, abs=0.5)
        assert derived.provenance == "sensor_spec"
        assert derived.cx == 2640.0
        assert derived.cy == 1978.0
        # Horizontal FOV: 2*atan(17.3 / (2*12.29)) = 70.3 deg. (DJI's quoted
        # 84 deg is the *diagonal* FOV — the 24 mm equivalence matches the
        # diagonal, not the horizontal, because 4:3 and 3:2 differ in aspect.)
        assert derived.hfov_deg == pytest.approx(70.3, abs=0.5)

    def test_sensor_format_lookup(self) -> None:
        derived = focal_from_mm(12.29, None, 5280, 3956, sensor_format="4/3")
        assert derived.fx == pytest.approx(3751.4, abs=0.5)
        assert derived.sensor_width_mm == 17.3
        assert derived.sensor_height_mm == 13.0

    def test_unknown_sensor_format_is_rejected(self) -> None:
        with pytest.raises(Uav3dError, match="Unknown sensor_format"):
            focal_from_mm(12.29, None, 5280, 3956, sensor_format="APS-C")

    def test_square_pixels_when_only_width_known(self) -> None:
        derived = focal_from_mm(8.8, 13.2, 1920, 1080)
        assert derived.fx == derived.fy == pytest.approx(1280.0)

    def test_missing_sensor_size_is_rejected(self) -> None:
        with pytest.raises(Uav3dError, match="without sensor dimensions"):
            focal_from_mm(8.8, None, 1920, 1080)

    def test_sensor_table_has_only_verified_formats(self) -> None:
        assert SENSOR_FORMATS["4/3"].width_mm == 17.3
        assert SENSOR_FORMATS["1-inch"].width_mm == 13.2
        assert SENSOR_FORMATS["1/1.3-inch"].width_mm == 9.6


class TestFocalSanityChecks:
    def test_fx_fy_mismatch_is_rejected(self) -> None:
        # 10 mm lens, 1-inch sensor (13.2x8.8), 16:9 video: fx=1454, fy=1227
        # — a 17% mismatch, because the 16:9 video is a crop of the 3:2
        # sensor and the full-frame sensor height does not apply. Rejecting
        # this is the point: the caller must supply the effective crop dims.
        with pytest.raises(Uav3dError, match="differ by"):
            focal_from_mm(10.0, 13.2, 1920, 1080, sensor_height_mm=8.8)

    def test_fx_fy_within_2_percent_passes(self) -> None:
        derived = focal_from_measured_px(1000.0, 1920, 1080, fy=1015.0)
        assert derived.fx == 1000.0
        assert derived.fy == 1015.0

    def test_too_short_focal_is_rejected(self) -> None:
        # 0.2 * width: far below the 0.3x floor.
        with pytest.raises(Uav3dError, match="outside the plausible range"):
            focal_from_measured_px(384.0, 1920, 1080)

    def test_too_long_focal_is_rejected(self) -> None:
        # 4 * width: far above the 3x ceiling.
        with pytest.raises(Uav3dError, match="outside the plausible range"):
            focal_from_measured_px(7680.0, 1920, 1080)

    def test_35mm_equivalent_focal_is_NOT_caught_by_the_range_gate(self) -> None:
        # Honest limit of the plausibility gate: feeding the 24 mm
        # *equivalent* as the real focal for a Mavic 3 (real: 12.29 mm)
        # doubles the focal to ~7324 px — still inside the 0.3x-3x range, so
        # the gate passes it. The gate catches gross errors (wrong units, a
        # factor of 10), not a 2x mistake in the middle of the range. The
        # defences against the 35mm mistake are the EXIF rejection (a still
        # carrying only FocalLengthIn35mmFilm raises) and the SOP — which is
        # why both exist. This test locks in that the gate must NOT be
        # relied on for this trap.
        derived = focal_from_mm(24.0, 17.3, 5280, 3956, sensor_height_mm=13.0)
        assert derived.fx == pytest.approx(7324.3, abs=1.0)
        # 24 / 12.29 = 1.95x the true focal — a ~95% error the gate cannot see.
        assert derived.fx / 3751.4 == pytest.approx(24.0 / 12.29, abs=0.01)


class TestMeasuredFocal:
    def test_synthetic_benchmark_focal(self) -> None:
        # The synthetic street-canyon video: true focal 265 px at 480x270.
        derived = focal_from_measured_px(265.0, 480, 270)
        assert derived.fx == derived.fy == 265.0
        assert derived.provenance == "measured"
        assert derived.cx == 240.0
        assert derived.cy == 135.0

    def test_measured_principal_point_is_kept(self) -> None:
        derived = focal_from_measured_px(1000.0, 1920, 1080, cx=950.0, cy=545.0)
        assert derived.cx == 950.0
        assert derived.cy == 545.0


class TestExifFocal:
    def _write_still(self, tmp_path, exif_tags, size=(4000, 3000)):
        path = tmp_path / "still.jpg"
        img = Image.new("RGB", size, color=(120, 130, 140))
        exif = Image.Exif()
        for tag, value in exif_tags.items():
            exif[tag] = value
        img.save(path, exif=exif)
        return path

    def test_exif_focal_length_with_sensor_format(self, tmp_path) -> None:
        # Mavic 3 still: EXIF FocalLength 12.29 mm, 4/3 sensor, still is
        # 4000x3000 and the video is 1920x1440 (same 4:3 aspect).
        path = self._write_still(tmp_path, {37386: 12.29})
        derived = focal_from_exif(path, 1920, 1440, sensor_format="4/3")
        assert derived.provenance == "exif"
        assert derived.fx == pytest.approx(12.29 / 17.3 * 1920, abs=0.5)
        assert derived.focal_mm == pytest.approx(12.29)

    def test_35mm_only_exif_is_rejected(self, tmp_path) -> None:
        # The trap: the still carries ONLY FocalLengthIn35mmFilm (24 mm).
        # That value is useless for deriving focal pixels and must raise,
        # never silently produce a focal wrong by the crop factor.
        path = self._write_still(tmp_path, {41989: 24})
        with pytest.raises(Uav3dError, match="FocalLengthIn35mmFilm"):
            focal_from_exif(path, 1920, 1440, sensor_format="4/3")

    def test_no_focal_tag_is_rejected(self, tmp_path) -> None:
        path = self._write_still(tmp_path, {})
        with pytest.raises(Uav3dError, match="no FocalLength EXIF tag"):
            focal_from_exif(path, 1920, 1440, sensor_format="4/3")

    def test_aspect_mismatch_is_rejected(self, tmp_path) -> None:
        # Still is 4:3, video is 16:9: different sensor crops, so the EXIF
        # focal cannot be rescaled safely.
        path = self._write_still(tmp_path, {37386: 12.29}, size=(4000, 3000))
        with pytest.raises(Uav3dError, match="different aspect ratios"):
            focal_from_exif(path, 1920, 1080, sensor_format="4/3")

    def test_focal_plane_tags_supply_sensor_size(self, tmp_path) -> None:
        # FocalPlaneXResolution in pixels/inch: 4000 px still, sensor 17.3 mm
        # wide -> 4000 / (17.3/25.4) = 5872.8 px/inch.
        px_per_inch = 4000 / (17.3 / 25.4)
        path = self._write_still(
            tmp_path, {37386: 12.29, 41486: px_per_inch, 41488: 2}
        )
        derived = focal_from_exif(path, 1920, 1440)
        assert derived.provenance == "exif"
        assert derived.sensor_width_mm == pytest.approx(17.3, abs=0.05)
        assert derived.fx == pytest.approx(12.29 / 17.3 * 1920, abs=1.0)


class TestResolveFocal:
    def test_measured_wins_over_spec(self) -> None:
        spec = FocalSpec(focal_px=265.0, focal_mm=12.29, sensor_format="4/3")
        derived = resolve_focal(spec, 480, 270)
        assert derived.provenance == "measured"
        assert derived.fx == 265.0

    def test_spec_wins_over_guess(self) -> None:
        spec = FocalSpec(focal_mm=12.29, sensor_format="4/3")
        derived = resolve_focal(spec, 5280, 3956)
        assert derived.provenance == "sensor_spec"

    def test_empty_spec_falls_back_to_the_guess(self, caplog) -> None:
        derived = resolve_focal(FocalSpec(), 480, 270)
        assert derived.provenance == "guessed"
        assert derived.fx == derived.fy == 384.0
        assert "falling back to the 0.8 * width guess" in caplog.text

    def test_guess_focal_is_0p8_width(self) -> None:
        derived = guessed_focal_px(1920, 1080)
        assert derived.fx == derived.fy == 1536.0
        assert derived.hfov_deg == pytest.approx(
            math.degrees(2 * math.atan(1920 / (2 * 1536))), abs=0.01
        )


class TestVideoSourceFocalWiring:
    def test_derived_focal_reaches_calibration(self, fake_video_file) -> None:
        # 320x240 fake video; 265 px would fail the 0.3x gate at this width
        # (0.3*320=96, 3*320=960 — 265 is inside), so use a measured 200 px.
        source = VideoFileSource(fake_video_file, focal=FocalSpec(focal_px=200.0))
        calib = source.calibration()
        assert calib.params == [200.0, 200.0, 160.0, 120.0]
        assert calib.provenance == "measured"
        assert calib.source == f"video_measured_{fake_video_file.name}"

    def test_guess_fallback_is_labelled(self, fake_video_file) -> None:
        source = VideoFileSource(fake_video_file)
        calib = source.calibration()
        assert calib.params == [256.0, 256.0, 160.0, 120.0]  # 0.8 * 320
        assert calib.provenance == "guessed"
        assert calib.source.startswith("video_guessed_")

    def test_config_focal_fields_exist(self) -> None:
        cfg = SfmConfig(focal_mm=12.29, sensor_format="4/3")
        assert cfg.focal_mm == 12.29
        assert cfg.sensor_format == "4/3"
        assert cfg.focal_px is None
        assert SfmConfig().focal_mm is None

    def test_config_principal_point_fields_exist(self) -> None:
        # Phase 5 (UAVLight): frames asymmetrically cropped in x, cx is NOT
        # the image centre (357.8 px vs 628.5 px centre on 1257 px wide).
        cfg = SfmConfig(focal_px=1086.06, cx=357.80, cy=470.15)
        assert cfg.cx == 357.80
        assert cfg.cy == 470.15
        assert SfmConfig().cx is None
        assert SfmConfig().cy is None

    def test_ba_global_ratio_fields_default_to_colmap(self) -> None:
        # Phase 5 (KABR): the 1.1 defaults trigger a global BA after nearly
        # every frame on feature-dense scenes; 2.0 is set per-config.
        assert SfmConfig().ba_global_frames_ratio == 1.1
        assert SfmConfig().ba_global_points_ratio == 1.1
        cfg = SfmConfig(ba_global_frames_ratio=2.0, ba_global_points_ratio=2.0)
        assert cfg.ba_global_frames_ratio == 2.0
        assert cfg.ba_global_points_ratio == 2.0

    def test_principal_point_reaches_calibration(self, fake_video_file) -> None:
        source = VideoFileSource(
            fake_video_file,
            focal=FocalSpec(focal_px=200.0, cx=100.0, cy=90.0),
        )
        calib = source.calibration()
        assert calib.params == [200.0, 200.0, 100.0, 90.0]
        assert calib.provenance == "measured"


class TestCalibrationProvenanceRoundTrip:
    def test_provenance_survives_write_read(self, tmp_path, fake_video_file) -> None:
        source = VideoFileSource(fake_video_file, focal=FocalSpec(focal_px=200.0))
        write_calibration(source.calibration(), tmp_path)
        restored = read_calibration(tmp_path)
        assert restored is not None
        assert restored.provenance == "measured"
        payload = json.loads((tmp_path / "calibration.json").read_text())
        assert payload["provenance"] == "measured"

    def test_legacy_file_without_provenance_reads_unknown(self, tmp_path) -> None:
        (tmp_path / "calibration.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "model": "PINHOLE",
                    "width": 640,
                    "height": 480,
                    "params": [512.0, 512.0, 320.0, 240.0],
                    "source": "video_estimate_old.mp4",
                }
            )
        )
        restored = read_calibration(tmp_path)
        assert restored is not None
        assert restored.provenance == "unknown"
