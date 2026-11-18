"""Tests for georeferencing: ENU/UTM transforms, Sim(3), scale provenance (Phase 2)."""

import json

import numpy as np
import pytest

from uav3d.telemetry.georef import (
    GeorefFrame,
    ScaleProvenance,
    apply_georef_to_vertices,
    load_georef,
    resolve_scale_from_dense_report,
    sim3_apply,
    utm_epsg_for,
    utm_zone_label,
)

# PinPoint flight01 first GPS fix (dataset report §1).
PINPOINT_ORIGIN = (41.7716476, -0.7458537, 324.72)


class TestUtmZone:
    def test_pinpoint_is_zone_30n(self):
        assert utm_epsg_for(*PINPOINT_ORIGIN[:2]) == 32630
        assert utm_zone_label(*PINPOINT_ORIGIN[:2]) == "30N"

    def test_southern_hemisphere(self):
        assert utm_epsg_for(-33.9, 18.4) == 32734
        assert utm_zone_label(-33.9, 18.4) == "34S"


class TestEnuTransform:
    @pytest.fixture
    def frame(self):
        return GeorefFrame.from_first_fix(*PINPOINT_ORIGIN)

    def test_origin_maps_to_zero(self, frame):
        enu = frame.to_enu(*PINPOINT_ORIGIN)
        assert np.allclose(enu, [0.0, 0.0, 0.0], atol=1e-6)

    def test_round_trip(self, frame):
        # A point ~500 m east, ~300 m north, +100 m up of the origin.
        lat, lon, alt = 41.7743476, -0.7398537, 424.72
        enu = frame.to_enu(lat, lon, alt)
        assert enu[0] == pytest.approx(500.0, abs=5.0)
        assert enu[1] == pytest.approx(300.0, abs=5.0)
        assert enu[2] == pytest.approx(100.0, abs=0.5)
        lat2, lon2, alt2 = frame.from_enu(*enu)
        assert lat2 == pytest.approx(lat, abs=1e-7)
        assert lon2 == pytest.approx(lon, abs=1e-7)
        assert alt2 == pytest.approx(alt, abs=1e-3)

    def test_frame_metadata(self, frame):
        assert frame.utm_epsg == 32630
        assert frame.crs_name == "UTM zone 30N (EPSG:32630)"
        d = frame.to_dict()
        assert d["frame"] == "ENU"
        assert d["origin"]["kind"] == "first_gps_fix"
        assert d["origin"]["lat"] == pytest.approx(PINPOINT_ORIGIN[0])
        assert d["crs"] == "EPSG:32630"

    def test_to_utm_sanity(self, frame):
        # 41.77 N is ~4,625 km north of the equator; zone 30N central
        # meridian is 3 W, so -0.745 W sits east of it (easting > 500 km).
        easting, northing, up = frame.to_utm(*PINPOINT_ORIGIN)
        assert 500_000 < easting < 900_000
        assert 4_600_000 < northing < 4_650_000
        assert up == pytest.approx(PINPOINT_ORIGIN[2])


class TestSim3Apply:
    def test_scale_rotation_translation(self):
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rot_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out = sim3_apply(points, scale=2.0, rotation=rot_z_90, translation=np.array([10.0, 0.0, 0.0]))
        # (1,0,0) -> rot -> (0,1,0) -> *2 -> (0,2,0) -> +t -> (10,2,0)
        assert np.allclose(out[0], [10.0, 2.0, 0.0])
        assert np.allclose(out[1], [8.0, 0.0, 0.0])


class TestScaleProvenance:
    def test_gnss_is_metric(self):
        prov = ScaleProvenance(value_m_per_unit=6.76, source="gnss", status="gnss_aligned",
                               gnss_frames=120, gnss_rejected=3, gnss_alignment_rms_m=1.4)
        assert prov.is_metric
        assert prov.units == "metres"
        d = prov.to_dict()
        assert d["scale_m_per_unit"] == 6.76
        assert d["gnss_rejected"] == 3

    def test_none_is_not_metric(self):
        prov = ScaleProvenance(value_m_per_unit=1.0, source="none",
                               status="unverified_monocular_scale")
        assert not prov.is_metric
        assert prov.units == "sfm_units"

    def test_resolve_from_dense_report_gnss(self, tmp_path):
        dense_dir = tmp_path / "dense"
        dense_dir.mkdir()
        report = {
            "scale": {
                "value": 3.25,
                "source": "gnss",
                "status": "gnss_aligned",
                "gnss_frames": 42,
                "gnss_rejected": 1,
                "gnss_alignment_rms_m": 0.9,
            }
        }
        (dense_dir / "dense_report.json").write_text(json.dumps(report))
        prov = resolve_scale_from_dense_report(dense_dir)
        assert prov.value_m_per_unit == 3.25
        assert prov.source == "gnss"
        assert prov.is_metric
        assert prov.gnss_frames == 42
        assert prov.gnss_rejected == 1

    def test_resolve_missing_report_is_honest_none(self, tmp_path):
        prov = resolve_scale_from_dense_report(tmp_path / "nope")
        assert prov.source == "none"
        assert prov.value_m_per_unit == 1.0
        assert not prov.is_metric

    def test_resolve_malformed_report_is_honest_none(self, tmp_path):
        dense_dir = tmp_path / "dense"
        dense_dir.mkdir()
        (dense_dir / "dense_report.json").write_text("{not json")
        prov = resolve_scale_from_dense_report(dense_dir)
        assert prov.source == "none"


class TestLoadGeoref:
    def _georef(self):
        return {
            "frame": "ENU",
            "origin": {"lat": 41.77, "lon": -0.74, "alt_m": 320.0, "kind": "first_gps_fix"},
            "crs": "EPSG:32630",
            "crs_name": "UTM zone 30N (EPSG:32630)",
            "sim3": {
                "scale_m_per_unit": 2.0,
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation_m": [100.0, 200.0, 5.0],
            },
        }

    def test_load_present(self, tmp_path):
        dense_dir = tmp_path / "dense"
        dense_dir.mkdir()
        (dense_dir / "georef.json").write_text(json.dumps(self._georef()))
        data = load_georef(dense_dir)
        assert data is not None
        assert data["crs"] == "EPSG:32630"

    def test_load_missing_is_none(self, tmp_path):
        assert load_georef(tmp_path) is None

    def test_load_without_sim3_is_none(self, tmp_path):
        dense_dir = tmp_path / "dense"
        dense_dir.mkdir()
        (dense_dir / "georef.json").write_text(json.dumps({"frame": "ENU"}))
        assert load_georef(dense_dir) is None

    def test_apply_georef_does_not_double_scale(self):
        # Vertices are ALREADY metres (dense scaled them); the georef
        # transform applies rotation + translation only, never the scale.
        georef = self._georef()
        verts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = apply_georef_to_vertices(verts, georef)
        # Identity rotation, translation (100, 200, 5); scale 2.0 ignored.
        assert np.allclose(out[0], [101.0, 202.0, 8.0])
        assert np.allclose(out[1], [104.0, 205.0, 11.0])
        # Bbox span is preserved (no rescaling).
        assert np.allclose(out.max(0) - out.min(0), [3.0, 3.0, 3.0])
