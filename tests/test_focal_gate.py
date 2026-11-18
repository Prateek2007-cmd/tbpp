"""Tests for the focal-drift gate and matcher wiring (Phases 2 and 3).

The gate is the fix for the forward-flight focal/depth ambiguity: with
``refine_intrinsics=True`` COLMAP solved f ~ 1159 px against a 1024 px prior
on the city flyover (+13%), absorbing a depth stretch into the camera, and
nothing in the pipeline flagged it. The gate compares the solved focal with
the prior and reports — it never clamps the focal.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uav3d.config import SfmConfig
from uav3d.exceptions import Uav3dError
from uav3d.sfm.focal_gate import check_focal_drift, focal_drift_rel, solved_focal_px
from uav3d.sfm.runner import match_features


class TestSolvedFocalPx:
    def test_simple_pinhole_uses_first_param(self) -> None:
        assert solved_focal_px("SIMPLE_PINHOLE", [1024.0, 960.0, 540.0]) == 1024.0

    def test_pinhole_averages_fx_fy(self) -> None:
        assert solved_focal_px("PINHOLE", [1000.0, 1020.0, 960.0, 540.0]) == 1010.0

    def test_opencv_averages_fx_fy(self) -> None:
        assert solved_focal_px("OPENCV", [1060.0, 1060.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0]) == 1060.0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported camera model"):
            solved_focal_px("FISHEYE_MAGIC", [1000.0])

    def test_empty_params_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            solved_focal_px("SIMPLE_PINHOLE", [])


class TestFocalDrift:
    def test_city_flyover_drift_is_about_13_percent(self) -> None:
        # The measured case from the handoff: 1024 px prior, 1159 px solved.
        assert focal_drift_rel(1024.0, 1159.0) == pytest.approx(0.1318, abs=1e-3)

    def test_drift_within_gate_passes(self) -> None:
        result = check_focal_drift(1024.0, 1100.0, max_rel_drift=0.10)
        assert result.exceeded is False
        assert result.rel_drift == pytest.approx(0.0742, abs=1e-3)

    def test_drift_beyond_gate_is_flagged(self) -> None:
        result = check_focal_drift(1024.0, 1159.0, max_rel_drift=0.10)
        assert result.exceeded is True
        assert "1159.0" in result.message
        assert "1024.0" in result.message

    def test_negative_drift_is_flagged_by_magnitude(self) -> None:
        result = check_focal_drift(1000.0, 850.0, max_rel_drift=0.10)
        assert result.exceeded is True
        assert result.rel_drift == pytest.approx(-0.15)

    def test_exactly_at_gate_passes(self) -> None:
        result = check_focal_drift(1000.0, 1100.0, max_rel_drift=0.10)
        assert result.exceeded is False

    def test_nonpositive_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="prior focal must be positive"):
            check_focal_drift(0.0, 1000.0)

    def test_negative_max_rel_raises(self) -> None:
        with pytest.raises(ValueError, match="max_rel_drift"):
            check_focal_drift(1000.0, 1000.0, max_rel_drift=-0.1)


class TestSfmConfigFocalGate:
    def test_defaults_are_warn_at_5_percent(self) -> None:
        # Phase 1 tightened the gate from 10% to 5% (Pix4D's green threshold).
        cfg = SfmConfig()
        assert cfg.focal_drift_max_rel == 0.05
        assert cfg.focal_drift_action == "warn"
        assert cfg.matcher_type == "sequential"

    def test_check_focal_drift_default_is_5_percent(self) -> None:
        # The module-level default must match the config default, so callers
        # that omit max_rel_drift (e.g. run_video_pipeline.summarise) get 5%.
        assert check_focal_drift(1000.0, 1040.0).exceeded is False  # 4% passes
        assert check_focal_drift(1000.0, 1060.0).exceeded is True  # 6% flagged
        assert check_focal_drift(1000.0, 1060.0).max_rel_drift == 0.05

    def test_fail_action_is_accepted(self) -> None:
        cfg = SfmConfig(focal_drift_action="fail")
        assert cfg.focal_drift_action == "fail"

    def test_bad_action_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SfmConfig(focal_drift_action="clamp")  # never silently clamp


class TestMatcherWiring:
    def test_sequential_dispatch_calls_sequential_matcher(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "uav3d.sfm.runner.pycolmap.match_sequential",
            lambda *args, **kwargs: calls.append("sequential"),
        )
        assert match_features("db.db", SfmConfig(matcher_type="sequential")) == "sequential"
        assert calls == ["sequential"]

    def test_exhaustive_dispatch_calls_exhaustive_matcher(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "uav3d.sfm.runner.pycolmap.match_exhaustive",
            lambda *args, **kwargs: calls.append("exhaustive"),
        )
        assert match_features("db.db", SfmConfig(matcher_type="exhaustive")) == "exhaustive"
        assert calls == ["exhaustive"]

    def test_unknown_matcher_is_an_error_not_a_silent_fallback(self) -> None:
        with pytest.raises(Uav3dError, match="Unknown matcher_type"):
            match_features("db.db", SfmConfig(matcher_type="vocab_tree"))
