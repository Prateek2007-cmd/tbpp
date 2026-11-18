"""Unit tests for the Phase 4 viewer (uav3d.modeling.dashboard).

The viewer shows the run's REAL measured numbers from its report JSONs, or
"not measured" when a metric is absent. It never hardcodes values and never
invents geometry language (no inpainting/completion claims).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uav3d.dense.glb_export import write_glb
from uav3d.exceptions import DataError
from uav3d.modeling.dashboard import (
    THREE_VERSION,
    collect_viewer_stats,
    find_glb,
    generate_viewer_html,
)


def _tiny_glb(path: Path) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_glb(path, vertices, faces, np.ones((4, 3)) * 0.5)


def _fixture_run(run_dir: Path, with_reports: bool = True) -> None:
    dense_dir = run_dir / "dense"
    dense_dir.mkdir(parents=True)
    _tiny_glb(dense_dir / "mesh.glb")
    if not with_reports:
        return
    (run_dir / "poses.json").write_text(
        json.dumps(
            {
                "dataset_label": "fixture flight",
                "metrics": {
                    "num_images_total": 60,
                    "num_images_registered": 58,
                    "registration_rate": 58 / 60,
                    "mean_reprojection_error_px": 0.281,
                    "num_points3D": 7635,
                },
            }
        ),
        encoding="utf-8",
    )
    (dense_dir / "dense_report.json").write_text(
        json.dumps(
            {
                "dataset_label": "fixture flight",
                "geometry": {"num_registered_views": 60, "num_reference_views": 30},
                "depth": {"valid_fraction": 0.5217},
                "fusion": {"num_points": 404650},
                "mesh": {"num_vertices": 204825, "num_faces": 298119},
                "scale": {
                    "value": 6.664,
                    "source": "gnss",
                    "status": "gnss_aligned",
                    "gnss_frames": 58,
                    "gnss_alignment_rms_m": 0.0505,
                },
                "evaluation": {
                    "accuracy_median": 0.47,
                    "precision_at_1.0m": 0.96,
                    "precision_at_0.5m": 0.54,
                },
                "stage_seconds": {"mvs": 178.9, "fusion": 5.1},
                "total_seconds": 196.5,
            }
        ),
        encoding="utf-8",
    )
    (dense_dir / "coverage_report.json").write_text(
        json.dumps(
            {
                "num_vertices": 204825,
                "fraction_observed_ge_2": 0.912,
                "fraction_observed_ge_3": 0.845,
                "fraction_observed_ge_5": 0.701,
                "summary": "91.2% of the model surface (204,825 vertices) was directly observed by 2 or more views; the rest is unobserved — no geometry is invented there.",
            }
        ),
        encoding="utf-8",
    )
    (dense_dir / "coverage.json").write_text(
        json.dumps({"num_vertices": 4, "counts": [3, 3, 1, 0]}), encoding="utf-8"
    )


def test_viewer_shows_real_numbers(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()
    _fixture_run(run_dir)
    viewer = generate_viewer_html(run_dir)

    assert viewer.exists()
    content = viewer.read_text(encoding="utf-8")
    # Model reference: relative, so the run dir stays portable.
    assert "dense/mesh.glb" in content
    assert str(run_dir) not in content, "no absolute paths in the viewer"
    # Real measured numbers from the fixture reports.
    assert "58/60 (96.7%)" in content
    assert "0.281 px" in content
    assert "7,635" in content
    assert "404,650" in content
    assert "204,825 verts" in content
    assert "52.2%" in content  # depth valid
    assert "0.47 m" in content  # median error
    assert "96.0%" in content  # precision @ 1.0 m
    assert "54.0%" in content  # precision @ 0.5 m
    assert "91.2%" in content  # coverage >= 2 views
    assert "GNSS scale — 6.664 m/unit, 58 GNSS fixes, RMS 0.051 m" in content
    assert "3m 16.5s" in content  # total runtime
    assert "mvs" in content and "fusion" in content  # per-stage runtimes
    assert "no geometry is invented there" in content
    # Coverage toggle is wired when coverage.json exists.
    assert "dense/coverage.json" in content
    # Pinned Three.js, vendored locally — the viewer works fully offline.
    assert THREE_VERSION in content
    assert "./viewer_assets/three.module.js" in content
    assert "./viewer_assets/jsm/" in content
    assert "cdn.jsdelivr.net" not in content
    assert "fonts.googleapis.com" not in content
    assert (run_dir / "viewer_assets" / "three.module.js").exists()
    assert (run_dir / "viewer_assets" / "jsm" / "loaders" / "GLTFLoader.js").exists()
    assert "importmap" in content
    assert "OrbitControls" in content
    # Light theme, no dark-theme leftovers, no hallucination language.
    assert "#f4f2ee" in content
    assert "#0a0d14" not in content
    assert "inpaint" not in content.lower()
    assert "watertight" not in content.lower()
    assert "Occlusion Inpainting" not in content


def test_viewer_missing_metrics_show_not_measured(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_bare"
    run_dir.mkdir()
    _fixture_run(run_dir, with_reports=False)
    viewer = generate_viewer_html(run_dir)

    content = viewer.read_text(encoding="utf-8")
    assert "dense/mesh.glb" in content
    # No reports at all: every metric reads "not measured", nothing is faked.
    assert content.count("not measured") >= 6
    assert "Not measured for this run" in content  # coverage note


def test_viewer_missing_evaluation_only(tmp_path: Path) -> None:
    """No ground truth was scored: accuracy rows say not measured, but the
    SfM numbers that WERE measured still show."""
    run_dir = tmp_path / "run_noeval"
    run_dir.mkdir()
    _fixture_run(run_dir)
    dense_report = json.loads((run_dir / "dense" / "dense_report.json").read_text())
    dense_report["evaluation"] = None
    (run_dir / "dense" / "dense_report.json").write_text(json.dumps(dense_report))

    content = generate_viewer_html(run_dir).read_text(encoding="utf-8")
    assert "58/60 (96.7%)" in content
    assert "0.281 px" in content
    assert "96.0%" not in content
    assert content.count("not measured") >= 3  # median + both precisions


def test_viewer_accuracy_from_run_root_score_file(tmp_path: Path) -> None:
    """Dense evaluation absent, but an external scorer wrote a run-root
    score JSON (the UAVLight case): the measured accuracy shows instead of
    'not measured', with the scoring source named."""
    run_dir = tmp_path / "run_scored"
    run_dir.mkdir()
    _fixture_run(run_dir)
    dense_report = json.loads((run_dir / "dense" / "dense_report.json").read_text())
    dense_report["evaluation"] = None
    (run_dir / "dense" / "dense_report.json").write_text(json.dumps(dense_report))
    (run_dir / "uavlight_score.json").write_text(
        json.dumps(
            {
                "clouds": {
                    "dense_fused": {
                        "precision_at_1m_pct": 84.768,
                        "precision_at_0.5m_pct": 72.799,
                        "median_error_m": 0.2636,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    stats = collect_viewer_stats(run_dir)
    assert stats["precision_at_1m"] == pytest.approx(0.84768)
    assert stats["precision_at_0_5m"] == pytest.approx(0.72799)
    assert stats["median_error_m"] == pytest.approx(0.2636)
    assert stats["accuracy_source"] == (
        "ground-truth scoring (uavlight_score.json, dense_fused cloud)"
    )

    content = generate_viewer_html(run_dir).read_text(encoding="utf-8")
    assert "0.26 m" in content  # median error
    assert "84.8%" in content  # precision @ 1.0 m
    assert "72.8%" in content  # precision @ 0.5 m
    assert "ground-truth scoring (uavlight_score.json, dense_fused cloud)" in content


def test_viewer_explicit_glb(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_city"
    run_dir.mkdir()
    _tiny_glb(run_dir / "city_textured.glb")
    viewer = generate_viewer_html(run_dir, glb_path="city_textured.glb")
    content = viewer.read_text(encoding="utf-8")
    assert "city_textured.glb" in content


def test_viewer_no_glb_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_noglb"
    run_dir.mkdir()
    with pytest.raises(DataError, match="No GLB model"):
        generate_viewer_html(run_dir)
    with pytest.raises(DataError, match="GLB not found"):
        find_glb(run_dir, glb_path="missing.glb")


def test_collect_viewer_stats_missing_files(tmp_path: Path) -> None:
    stats = collect_viewer_stats(tmp_path)
    assert stats["registration"] is None
    assert stats["median_error_m"] is None
    assert stats["precision_at_1m"] is None
    assert stats["coverage_ge2"] is None
    assert stats["has_coverage"] is False
    assert stats["stage_seconds"] == {}
