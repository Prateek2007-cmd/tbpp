"""Tests for GPS-constrained (pose-prior) bundle adjustment.

Covers the wrapper in ``uav3d.sfm.pose_prior``: holdout selection, ENU
conversion, prior construction + covariance guards, and the full
``apply_pose_prior_ba`` flow on a small synthetic reconstruction built
from COLMAP text files (20 cameras on a straight strip, SfM units 2x
metres, so the before/after scale 0.5 -> ~1.0 exercises the
georeferencing). The synthetic BA runs in seconds; nothing here touches
a real run directory or the network.
"""

import json
import math
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from uav3d.exceptions import Uav3dError
from uav3d.sfm.pose_prior import (
    apply_pose_prior_ba,
    build_pose_priors,
    latlon_to_enu_colmap,
    select_holdout_indices,
)

# Synthetic strip geometry: cameras every 10 SfM units, GPS fixes every
# 5 m east, so the free-SfM scale is 0.5 m/unit and a correct prior-BA
# brings it to ~1.0.
N_IMAGES = 20
CAM_SPACING_UNITS = 10.0
GPS_SPACING_M = 5.0
ORIGIN_LAT = 41.93783433
ORIGIN_LON = -8.79247333
ORIGIN_ALT = 52.0
FOCAL_PX = 2560.0


class TestHoldoutSelection:
    def test_default_10_percent_on_120_images(self) -> None:
        holdout = select_holdout_indices(120, 0.10)
        assert holdout == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119]
        assert len(holdout) == 12

    def test_spread_along_strip_not_clustered(self) -> None:
        holdout = select_holdout_indices(34, 0.10)
        assert holdout == [9, 19, 29]
        # First and last thirds both represented.
        assert holdout[0] < 34 // 3
        assert holdout[-1] > 2 * 34 // 3

    def test_zero_fraction_holds_out_nothing(self) -> None:
        assert select_holdout_indices(20, 0.0) == []

    def test_quarter_fraction(self) -> None:
        assert select_holdout_indices(20, 0.25) == [3, 7, 11, 15, 19]

    def test_invalid_fraction_rejected(self) -> None:
        with pytest.raises(Uav3dError):
            select_holdout_indices(20, 0.5)
        with pytest.raises(Uav3dError):
            select_holdout_indices(20, -0.1)


class TestEnuConversion:
    def test_origin_maps_to_zero(self) -> None:
        lla = np.array([[ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT]])
        enu = latlon_to_enu_colmap(lla, ORIGIN_LAT, ORIGIN_LON)
        assert enu.shape == (1, 3)
        assert np.allclose(enu[0], [0.0, 0.0, 0.0], atol=1e-9)

    def test_east_north_up_scales(self) -> None:
        # 0.001 deg east at lat ~42N is ~82.7 m; 0.001 deg north ~111.2 m.
        lla = np.array(
            [
                [ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT],
                [ORIGIN_LAT, ORIGIN_LON + 0.001, ORIGIN_ALT],
                [ORIGIN_LAT + 0.001, ORIGIN_LON, ORIGIN_ALT],
                [ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT + 3.0],
            ]
        )
        enu = latlon_to_enu_colmap(lla, ORIGIN_LAT, ORIGIN_LON)
        east_1e3 = math.radians(0.001) * 6378137.0 * math.cos(math.radians(ORIGIN_LAT))
        north_1e3 = math.radians(0.001) * 6378137.0
        assert enu[1, 0] == pytest.approx(east_1e3, rel=0.01)
        assert abs(enu[1, 1]) < 0.01
        assert enu[2, 1] == pytest.approx(north_1e3, rel=0.01)
        assert abs(enu[2, 0]) < 0.01
        assert enu[3, 2] == pytest.approx(3.0, abs=1e-6)

    def test_monotonic_in_lon(self) -> None:
        lons = ORIGIN_LON + np.linspace(0, 0.001, 5)
        lla = np.column_stack(
            [np.full(5, ORIGIN_LAT), lons, np.full(5, ORIGIN_ALT)]
        )
        enu = latlon_to_enu_colmap(lla, ORIGIN_LAT, ORIGIN_LON)
        assert np.all(np.diff(enu[:, 0]) > 0)


class TestBuildPosePriors:
    def test_priors_are_cartesian_with_isotropic_covariance(self) -> None:
        enu = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        priors = build_pose_priors(enu, [1, 2, 3], sigma_m=3.0)
        assert len(priors) == 3
        for k, image_id in enumerate([1, 2, 3]):
            prior = priors[image_id]
            assert np.allclose(prior.position, enu[k])
            assert np.allclose(prior.position_covariance, np.eye(3) * 9.0)
            assert (
                prior.coordinate_system
                == pycolmap.PosePriorCoordinateSystem.CARTESIAN
            )
            assert prior.is_valid()
            assert prior.is_covariance_valid()

    def test_holdout_excluded_from_priors(self) -> None:
        enu = np.zeros((10, 3))
        enu[:, 0] = np.arange(10) * 5.0
        priors = build_pose_priors(
            enu, list(range(1, 11)), sigma_m=1.0, holdout_indices={9}
        )
        assert len(priors) == 9
        assert 10 not in priors  # image_id 10 is index 9

    def test_nonpositive_sigma_rejected(self) -> None:
        with pytest.raises(Uav3dError):
            build_pose_priors(np.zeros((3, 3)), [1, 2, 3], sigma_m=0.0)
        with pytest.raises(Uav3dError):
            build_pose_priors(np.zeros((3, 3)), [1, 2, 3], sigma_m=-1.0)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(Uav3dError):
            build_pose_priors(np.zeros((3, 3)), [1, 2], sigma_m=1.0)
        with pytest.raises(Uav3dError):
            build_pose_priors(np.zeros((3, 2)), [1, 2, 3], sigma_m=1.0)

    def test_nonfinite_position_rejected(self) -> None:
        enu = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [10.0, 0.0, 0.0]])
        with pytest.raises(Uav3dError):
            build_pose_priors(enu, [1, 2, 3], sigma_m=1.0)

    def test_fewer_than_three_priors_rejected(self) -> None:
        # The initial RANSAC Sim(3) alignment needs >= 3 priors.
        enu = np.zeros((3, 3))
        with pytest.raises(Uav3dError):
            build_pose_priors(enu, [1, 2, 3], sigma_m=1.0, holdout_indices={1, 2})


def _wiggle_m(i: int) -> tuple[float, float]:
    """Lateral/vertical wiggle (metres) of camera i, relative to camera 0.

    The strip must NOT be perfectly collinear: COLMAP's Sim(3) estimator
    returns no model for a collinear 3-point sample, so a perfectly
    straight synthetic strip makes the adjuster's initial alignment fail
    and the priors silently drop out (verified). Real flights always have
    metres of lateral/vertical extent; the wiggle reproduces that.
    """
    north = 2.0 * math.sin(i * 0.7) - 2.0 * math.sin(0.0)
    up = 1.5 * math.sin(i * 0.5 + 1.0) - 1.5 * math.sin(1.0)
    return north, up


def _build_synthetic_run_dir(run_dir: Path) -> None:
    """Create a minimal run dir: sparse model + telemetry + keyframes.

    20 cameras on a near-straight strip (with a lateral/vertical wiggle so
    the Sim(3) alignment is non-degenerate), 10 SfM units apart, looking
    along +z; 60 points spread along the strip ahead of the cameras. GPS
    fixes are 5 m apart east, so the free-SfM scale is 0.5 m/unit.
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(
        low=[-10.0, -10.0, 20.0], high=[210.0, 10.0, 60.0], size=(60, 3)
    )
    f, cx, cy = FOCAL_PX, 1920.0, 1080.0
    # Camera centres in SfM units: 2x the metric GPS positions (scale 0.5).
    centres = []
    for i in range(N_IMAGES):
        north_m, up_m = _wiggle_m(i)
        centres.append(
            np.array([i * CAM_SPACING_UNITS, 2.0 * north_m, 2.0 * up_m])
        )
    obs: dict[int, list[tuple[int, np.ndarray]]] = {j: [] for j in range(60)}
    for j, X in enumerate(pts):
        for i in range(N_IMAGES):
            p_cam = X - centres[i]
            if p_cam[2] <= 1.0:
                continue
            uv = np.array(
                [f * p_cam[0] / p_cam[2] + cx, f * p_cam[1] / p_cam[2] + cy]
            )
            if not (0 <= uv[0] < 3840 and 0 <= uv[1] < 2160):
                continue
            obs[j].append((i + 1, uv))
    keep = {j: v for j, v in obs.items() if len(v) >= 2}
    assert len(keep) >= 30, f"too few triangulated points: {len(keep)}"

    img_pts: dict[int, list[tuple[np.ndarray, int]]] = {
        i: [] for i in range(1, N_IMAGES + 1)
    }
    for j, v in keep.items():
        for img_id, uv in v:
            img_pts[img_id].append((uv, j + 1))

    txt_dir = run_dir / "colmap" / "sparse" / "txt"
    txt_dir.mkdir(parents=True)
    (txt_dir / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 SIMPLE_PINHOLE 3840 2160 {FOCAL_PX} 1920.0 1080.0\n",
        encoding="utf-8",
    )
    lines = ["# Image list"]
    for i in range(1, N_IMAGES + 1):
        # Identity rotation, t = -C.
        c = centres[i - 1]
        lines.append(
            f"{i} 1 0 0 0 {-c[0]:.4f} {-c[1]:.4f} {-c[2]:.4f} 1 "
            f"video_{i - 1:06d}.jpg"
        )
        p2d = " ".join(
            f"{uv[0]:.2f} {uv[1]:.2f} {pid}" for uv, pid in img_pts[i]
        )
        lines.append(p2d)
    (txt_dir / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plines = ["# 3D point list"]
    for j, v in keep.items():
        X = pts[j]
        elems = []
        for img_id, _uv in v:
            idx = [k for k, (_, pid) in enumerate(img_pts[img_id]) if pid == j + 1][0]
            elems.append(f"{img_id} {idx}")
        plines.append(
            f"{j + 1} {X[0]:.4f} {X[1]:.4f} {X[2]:.4f} 128 128 128 0.5 "
            + " ".join(elems)
        )
    (txt_dir / "points3D.txt").write_text(
        "\n".join(plines) + "\n", encoding="utf-8"
    )
    rec = pycolmap.Reconstruction(str(txt_dir))
    assert rec.num_reg_images() == N_IMAGES
    sparse_dir = run_dir / "colmap" / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    rec.write(str(sparse_dir))
    # The text staging dir must not linger: dense takes the FIRST sorted
    # sparse subdir, and the wrapper must not be confused either.
    import shutil

    shutil.rmtree(txt_dir)

    # Telemetry: fixes 5 m apart east + the same wiggle, constant base
    # altitude. Lat/lon from ENU by the spherical approximation (the
    # wrapper converts back with COLMAP's ellipsoid; the ~0.2% scale
    # difference is absorbed by the Sim(3) fit and is far below the
    # test's 0.5 m assertions).
    deg_per_m = 1.0 / 111319.49
    deg_per_m_east = deg_per_m / math.cos(math.radians(ORIGIN_LAT))
    records = []
    for i in range(N_IMAGES):
        north_m, up_m = _wiggle_m(i)
        records.append(
            {
                "timestamp_us": i * 450_000,
                "frame_id": f"video_{i:06d}",
                "lat": ORIGIN_LAT + north_m * deg_per_m,
                "lon": ORIGIN_LON + i * GPS_SPACING_M * deg_per_m_east,
                "alt": ORIGIN_ALT + up_m,
                "source": "synthetic",
            }
        )
    (run_dir / "telemetry.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic pose-prior test",
                "sync_status": "timestamp_nearest_offset_estimated",
                "offset_us": 0,
                "summary": {
                    "matched_telemetry_frames": N_IMAGES,
                    "total_frames": N_IMAGES,
                    "match_rate": 1.0,
                },
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    keyframes = [
        {
            "frame_id": f"video_{i:06d}",
            "timestamp_us": i * 450_000,
            "file_path": f"sfm/images/video_{i:06d}.jpg",
            "score": 100.0,
            "selected": True,
            "selection_reason": "synthetic",
        }
        for i in range(N_IMAGES)
    ]
    (run_dir / "keyframes.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic pose-prior test",
                "keyframes": keyframes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Pre-BA poses.json + poses_enriched.json + a stale npz cache, so the
    # wrapper's refreshes have something to refresh.
    from uav3d.models.frames import CameraCalibration
    from uav3d.sfm.colmap_export import export_poses

    calibration = CameraCalibration(
        model="SIMPLE_PINHOLE",
        width=3840,
        height=2160,
        params=[FOCAL_PX, 1920.0, 1080.0],
        source="synthetic",
        provenance="sensor_spec",
    )
    export_poses(
        reconstruction=rec,
        all_image_names=[f"video_{i:06d}.jpg" for i in range(N_IMAGES)],
        camera_calibration=calibration,
        output_path=run_dir / "poses.json",
        dataset_label="synthetic pose-prior test",
        timestamps_map={f"video_{i:06d}": i * 450_000 for i in range(N_IMAGES)},
    )
    enriched = json.loads((run_dir / "poses.json").read_text(encoding="utf-8"))
    for pose in enriched["poses"]:
        pose["telemetry_recorded"] = True
    (run_dir / "poses_enriched.json").write_text(
        json.dumps(enriched, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "sparse_points.npz",
        points=np.zeros((10, 3), dtype=np.float32),
    )


class TestApplyPosePriorBa:
    def test_full_flow_georeferences_and_refreshes(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)

        report = apply_pose_prior_ba(run_dir, sigma_m=3.0, holdout_fraction=0.10)

        # Report fields.
        assert report.sigma_m == 3.0
        assert report.n_images == N_IMAGES
        assert report.n_priors == N_IMAGES - 2  # every 10th held out
        assert report.n_holdout == 2
        assert report.holdout_frame_ids == ["video_000009", "video_000019"]
        assert report.focal_unchanged is True
        assert report.max_num_iterations == 250
        assert report.robust_loss_on_prior_position is True

        # Before: free SfM at 0.5 m/unit scale, ~zero Sim(3) residual
        # (the synthetic geometry is exact up to scale).
        before = report.before
        assert before["gps_sim3_colmap_enu_m"]["scale_factor"] == pytest.approx(
            0.5, rel=0.02
        )
        assert before["gps_sim3_colmap_enu_m"]["median_m"] < 0.05
        assert before["focal_params"]["1"][0] == pytest.approx(FOCAL_PX)

        # After: metric scale ~1.0, raw ENU residual near zero (GPS is
        # exactly consistent with the synthetic geometry).
        after = report.after
        assert after["gps_sim3_colmap_enu_m"]["scale_factor"] == pytest.approx(
            1.0, rel=0.02
        )
        assert after["raw_enu_colmap_m"]["median_m"] < 0.5
        assert after["focal_params"]["1"][0] == pytest.approx(FOCAL_PX)
        assert after["held_out"]["n"] == 2
        assert after["held_out"]["raw_median_m"] < 0.5
        assert len(after["per_image_sim3_m"]) == N_IMAGES
        assert len(after["per_image_raw_enu_m"]) == N_IMAGES

        # Solver guards: residual count proves the priors were in the
        # Ceres problem (2 per observation + 3 per prior).
        assert (
            report.solver["num_residuals"]
            == report.solver["expected_residuals"]
            == 2 * before["n_obs"] + 3 * report.n_priors
        )
        assert report.solver["termination_type"] in (
            "CONVERGENCE",
            "NO_CONVERGENCE",
        )
        assert report.wall_time_s["total_s"] >= 0  # synthetic run is sub-second

        # Backup + model rewrite.
        backup = run_dir / "colmap" / "sparse" / "0-pre-priorba"
        assert backup.exists()
        assert (backup / "cameras.bin").exists()
        assert (run_dir / "colmap" / "sparse" / "0" / "cameras.bin").exists()
        rec_after = pycolmap.Reconstruction(
            str(run_dir / "colmap" / "sparse" / "0")
        )
        assert rec_after.num_reg_images() == N_IMAGES
        # First camera centre is now at the ENU origin (within a metre).
        assert np.linalg.norm(rec_after.images[1].projection_center()) < 1.0

        # Downstream refreshes.
        assert report.refreshed["poses_json"] == "re-exported from adjusted model"
        assert not (run_dir / "sparse_points.npz").exists()
        assert "rebuilt from post-BA poses" in report.refreshed["poses_enriched_json"]
        enriched = json.loads(
            (run_dir / "poses_enriched.json").read_text(encoding="utf-8")
        )
        assert all(p["telemetry_recorded"] for p in enriched["poses"])

        # Report file on disk matches the returned report.
        on_disk = json.loads(
            (run_dir / "pose_prior_ba.json").read_text(encoding="utf-8")
        )
        assert on_disk["sigma_m"] == 3.0
        assert on_disk["n_priors"] == report.n_priors
        assert on_disk["after"]["raw_enu_colmap_m"] == after["raw_enu_colmap_m"]

    def test_second_run_refuses_to_overwrite_backup(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        apply_pose_prior_ba(run_dir)
        with pytest.raises(Uav3dError, match="already exists"):
            apply_pose_prior_ba(run_dir)

    def test_missing_telemetry_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        (run_dir / "telemetry.json").unlink()
        with pytest.raises(Uav3dError, match="telemetry.json"):
            apply_pose_prior_ba(run_dir)

    def test_missing_sparse_model_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        import shutil

        shutil.rmtree(run_dir / "colmap" / "sparse" / "0")
        with pytest.raises(Uav3dError, match="Sparse model not found"):
            apply_pose_prior_ba(run_dir)

    def test_frame_without_fix_is_skipped_not_failed(self, tmp_path: Path) -> None:
        # Frames without a fix (e.g. hover frames before the first fix
        # when a sync offset applies) carry no prior and are not scored.
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        tel = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
        tel["records"] = [r for r in tel["records"] if r["frame_id"] != "video_000005"]
        (run_dir / "telemetry.json").write_text(
            json.dumps(tel, indent=2), encoding="utf-8"
        )
        report = apply_pose_prior_ba(run_dir)
        assert report.skipped_frame_ids == ["video_000005"]
        assert report.n_skipped == 1
        assert report.n_registered == N_IMAGES
        assert report.n_images == N_IMAGES - 1
        assert report.n_priors == N_IMAGES - 1 - report.n_holdout

    def test_too_few_fixes_raises(self, tmp_path: Path) -> None:
        # Fewer than 10 frames with fixes cannot be scored (Sim(3)
        # minimum) — fail loudly instead of producing a meaningless run.
        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        tel = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
        tel["records"] = tel["records"][:9]
        (run_dir / "telemetry.json").write_text(
            json.dumps(tel, indent=2), encoding="utf-8"
        )
        with pytest.raises(Uav3dError, match="Only 9 registered frames"):
            apply_pose_prior_ba(run_dir)


class TestPosePriorBaCli:
    def test_help_exits_zero(self) -> None:
        from uav3d.cli import main

        assert main(["pose-prior-ba", "--help"]) == 0

    def test_subcommand_runs_on_synthetic_run(self, tmp_path: Path) -> None:
        from uav3d.cli import main

        run_dir = tmp_path / "run_synth"
        run_dir.mkdir()
        _build_synthetic_run_dir(run_dir)
        code = main(
            [
                "pose-prior-ba",
                "--run-dir",
                str(run_dir),
                "--sigma",
                "3.0",
                "--holdout-fraction",
                "0.10",
            ]
        )
        assert code == 0
        assert (run_dir / "pose_prior_ba.json").exists()
