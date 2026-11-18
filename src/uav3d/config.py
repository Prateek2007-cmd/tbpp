"""Configuration management for uav3d pipeline."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uav3d.dense.config import DenseConfig
from uav3d.exceptions import ConfigError


class IngestConfig(BaseModel):
    """Configuration for dataset ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(default="zurich", description="Ingestion source type: zurich or video")
    start_frame: int = Field(default=0, ge=0, description="Start index for processing")
    max_frames: int = Field(default=300, gt=0, description="Maximum contiguous frames to ingest")


class KeyframeConfig(BaseModel):
    """Configuration for keyframe selection."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="laplacian_variance", description="Selection algorithm")
    stride: int = Field(default=2, ge=1, description="Temporal subsampling stride")
    max_frames: int = Field(default=250, gt=0, description="Maximum number of keyframes to select")
    min_blur: float = Field(default=60.0, ge=0.0, description="Minimum Laplacian variance blur score threshold")
    max_clipping: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum allowed pixel clipping ratio")
    max_gap: int = Field(default=15, ge=1, description="Maximum allowed source frame index gap between consecutive keyframes")
    orb_features: int = Field(default=2000, gt=0, description="Maximum ORB feature count cap")
    sharpness_threshold: float | None = Field(default=None, description="Optional alias for min_blur")
    target_count: int | None = Field(default=None, description="Optional alias for max_frames")


class SfmConfig(BaseModel):
    """Configuration for visual Structure-from-Motion."""

    model_config = ConfigDict(extra="forbid")

    camera_model: str = Field(default="OPENCV", description="COLMAP camera model")
    single_camera: bool = Field(default=True, description="Share a single camera intrinsic model across all frames")
    seq_overlap: int = Field(default=10, ge=1, description="Sequential matching overlap window")
    max_image_size: int = Field(default=1600, gt=0, description="Maximum image dimension for feature extraction")
    refine_intrinsics: bool = Field(default=False, description="Whether to refine camera intrinsics during bundle adjustment")
    focal_drift_max_rel: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Largest absolute relative focal drift (solved vs prior focal) that passes "
            "the focal-drift gate. Forward flight trades focal length against depth, "
            "so a solved focal far from the prior usually means a depth stretch was "
            "absorbed into the camera. 5% is Pix4D's green threshold. The gate only "
            "flags it; it never clamps the focal."
        ),
    )
    # --- Focal length for video runs (Phase 1). Video files carry no EXIF,
    # so the focal must come from one of these, in priority order:
    # focal_px (measured pre-flight calibration, best) > exif_image (a still
    # captured with the same camera/settings) > focal_mm + sensor dims/spec
    # > 0.8 * width guess (last resort, logged as a warning). See
    # uav3d/calibration/focal.py and docs/CAPTURE_SOP.md.
    focal_px: float | None = Field(
        default=None,
        gt=0.0,
        description="Measured focal length in pixels from a pre-flight calibration capture (cv2.calibrateCamera). Highest-priority focal source.",
    )
    focal_mm: float | None = Field(
        default=None,
        gt=0.0,
        description="Real lens focal length in millimetres (from the spec sheet or EXIF FocalLength). NEVER the 35mm-equivalent focal length.",
    )
    sensor_width_mm: float | None = Field(
        default=None,
        gt=0.0,
        description="Sensor width in millimetres for the video mode (full-frame width unless the video is a crop). Used with focal_mm.",
    )
    sensor_height_mm: float | None = Field(
        default=None,
        gt=0.0,
        description="Sensor height in millimetres for the video mode. Omit to assume square pixels (fy = fx).",
    )
    sensor_format: str | None = Field(
        default=None,
        description="Sensor format key looked up in the built-in table ('4/3', '1-inch', '1/1.3-inch'). Explicit sensor_width_mm wins over the table.",
    )
    exif_image: str | None = Field(
        default=None,
        description="Path to a still image (JPEG) captured with the same camera, zoom and focus as the flight; its EXIF FocalLength is used with the sensor size.",
    )
    cx: float | None = Field(
        default=None,
        description="Principal-point x in pixels (default: image centre). Set from a calibration capture / cameras.txt when the frames are asymmetrically cropped — UAVLight's cx is ~358 px vs a 628 px centre, and assuming the centre warps the reconstruction.",
    )
    cy: float | None = Field(
        default=None,
        description="Principal-point y in pixels (default: image centre). See cx.",
    )
    focal_drift_action: Literal["warn", "fail"] = Field(
        default="warn",
        description=(
            "What the focal-drift gate does when exceeded: 'warn' records the drift "
            "and continues, 'fail' aborts the SfM run."
        ),
    )
    feature_type: str = Field(default="SIFT", description="Feature extractor type")
    matcher_type: str = Field(
        default="sequential",
        description=(
            "COLMAP matcher type: 'sequential' (each frame matched to its temporal "
            "neighbours — fast, but drift accumulates over a long pass with no loop "
            "closure) or 'exhaustive' (every pair matched, which closes loops when "
            "the flight revisits an area, at O(n^2) matching cost)."
        ),
    )
    min_num_matches: int = Field(default=15, ge=1, description="Minimum inlier matches for pair")
    init_min_inliers: int = Field(default=15, ge=1, description="Minimum inliers for initial pair")
    init_min_tri_angle: float = Field(default=2.0, ge=0.0, description="Minimum triangulation angle for initial pair in degrees")
    abs_pose_min_inliers: int = Field(default=10, ge=1, description="Minimum inliers for absolute pose registration")
    min_model_size: int = Field(default=3, ge=2, description="Minimum registered images to keep reconstruction")
    ignore_watermarks: bool = Field(default=True, description="Whether to ignore watermark detection in sequential matching")
    # --- Global bundle-adjustment cadence (Phase 5). COLMAP runs a global BA
    # whenever the registered-frame count OR the point count grows by these
    # ratios since the last global BA (defaults 1.1 = COLMAP default). On a
    # feature-dense scene (1080p vegetation) every new frame triangulates
    # >10% new points, so the default triggers a 3-4 min global BA after
    # EVERY frame — a 120-frame run then needs hours (measured on KABR
    # DJI_0679, 2026-11-17: 51/120 in 26 min, ~4 min/frame). Raising the
    # ratios to 2.0 runs global BA when the model doubles (~7 BAs for 120
    # frames) plus the final global BA COLMAP always runs; local BA still
    # runs after every frame. Accuracy is verified after the fact by the
    # GPS-trajectory residual, not assumed. Farmland (few points/frame)
    # never triggers the points ratio, so the default is fine there.
    ba_global_frames_ratio: float = Field(
        default=1.1,
        gt=1.0,
        description="Run global BA when registered frames grow by this ratio since the last global BA (COLMAP default 1.1).",
    )
    ba_global_points_ratio: float = Field(
        default=1.1,
        gt=1.0,
        description="Run global BA when triangulated points grow by this ratio since the last global BA (COLMAP default 1.1).",
    )


class SyncConfig(BaseModel):
    """Configuration for telemetry time synchronization."""

    model_config = ConfigDict(extra="forbid")

    max_delta_us: int = Field(default=500000, ge=0, description="Max tolerance delta in microseconds (default 500ms)")
    strategy: str = Field(default="nearest", description="Sync strategy (nearest or linear)")
    estimate_offset: bool = Field(default=True, description="Whether to estimate constant clock offset for Mode B")
    min_match_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum match rate required for synchronized status")
    dataflash_sync_offset_us: int | None = Field(
        default=None,
        description=(
            "Coarse video↔log sync offset for ArduPilot DataFlash (.BIN) telemetry, "
            "in microseconds. DataFlash TimeUS is autopilot boot-relative, so the "
            "parser lands telemetry in the video-PTS basis as "
            "(TimeUS - first_fix) - this offset, where this offset is the "
            "video-start-minus-first-fix time (positive when the video starts "
            "after the first fix). The offset is PER-FLIGHT "
            "(PinPoint flight01: 1.2 s, flight02: 49.4 s) — there is deliberately "
            "no default. When None, the parser reads the flight's own "
            "campaign.json (config.sync.offset_s, next to the .BIN); when "
            "neither is available the parse FAILS LOUDLY instead of assuming. "
            "Mode B's offset estimator only searches within max_delta_us and "
            "cannot find an offset of seconds, so this coarse alignment must "
            "happen in the parser."
        ),
    )
    xlsx_sync_offset_us: int | None = Field(
        default=None,
        description=(
            "Residual video<->log sync offset for DJI AirData (.xlsx) "
            "telemetry, in microseconds. The xlsx parser already anchors the "
            "first fix to video t=0 (DJI0016: xlsx t=84400 ms <-> video t=0), "
            "so this is only the fine-scan residual correction (default None "
            "= 0, no correction). Pass the Stage 0 fine-scan minimum here "
            "when one is found; the sign follows the dataflash convention "
            "(timestamp_us = (time_ms - first_fix_ms) * 1000 - offset)."
        ),
    )


class ValidationConfig(BaseModel):
    """Configuration for trajectory evaluation and ATE validation."""

    model_config = ConfigDict(extra="forbid")

    alignment: str = Field(default="sim3", description="Trajectory alignment type: sim3, se3, or none")
    ate_threshold_meters: float = Field(default=5.0, gt=0.0, description="Absolute Trajectory Error warning threshold")


class SourceConfig(BaseModel):
    """Configuration for image sequence or video source."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="zurich", description="Source type: 'zurich' or 'video'")
    dataset_dir: str = Field(default="data/raw/zurich_mav", description="Path to Zurich dataset root")
    video_path: str | None = Field(default=None, description="Path to video file if type is 'video'")
    start_id: int = Field(default=0, ge=0, description="Starting numeric image/frame identifier")
    count: int = Field(default=250, gt=0, le=300, description="Contiguous frame count (<= 300)")
    stride: int = Field(default=1, ge=1, description="Frame sampling stride")


class PipelineConfig(BaseModel):
    """Root configuration for uav3d pipeline."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(default="./data", description="Root data directory")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    source: SourceConfig = Field(default_factory=SourceConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    keyframes: KeyframeConfig = Field(default_factory=KeyframeConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    dense: DenseConfig = Field(default_factory=DenseConfig)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Fails fast with ConfigError if file is missing, contains invalid YAML,
    or contains unrecognized/invalid configuration keys (extra='forbid').
    """
    raw_data: dict[str, Any] = {}

    if path is not None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise ConfigError(f"Configuration file not found: {path_obj}")
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    if not isinstance(loaded, dict):
                        raise ConfigError(f"Configuration in {path_obj} must be a dictionary/mapping, got {type(loaded).__name__}")
                    raw_data = loaded
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML configuration in {path_obj}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"IO error reading configuration file {path_obj}: {exc}") from exc

    # Honor UAV3D_DATA_DIR environment variable if data_dir not explicitly set in config
    env_data_dir = os.environ.get("UAV3D_DATA_DIR")
    if env_data_dir and "data_dir" not in raw_data:
        raw_data["data_dir"] = env_data_dir

    try:
        return PipelineConfig(**raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc
