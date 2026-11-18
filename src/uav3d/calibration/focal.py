"""Focal-length derivation from measured calibration, EXIF, or sensor spec.

Why this module exists: on single-pass forward flight the focal length is
the accuracy-limiting parameter (docs/HANDOFF_NEXT_SESSION.md §12). Free
bundle-adjustment refinement is unstable — it solved 447 px once and 492 px
the next run on identical synthetic data (true focal 265 px) — and locking
the wrong prior is just as bad (the 0.8*width guess, 384 px, gave a 3.84 m
median error), while reprojection error stayed ~0.28 px throughout. That is
the bas-relief trap: focal error is absorbed into depth and is invisible to
reprojection error. The only fix is a real focal length.

Provenance ladder, best to worst (recorded on every run, never silently
upgraded):

- ``measured``: focal in pixels from a pre-flight calibration capture
  (checkerboard / AprilGrid + ``cv2.calibrateCamera``) at the exact
  camera / zoom / focus that will fly. See docs/CAPTURE_SOP.md.
- ``exif``: focal length in millimetres read from a still image's EXIF
  (same camera and settings as the flight) plus sensor dimensions.
  Video files carry no per-frame EXIF, so this needs a stills capture.
- ``sensor_spec``: focal length in millimetres plus sensor dimensions
  from the camera spec sheet.
- ``guessed``: 0.8 * image width. Last resort only, always logged as a
  warning. A run built on a guessed focal must never be presented as
  metrically accurate.

Traps this module refuses to fall into:

- ``FocalLengthIn35mmFilm`` (EXIF tag 41989) is NEVER used: it is the
  focal length rescaled to a 36x24 mm frame, not the real lens focal
  length, and feeding it into ``fx = f_mm / sensor_width_mm * W`` gives
  a focal wrong by the crop factor.
- A |fx - fy| mismatch above 2% is rejected: square sensor pixels mean
  fx and fy must agree, and a mismatch almost always means the sensor
  dimensions belong to a different crop/mode than the video (e.g. a 3:2
  sensor shooting 16:9 video — the *effective* sensor height for the
  video crop must be used, not the full-frame height).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from uav3d.exceptions import Uav3dError

logger = logging.getLogger("uav3d.calibration.focal")

FocalProvenance = Literal["measured", "exif", "sensor_spec", "guessed"]

# Bounds for the plausibility gate. A focal outside 0.3x-3x the image width,
# or a horizontal field of view outside 20-120 degrees, is not a drone camera
# — it is a wrong sensor size, a 35mm-equivalent focal fed in as real, or a
# units mistake. Reject loudly instead of reconstructing garbage.
FOCAL_MIN_REL_WIDTH = 0.3
FOCAL_MAX_REL_WIDTH = 3.0
HFOV_MIN_DEG = 20.0
HFOV_MAX_DEG = 120.0
FX_FY_MAX_REL_DIFF = 0.02

# EXIF tags (see EXIF 2.32). FocalLength is the real lens focal length in mm.
# FocalLengthIn35mmFilm is deliberately NOT read — see the module docstring.
_EXIF_FOCAL_LENGTH = 37386
_EXIF_FOCAL_LENGTH_35MM = 41989
_EXIF_FOCAL_PLANE_X_RES = 41486
_EXIF_FOCAL_PLANE_Y_RES = 41487
_EXIF_FOCAL_PLANE_RES_UNIT = 41488


@dataclass(frozen=True)
class SensorFormat:
    """Physical dimensions of a standard sensor format, in millimetres."""

    width_mm: float
    height_mm: float
    used_by: str


# Nominal dimensions of standard sensor formats, verified against published
# specs (see docs/CAPTURE_SOP.md for sources). These are the *full-frame*
# dimensions: if the drone records video in a cropped mode (e.g. 16:9 from a
# 3:2 sensor), pass explicit sensor dimensions for the crop instead — the
# |fx-fy| sanity check will catch a mismatch, but explicit dims are better
# than a caught error. Add a format here only with a published spec behind it.
SENSOR_FORMATS: dict[str, SensorFormat] = {
    "4/3": SensorFormat(
        width_mm=17.3,
        height_mm=13.0,
        used_by="DJI Mavic 3 / Mavic 3 Pro / Mavic 3M main camera (Hasselblad, 12.29 mm lens)",
    ),
    "1-inch": SensorFormat(
        width_mm=13.2,
        height_mm=8.8,
        used_by="DJI Mavic 2 Pro, Phantom 4 Pro, Air 2S",
    ),
    "1/1.3-inch": SensorFormat(
        width_mm=9.6,
        height_mm=7.2,
        used_by="DJI Mini 3 Pro, Mini 4 Pro, Air 3",
    ),
}


@dataclass(frozen=True)
class DerivedFocal:
    """A focal length in pixels, with its provenance and derivation inputs."""

    fx: float
    fy: float
    cx: float
    cy: float
    provenance: FocalProvenance
    hfov_deg: float
    focal_mm: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    note: str = ""

    @property
    def focal_px(self) -> float:
        """Single shared focal (mean of fx/fy); what SIMPLE_PINHOLE uses."""
        return 0.5 * (self.fx + self.fy)


@dataclass
class FocalSpec:
    """Everything the caller knows about the camera's focal length.

    At most one derivation path needs to be filled in; ``resolve_focal``
    picks the best available one. All fields optional — an empty spec means
    "nothing known", which resolves to the logged ``guessed`` fallback.
    """

    focal_px: float | None = None
    focal_mm: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    sensor_format: str | None = None
    exif_image: str | Path | None = None
    cx: float | None = None
    cy: float | None = None


def _hfov_deg(fx: float, width_px: int) -> float:
    return math.degrees(2.0 * math.atan(width_px / (2.0 * fx)))


def _validate_focal_px(fx: float, fy: float, width_px: int, height_px: int) -> None:
    """Plausibility gate shared by every derivation path. Raises Uav3dError."""
    if not (fx > 0 and fy > 0):
        raise Uav3dError(f"Focal length must be positive, got fx={fx}, fy={fy}")
    mean_f = 0.5 * (fx + fy)
    if abs(fx - fy) / mean_f > FX_FY_MAX_REL_DIFF:
        raise Uav3dError(
            f"fx ({fx:.1f} px) and fy ({fy:.1f} px) differ by "
            f"{100.0 * abs(fx - fy) / mean_f:.1f}% (limit 2%). Square sensor pixels "
            "mean fx and fy must agree — the sensor dimensions almost certainly "
            "belong to a different crop/mode than this video (e.g. a 3:2 sensor "
            "shooting 16:9). Supply the *effective* sensor size for the video "
            "mode, or a measured focal in pixels."
        )
    lo, hi = FOCAL_MIN_REL_WIDTH * width_px, FOCAL_MAX_REL_WIDTH * width_px
    if not (lo <= fx <= hi) or not (lo <= fy <= hi):
        raise Uav3dError(
            f"Derived focal fx={fx:.1f} px, fy={fy:.1f} px is outside the plausible "
            f"range {lo:.0f}-{hi:.0f} px (0.3x-3x image width {width_px} px). Check "
            "the focal length and sensor size — a 35mm-equivalent focal fed in as "
            "the real focal length is the usual cause."
        )
    hfov = _hfov_deg(fx, width_px)
    if not (HFOV_MIN_DEG <= hfov <= HFOV_MAX_DEG):
        raise Uav3dError(
            f"Derived focal fx={fx:.1f} px gives a horizontal field of view of "
            f"{hfov:.1f} deg, outside the plausible {HFOV_MIN_DEG:.0f}-"
            f"{HFOV_MAX_DEG:.0f} deg range for a drone camera."
        )


def _centre_or_default(value: float | None, default: float, name: str) -> float:
    return float(value) if value is not None else default


def _sensor_dims_mm(
    sensor_width_mm: float | None,
    sensor_height_mm: float | None,
    sensor_format: str | None,
) -> tuple[float, float | None]:
    """Resolve sensor dimensions from explicit values or a format lookup."""
    if sensor_width_mm is not None:
        if sensor_width_mm <= 0:
            raise Uav3dError(f"sensor_width_mm must be positive, got {sensor_width_mm}")
        if sensor_height_mm is not None and sensor_height_mm <= 0:
            raise Uav3dError(f"sensor_height_mm must be positive, got {sensor_height_mm}")
        return float(sensor_width_mm), (
            float(sensor_height_mm) if sensor_height_mm is not None else None
        )
    if sensor_format is not None:
        fmt = SENSOR_FORMATS.get(sensor_format)
        if fmt is None:
            raise Uav3dError(
                f"Unknown sensor_format {sensor_format!r}. Known formats: "
                f"{', '.join(sorted(SENSOR_FORMATS))}. Or pass sensor_width_mm / "
                "sensor_height_mm explicitly."
            )
        return fmt.width_mm, fmt.height_mm
    raise Uav3dError(
        "Cannot derive focal from focal_mm without sensor dimensions: pass "
        "sensor_width_mm (+ sensor_height_mm), or sensor_format "
        f"(known: {', '.join(sorted(SENSOR_FORMATS))})."
    )


def focal_from_mm(
    focal_mm: float,
    sensor_width_mm: float | None,
    width_px: int,
    height_px: int,
    sensor_height_mm: float | None = None,
    sensor_format: str | None = None,
    cx: float | None = None,
    cy: float | None = None,
    provenance: FocalProvenance = "sensor_spec",
) -> DerivedFocal:
    """Derive focal pixels from a real lens focal length and sensor size.

    ``fx = (focal_mm / sensor_width_mm) * width_px`` (and likewise fy with
    sensor height). When only the sensor width is known, square pixels are
    assumed and fy = fx. The principal point defaults to the image centre.
    """
    if focal_mm is None or focal_mm <= 0:
        raise Uav3dError(f"focal_mm must be positive, got {focal_mm}")
    width_mm, height_mm = _sensor_dims_mm(sensor_width_mm, sensor_height_mm, sensor_format)
    fx = (float(focal_mm) / width_mm) * width_px
    fy = (float(focal_mm) / height_mm) * height_px if height_mm is not None else fx
    _validate_focal_px(fx, fy, width_px, height_px)
    return DerivedFocal(
        fx=fx,
        fy=fy,
        cx=_centre_or_default(cx, width_px / 2.0, "cx"),
        cy=_centre_or_default(cy, height_px / 2.0, "cy"),
        provenance=provenance,
        hfov_deg=_hfov_deg(fx, width_px),
        focal_mm=float(focal_mm),
        sensor_width_mm=width_mm,
        sensor_height_mm=height_mm,
    )


def focal_from_measured_px(
    fx: float,
    width_px: int,
    height_px: int,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> DerivedFocal:
    """Focal in pixels from a pre-flight calibration capture (best evidence).

    ``fx`` (and optionally ``fy``) come from ``cv2.calibrateCamera`` on a
    checkerboard / AprilGrid capture at the exact camera / zoom / focus that
    will fly. The principal point defaults to the image centre unless the
    calibration measured it.
    """
    fx_f = float(fx)
    fy_f = float(fy) if fy is not None else fx_f
    _validate_focal_px(fx_f, fy_f, width_px, height_px)
    return DerivedFocal(
        fx=fx_f,
        fy=fy_f,
        cx=_centre_or_default(cx, width_px / 2.0, "cx"),
        cy=_centre_or_default(cy, height_px / 2.0, "cy"),
        provenance="measured",
        hfov_deg=_hfov_deg(fx_f, width_px),
    )


def _exif_sensor_dims_mm(
    exif: dict,
    still_width_px: int,
    still_height_px: int,
) -> tuple[float | None, float | None]:
    """Sensor size from EXIF FocalPlane resolution tags, if the file has them.

    FocalPlaneXResolution is pixels per resolution unit; with the unit tag
    (2 = inch, 3 = cm) that gives pixels per mm, and sensor width is
    ``still_width_px / px_per_mm``. Returns (None, None) when the tags are
    absent — most drone JPEGs do not carry them.
    """
    res_x = exif.get(_EXIF_FOCAL_PLANE_X_RES)
    res_y = exif.get(_EXIF_FOCAL_PLANE_Y_RES)
    unit = exif.get(_EXIF_FOCAL_PLANE_RES_UNIT)
    if res_x is None or unit is None:
        return None, None
    mm_per_unit = {2: 25.4, 3: 10.0}.get(int(unit))
    if mm_per_unit is None:
        return None, None
    px_per_mm_x = float(res_x) / mm_per_unit
    width_mm = still_width_px / px_per_mm_x if px_per_mm_x > 0 else None
    height_mm = None
    if res_y is not None:
        px_per_mm_y = float(res_y) / mm_per_unit
        height_mm = still_height_px / px_per_mm_y if px_per_mm_y > 0 else None
    return width_mm, height_mm


def focal_from_exif(
    image_path: str | Path,
    width_px: int,
    height_px: int,
    sensor_width_mm: float | None = None,
    sensor_height_mm: float | None = None,
    sensor_format: str | None = None,
) -> DerivedFocal:
    """Derive focal pixels from a still image's EXIF focal length.

    ``image_path`` must be a still captured with the same camera, zoom and
    focus as the flight — video files carry no per-frame EXIF. The focal in
    pixels is computed at the still's resolution and rescaled to the video
    resolution; a still/video aspect mismatch is rejected, because it means
    a different sensor crop.

    Requires Pillow (present in the dev venv; the import is lazy so the
    core pipeline does not depend on it). Raises Uav3dError when the file
    has no usable focal length — in particular when the only focal tag is
    FocalLengthIn35mmFilm, which must never be used as a real focal length.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is in the venv
        raise Uav3dError(
            "Reading EXIF needs Pillow (pip install Pillow). Or pass focal_mm + "
            "sensor dimensions explicitly instead."
        ) from exc

    path = Path(image_path)
    if not path.exists():
        raise Uav3dError(f"EXIF still image not found: {path}")
    try:
        with Image.open(path) as img:
            still_w, still_h = img.size
            exif = dict(img.getexif() or {})
    except OSError as exc:
        raise Uav3dError(f"Could not read still image {path}: {exc}") from exc

    focal_mm = exif.get(_EXIF_FOCAL_LENGTH)
    if focal_mm is None:
        if exif.get(_EXIF_FOCAL_LENGTH_35MM) is not None:
            raise Uav3dError(
                f"{path} has only FocalLengthIn35mmFilm in its EXIF, which is the "
                "focal length rescaled to a 36x24 mm frame — not the real lens "
                "focal length. It must never be used to derive focal pixels. Get "
                "the real focal length from the camera spec sheet, DJI metadata, "
                "or a calibration capture (docs/CAPTURE_SOP.md)."
            )
        raise Uav3dError(
            f"{path} has no FocalLength EXIF tag. Get the focal length from the "
            "camera spec sheet, DJI metadata, or a calibration capture "
            "(docs/CAPTURE_SOP.md)."
        )

    # Sensor size: explicit dims win, then the format table, then the
    # FocalPlane resolution tags in the EXIF itself.
    if sensor_width_mm is None and sensor_format is None:
        exif_w, exif_h = _exif_sensor_dims_mm(exif, still_w, still_h)
        sensor_width_mm, sensor_height_mm = exif_w, exif_h
    derived = focal_from_mm(
        float(focal_mm),
        sensor_width_mm,
        still_w,
        still_h,
        sensor_height_mm=sensor_height_mm,
        sensor_format=sensor_format,
        provenance="exif",
    )

    # Rescale from the still's resolution to the video resolution.
    sx = width_px / still_w
    sy = height_px / still_h
    if abs(sx - sy) / (0.5 * (sx + sy)) > FX_FY_MAX_REL_DIFF:
        raise Uav3dError(
            f"Still image {still_w}x{still_h} and video {width_px}x{height_px} "
            "have different aspect ratios — they were taken in different sensor "
            "crops/modes, so the EXIF focal cannot be rescaled safely. Use a "
            "still captured in the same mode as the video, or pass focal_mm + "
            "sensor dimensions explicitly."
        )
    fx, fy = derived.fx * sx, derived.fy * sy
    _validate_focal_px(fx, fy, width_px, height_px)
    return DerivedFocal(
        fx=fx,
        fy=fy,
        cx=width_px / 2.0,
        cy=height_px / 2.0,
        provenance="exif",
        hfov_deg=_hfov_deg(fx, width_px),
        focal_mm=derived.focal_mm,
        sensor_width_mm=derived.sensor_width_mm,
        sensor_height_mm=derived.sensor_height_mm,
        note=f"EXIF from {path.name} ({still_w}x{still_h})",
    )


def guessed_focal_px(width_px: int, height_px: int) -> DerivedFocal:
    """The 0.8 * width fallback. Last resort only — always logged as a warning."""
    fx = fy = 0.8 * float(width_px)
    return DerivedFocal(
        fx=fx,
        fy=fy,
        cx=width_px / 2.0,
        cy=height_px / 2.0,
        provenance="guessed",
        hfov_deg=_hfov_deg(fx, width_px),
        note="0.8 * image width; no measured / EXIF / spec focal was provided",
    )


def resolve_focal(spec: FocalSpec | None, width_px: int, height_px: int) -> DerivedFocal:
    """Pick the best available focal derivation for a video frame size.

    Priority: measured focal_px > EXIF still > focal_mm + sensor dims >
    0.8*width guess. The guess is always logged as a warning so a run built
    on it is visibly low-confidence.
    """
    spec = spec or FocalSpec()
    if spec.focal_px is not None:
        return focal_from_measured_px(
            spec.focal_px, width_px, height_px, cx=spec.cx, cy=spec.cy
        )
    if spec.exif_image is not None:
        return focal_from_exif(
            spec.exif_image,
            width_px,
            height_px,
            sensor_width_mm=spec.sensor_width_mm,
            sensor_height_mm=spec.sensor_height_mm,
            sensor_format=spec.sensor_format,
        )
    if spec.focal_mm is not None:
        return focal_from_mm(
            spec.focal_mm,
            spec.sensor_width_mm,
            width_px,
            height_px,
            sensor_height_mm=spec.sensor_height_mm,
            sensor_format=spec.sensor_format,
            cx=spec.cx,
            cy=spec.cy,
            provenance="sensor_spec",
        )
    logger.warning(
        "No focal length provided (no focal_px / focal_mm / EXIF still) — "
        "falling back to the 0.8 * width guess (%.1f px for %dx%d). On "
        "single-pass forward flight a wrong focal stretches the whole model "
        "along the flight axis while reprojection error stays low. Provide a "
        "measured focal (docs/CAPTURE_SOP.md) or focal_mm + sensor size.",
        0.8 * width_px,
        width_px,
        height_px,
    )
    return guessed_focal_px(width_px, height_px)
