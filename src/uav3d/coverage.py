"""Per-surface observation coverage: which of the model was actually seen.

The honesty feature. A single-pass drone flight only ever sees part of a
scene — rear facades, the ground under a roof overhang, anything occluded by
the building in front of it are *unobserved*, and this pipeline invents no
geometry there (the old completion code that did is quarantined in
``uav3d.modeling.legacy``). This module measures that honestly:

* every mesh vertex is projected into every reference view's filtered MVS
  depth map;
* a view "observes" a vertex when the depth map agrees with the vertex's
  camera-frame depth within a tolerance derived from the dense stage's own
  depth sigma;
* the per-vertex observation count is reported as coverage fractions
  (observed by >= 2 / >= 3 / >= 5 views) and shipped as a per-vertex array
  the viewer can colour by.

No ray-caster is built here: the filtered depth maps *are* the observation
record — a pixel with valid depth in a view is a surface that view measured.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import numpy as np

from uav3d.dense.cameras import CameraView, build_views, load_run_inputs
from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.coverage")

#: Thresholds (minimum agreeing views) reported in every coverage report.
COVERAGE_THRESHOLDS: tuple[int, ...] = (2, 3, 5)

#: Agreement tolerance in multiples of the dense stage's depth sigma.
TOLERANCE_SIGMAS: float = 3.0


# ---------------------------------------------------------------------------
# GLB vertex reading (dependency-free mirror of dense/glb_export.py)
# ---------------------------------------------------------------------------

_GLTF_COMPONENT_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_GLTF_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# glTF is Y-up; the pipeline works in a Z-up world. dense/glb_export.write_glb
# rotates -90 deg about X on write (gltf = (x, z, -y)); the inverse below
# restores SfM-frame coordinates so GLB vertices can be projected into the
# MVS cameras. The rotation is per-vertex, so vertex ORDER (which the
# viewer's coverage array relies on) is untouched.
_GLTF_TO_SFM_ROT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def gltf_positions_to_sfm(positions: np.ndarray) -> np.ndarray:
    """Undo write_glb's Z-up -> Y-up rotation: glTF (x, y, z) -> SfM (x, -z, y)."""
    return np.asarray(positions, dtype=np.float64) @ _GLTF_TO_SFM_ROT


def read_glb_positions(path: str | Path) -> np.ndarray:
    """Read POSITION vertices from a GLB file (first mesh, first primitive).

    Returns an ``(N, 3)`` float64 array in the order the GLB stores them —
    which is the order the viewer sees them, so per-vertex coverage arrays
    computed from these positions line up with the rendered mesh by index.
    """
    raw = Path(path).read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise DataError(f"Not a GLB file: {path}")
    _magic, _version, _length = struct.unpack_from("<III", raw, 0)
    json_len, json_type = struct.unpack_from("<II", raw, 12)
    if json_type != 0x4E4F534A:  # "JSON"
        raise DataError(f"GLB has no JSON chunk: {path}")
    gltf = json.loads(raw[20 : 20 + json_len].decode("utf-8"))

    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", raw, bin_off)
    if bin_type != 0x004E4942:  # "BIN"
        raise DataError(f"GLB has no BIN chunk: {path}")
    binary = raw[bin_off + 8 : bin_off + 8 + bin_len]

    try:
        primitive = gltf["meshes"][0]["primitives"][0]
        pos_accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
    except (KeyError, IndexError) as exc:
        raise DataError(f"GLB has no mesh POSITION attribute: {path}") from exc
    if pos_accessor["componentType"] != 5126 or pos_accessor["type"] != "VEC3":
        raise DataError(
            f"Unsupported POSITION accessor in {path}: "
            f"componentType={pos_accessor['componentType']}, type={pos_accessor['type']}"
        )
    view = gltf["bufferViews"][pos_accessor["bufferView"]]
    count = int(pos_accessor["count"])
    item_size = 12  # 3 x float32
    stride = int(view.get("byteStride", item_size))
    start = int(view.get("byteOffset", 0)) + int(pos_accessor.get("byteOffset", 0))
    if stride == item_size:
        return np.frombuffer(binary, dtype="<f4", count=count * 3, offset=start).reshape(count, 3).astype(np.float64)
    # Interleaved: copy out the strided rows.
    out = np.empty((count, 3), dtype=np.float64)
    for i in range(count):
        out[i] = np.frombuffer(binary, dtype="<f4", count=3, offset=start + i * stride)
    return out


def read_obj_positions(path: str | Path) -> np.ndarray:
    """Read ``v x y z`` vertex lines from a Wavefront OBJ file."""
    verts: list[list[float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise DataError(f"No vertices found in OBJ file: {path}")
    return np.array(verts, dtype=np.float64)


# ---------------------------------------------------------------------------
# Observation counting (pure numpy — unit-testable without a run dir)
# ---------------------------------------------------------------------------


def count_observations(
    vertices: np.ndarray,
    views: list[CameraView],
    depth_maps: dict[int, np.ndarray],
    tolerance: float,
) -> np.ndarray:
    """Count, per vertex, how many views' depth maps agree with it.

    Args:
        vertices: ``(N, 3)`` world points in the SAME units as the depth maps
            (SfM units for this pipeline — mesh vertices in metres must be
            un-scaled first; see :func:`coverage_report_for_run`).
        views: camera views, indexed the same way as ``depth_maps`` keys.
        depth_maps: view index -> ``(H, W)`` depth map in SfM units; pixels
            with depth <= 0 are unobserved in that view.
        tolerance: max |depth_map - vertex depth| that still counts as an
            observation, in depth-map units.

    Returns:
        ``(N,)`` uint16 array of observation counts.
    """
    counts = np.zeros(vertices.shape[0], dtype=np.uint16)
    if vertices.shape[0] == 0:
        return counts
    for index, depth in depth_maps.items():
        if index >= len(views):
            logger.warning("Depth map for view %d has no matching camera; skipping", index)
            continue
        view = views[index]
        height, width = depth.shape
        uv, z = view.project(vertices)
        u = np.round(uv[:, 0] - 0.5).astype(np.int64)
        v = np.round(uv[:, 1] - 0.5).astype(np.int64)
        in_frame = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(in_frame):
            continue
        sampled = np.zeros(vertices.shape[0], dtype=np.float64)
        sampled[in_frame] = depth[v[in_frame], u[in_frame]]
        agrees = in_frame & (sampled > 0) & (np.abs(sampled - z) <= tolerance)
        counts += agrees.astype(np.uint16)
    return counts


def summarize_coverage(
    counts: np.ndarray,
    thresholds: tuple[int, ...] = COVERAGE_THRESHOLDS,
) -> dict[str, float | int]:
    """Turn per-vertex observation counts into report fractions."""
    total = int(counts.shape[0])
    summary: dict[str, float | int] = {"num_vertices": total}
    if total == 0:
        summary.update({f"fraction_observed_ge_{t}": 0.0 for t in thresholds})
        summary.update({"mean_observations": 0.0, "median_observations": 0.0, "fraction_unobserved": 0.0})
        return summary
    for t in thresholds:
        summary[f"fraction_observed_ge_{t}"] = float(np.mean(counts >= t))
    summary["fraction_unobserved"] = float(np.mean(counts == 0))
    summary["mean_observations"] = float(np.mean(counts))
    summary["median_observations"] = float(np.median(counts))
    return summary


def human_summary(summary: dict[str, float | int]) -> str:
    """One honest sentence for the report and the viewer."""
    observed = float(summary.get("fraction_observed_ge_2", 0.0)) * 100.0
    total = int(summary.get("num_vertices", 0))
    return (
        f"{observed:.1f}% of the model surface ({total:,} vertices) was directly "
        f"observed by 2 or more views; the rest is unobserved — no geometry is "
        f"invented there."
    )


# ---------------------------------------------------------------------------
# Run-directory driver
# ---------------------------------------------------------------------------


def _load_depth_maps(filtered_dir: Path) -> dict[int, np.ndarray]:
    maps: dict[int, np.ndarray] = {}
    for path in sorted(filtered_dir.glob("*.npy")):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        maps[index] = np.load(path)
    return maps


def coverage_report_for_run(
    run_dir: str | Path,
    tolerance: float | None = None,
    write: bool = True,
) -> dict:
    """Compute coverage for a finished dense run and write the report files.

    Reads ``dense/dense_report.json`` (scale + depth sigma), ``poses.json`` /
    ``keyframes.json`` (cameras), ``dense/depth_filtered/*.npy`` (the
    observation record) and ``dense/mesh.glb`` (vertices; falls back to
    ``dense/mesh.obj``). Writes ``dense/coverage_report.json`` and
    ``dense/coverage.json`` (per-vertex counts for the viewer) and returns the
    report dict.

    Raises:
        DataError: if the run has no dense output, no depth maps, or no mesh.
    """
    run_path = Path(run_dir)
    dense_dir = run_path / "dense"
    report_path = dense_dir / "dense_report.json"
    if not report_path.exists():
        raise DataError(f"No dense_report.json in {dense_dir}; run 'uav3d dense' first.")
    dense_report = json.loads(report_path.read_text(encoding="utf-8"))

    filtered_dir = dense_dir / "depth_filtered"
    if not filtered_dir.exists():
        raise DataError(f"No filtered depth maps in {filtered_dir}; cannot measure coverage.")
    depth_maps = _load_depth_maps(filtered_dir)
    if not depth_maps:
        raise DataError(f"No .npy depth maps found in {filtered_dir}.")

    glb_path = dense_dir / "mesh.glb"
    obj_path = dense_dir / "mesh.obj"
    if glb_path.exists():
        # GLB vertices are Y-up (write_glb rotates on export); rotate back to
        # the Z-up SfM frame before projecting into the MVS cameras.
        vertices_m = gltf_positions_to_sfm(read_glb_positions(glb_path))
        mesh_source = "mesh.glb"
    elif obj_path.exists():
        # OBJ is written in raw SfM-frame coordinates (no rotation).
        vertices_m = read_obj_positions(obj_path)
        mesh_source = "mesh.obj"
    else:
        raise DataError(f"No mesh in {dense_dir}; re-run dense with --mesh to measure coverage.")

    # Mesh vertices are in METRES (the dense stage scales points at export);
    # depth maps are in SfM units. Un-scale before projecting.
    scale = dense_report.get("scale") or {}
    scale_value = float(scale.get("value", 1.0) or 1.0)
    vertices = vertices_m / scale_value if scale_value != 1.0 else vertices_m

    working_width = int((dense_report.get("geometry") or {}).get("working_width", 0) or 0)
    if working_width <= 0:
        # Fall back to the depth-map width: build_views scales to this width.
        working_width = int(next(iter(depth_maps.values())).shape[1])
    poses, keyframes = load_run_inputs(run_path)
    views = build_views(poses, keyframes, run_path, working_width)

    if tolerance is None:
        fusion = dense_report.get("fusion") or {}
        sigma = fusion.get("depth_sigma_units")
        if sigma:
            tolerance = TOLERANCE_SIGMAS * float(sigma)
        else:
            median_depth = float((dense_report.get("geometry") or {}).get("median_depth", 0.0) or 0.0)
            tolerance = 0.01 * median_depth if median_depth > 0 else 1.0
    logger.info(
        "Coverage: %d vertices x %d depth maps, tolerance %.4f (SfM units)",
        vertices.shape[0],
        len(depth_maps),
        tolerance,
    )
    counts = count_observations(vertices, views, depth_maps, float(tolerance))
    summary = summarize_coverage(counts)

    report = {
        "schema_version": "1.0.0",
        "run_dir": str(run_path),
        "mesh_source": mesh_source,
        "num_views_with_depth": len(depth_maps),
        "tolerance_units": float(tolerance),
        "tolerance_note": (
            f"{TOLERANCE_SIGMAS:g} x dense depth sigma "
            f"({float((dense_report.get('fusion') or {}).get('depth_sigma_units', 0.0)):.4f} SfM units)"
            if (dense_report.get("fusion") or {}).get("depth_sigma_units")
            else "1% of median depth (no depth sigma in dense report)"
        ),
        "scale_m_per_unit": scale_value,
        **{k: v for k, v in summary.items()},
        "summary": human_summary(summary),
    }
    if write:
        (dense_dir / "coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (dense_dir / "coverage.json").write_text(
            json.dumps(
                {"num_vertices": int(counts.shape[0]), "counts": counts.astype(int).tolist()},
            ),
            encoding="utf-8",
        )
        logger.info("Coverage report: %s", report["summary"])
    return report
