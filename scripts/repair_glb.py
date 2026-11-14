"""Repair an existing map_3d.glb so a viewer can shade it.

The GLBs this repo produced before ``uav3d.dense.glb_export`` existed carry
POSITION, COLOR_0 and indices but no NORMAL and no material. three.js then
builds a flat-shaded MeshStandardMaterial, the lighting term collapses to the
ambient floor, and the model renders as an unreadable dark blob that looks
uniformly tinted.

This script rewrites such a file in place (or to a new path) with:

  * area-weighted vertex normals,
  * an explicit PBR material with white base colour and vertex colours enabled,
  * double-sided rendering for the open photogrammetric surface.

It reads the geometry straight out of the GLB, so it does not need the original
run directory, the point cloud, or a re-run of the pipeline.

Usage:
    python scripts/repair_glb.py data/outputs/run_20261111_010644/map_3d.glb
    python scripts/repair_glb.py OLD.glb --out NEW.glb
    python scripts/repair_glb.py --all          # every map_3d.glb under data/
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from uav3d.dense.glb_export import write_glb

_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytes]:
    """Split a GLB into its JSON chunk and binary buffer."""
    blob = path.read_bytes()
    if len(blob) < 12:
        raise ValueError(f"{path} is too short to be a GLB")
    magic, version, _total = struct.unpack_from("<III", blob, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB (bad magic)")
    if version != 2:
        raise ValueError(f"{path} is glTF version {version}; only 2 is supported")

    offset = 12
    json_chunk = b""
    binary_chunk = b""
    while offset + 8 <= len(blob):
        length, kind = struct.unpack_from("<II", blob, offset)
        offset += 8
        payload = blob[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A:
            json_chunk = payload
        elif kind == 0x004E4942:
            binary_chunk = payload
    if not json_chunk:
        raise ValueError(f"{path} has no JSON chunk")
    return json.loads(json_chunk.decode("utf-8")), binary_chunk


def accessor_array(gltf: dict, buffer: bytes, index: int) -> np.ndarray:
    """Decode one accessor into a numpy array."""
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    width = _WIDTH[accessor["type"]]
    dtype = _COMPONENT[accessor["componentType"]]
    count = accessor["count"]
    flat = np.frombuffer(buffer, dtype=dtype, count=count * width, offset=start)
    return flat.reshape(count, width)


def repair(path: Path, out: Path | None = None) -> Path:
    """Rewrite ``path`` with normals and a material. Returns the written path."""
    gltf, buffer = read_glb(path)
    meshes = gltf.get("meshes") or []
    if not meshes:
        raise ValueError(f"{path} contains no meshes")

    # A GLB may hold several primitives; concatenate them into one mesh so the
    # repaired file stays a single draw call.
    all_positions: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    vertex_offset = 0
    has_color = True

    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError(f"{path}: only triangle primitives are supported")
            attrs = primitive["attributes"]
            positions = accessor_array(gltf, buffer, attrs["POSITION"]).astype(np.float64)
            if "indices" in primitive:
                faces = accessor_array(gltf, buffer, primitive["indices"]).reshape(-1, 3)
            else:
                faces = np.arange(positions.shape[0], dtype=np.int64).reshape(-1, 3)

            if "COLOR_0" in attrs:
                colors = accessor_array(gltf, buffer, attrs["COLOR_0"])
                if colors.dtype != np.uint8:
                    colors = np.clip(colors, 0.0, 1.0) * 255.0
                colors = colors[:, :3].astype(np.float64) / 255.0
            else:
                has_color = False
                colors = np.full((positions.shape[0], 3), 0.72)

            all_positions.append(positions)
            all_faces.append(faces.astype(np.int64) + vertex_offset)
            all_colors.append(colors)
            vertex_offset += positions.shape[0]

    vertices = np.concatenate(all_positions, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors = np.concatenate(all_colors, axis=0) if has_color else None

    destination = out or path
    return write_glb(destination, vertices, faces, colors, name=path.stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glb", nargs="?", type=Path, help="GLB to repair")
    parser.add_argument("--out", type=Path, default=None, help="Write here instead of in place")
    parser.add_argument(
        "--all", action="store_true", help="Repair every map_3d.glb under data/"
    )
    args = parser.parse_args()

    if args.all:
        targets = sorted((REPO_ROOT / "data").rglob("map_3d.glb"))
        if not targets:
            print("No map_3d.glb files found under data/")
            return 1
    elif args.glb:
        targets = [args.glb]
    else:
        parser.error("give a GLB path or --all")
        return 2

    failures = 0
    for target in targets:
        try:
            written = repair(target, args.out)
            size_mb = written.stat().st_size / (1024 * 1024)
            print(f"repaired {target} -> {written} ({size_mb:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAILED   {target}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
