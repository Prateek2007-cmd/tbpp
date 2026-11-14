"""GLB export that a viewer can actually shade.

Why this module exists
----------------------
``trimesh.Trimesh.export('*.glb')`` writes only POSITION, COLOR_0 and indices
when the mesh has no vertex normals. The resulting glTF has:

  * no ``materials`` array,
  * no ``NORMAL`` attribute,
  * ``flatShading`` forced on by the loader because normals are absent.

three.js then builds a ``MeshStandardMaterial`` with ``vertexColors: true`` and
``flatShading: true``. With no normals, every fragment's lighting term collapses
to the ambient floor, so a mesh whose vertex colours average RGB (114, 115, 102)
renders as a near-black silhouette. In several viewers it reads as a flat
tinted blob, which is why the model looked "fully green" and impossible to
judge.

This exporter writes the two things that were missing:

  * per-vertex normals (area-weighted, from the triangle geometry), and
  * an explicit PBR material with ``baseColorFactor`` white and vertex colours
    enabled, so the captured colour survives instead of being multiplied by a
    default grey.

It also writes a ``doubleSided`` material, because a photogrammetric surface
reconstructed from a single flight line is open at the back and would otherwise
show holes wherever winding disagrees with the view direction.

The writer is dependency-free (struct + json) so it cannot drift with a trimesh
version bump.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# glTF component type codes.
_FLOAT = 5126
_UINT = 5125
_UBYTE = 5121
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals.

    The cross product of a triangle's edges has magnitude proportional to twice
    its area, so summing un-normalised face normals into their vertices weights
    each face by area for free. That is the behaviour a viewer expects: large
    triangles dominate the shading of a shared vertex.
    """
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.size == 0:
        return normals
    tri = vertices[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # A vertex touched only by degenerate triangles has no direction; give it a
    # unit up vector rather than a zero vector, which some loaders reject.
    degenerate = lengths[:, 0] < 1e-20
    normals = np.divide(normals, np.maximum(lengths, 1e-20))
    normals[degenerate] = (0.0, 0.0, 1.0)
    return normals


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    """glTF buffer views must start on 4-byte boundaries."""
    remainder = len(data) % 4
    return data + fill * (4 - remainder) if remainder else data


def write_glb(
    path: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    name: str = "mesh",
    double_sided: bool = True,
    roughness: float = 0.92,
    metallic: float = 0.0,
) -> Path:
    """Write a single-mesh GLB with normals, vertex colours and a real material.

    Args:
        path: Destination ``.glb`` file.
        vertices: ``(n, 3)`` float positions.
        faces: ``(m, 3)`` integer triangle indices.
        colors: Optional ``(n, 3)`` or ``(n, 4)`` colours. Values in ``[0, 1]``
            are treated as float; values above 1 are treated as 0-255 and
            normalised. ``None`` writes a plain white material.
        name: Mesh/node name recorded in the glTF.
        double_sided: Render back faces. Correct for open photogrammetric
            surfaces; set ``False`` only for a verified watertight mesh.
        roughness: PBR roughness. High by default: the surface is diffuse rock,
            concrete and foliage, not a specular material.
        metallic: PBR metalness. Zero for captured surfaces.

    Returns:
        The written path.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (n, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (m, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError("faces reference vertices outside the vertex array")

    normals = vertex_normals(vertices, faces)

    # glTF is Y-up; the pipeline works in a Z-up world. Rotate -90 deg about X
    # so the model stands upright in every viewer instead of lying on its side.
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    positions_out = vertices @ rot.T
    normals_out = normals @ rot.T

    pos_bytes = _pad4(np.ascontiguousarray(positions_out, dtype="<f4").tobytes())
    nrm_bytes = _pad4(np.ascontiguousarray(normals_out, dtype="<f4").tobytes())
    idx_bytes = _pad4(np.ascontiguousarray(faces.reshape(-1), dtype="<u4").tobytes())

    views: list[dict] = []
    blobs: list[bytes] = []

    def add_view(data: bytes, target: int) -> int:
        offset = sum(len(b) for b in blobs)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        blobs.append(data)
        return len(views) - 1

    pos_view = add_view(pos_bytes, _ARRAY_BUFFER)
    nrm_view = add_view(nrm_bytes, _ARRAY_BUFFER)
    idx_view = add_view(idx_bytes, _ELEMENT_ARRAY_BUFFER)

    attributes: dict[str, int] = {"POSITION": 0, "NORMAL": 1}
    accessors: list[dict] = [
        {
            "bufferView": pos_view,
            "componentType": _FLOAT,
            "count": int(positions_out.shape[0]),
            "type": "VEC3",
            "min": positions_out.min(axis=0).tolist(),
            "max": positions_out.max(axis=0).tolist(),
        },
        {
            "bufferView": nrm_view,
            "componentType": _FLOAT,
            "count": int(normals_out.shape[0]),
            "type": "VEC3",
        },
    ]

    if colors is not None:
        rgba = np.asarray(colors, dtype=np.float64)
        if rgba.ndim != 2 or rgba.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"colors must be (n, 3) or (n, 4) matching vertices, got {rgba.shape}"
            )
        if rgba.max() > 1.0 + 1e-6:
            rgba = rgba / 255.0
        if rgba.shape[1] == 3:
            rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1))], axis=1)
        rgba = np.clip(rgba, 0.0, 1.0)
        # COLOR_0 is unsigned-byte normalised, so the float range must be scaled
        # up before the cast; casting 0..1 floats directly truncates to zero.
        rgba = np.rint(rgba * 255.0).astype(np.uint8)
        col_bytes = _pad4(np.ascontiguousarray(rgba, dtype="<u1").tobytes())
        col_view = add_view(col_bytes, _ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": col_view,
                "componentType": _UBYTE,
                "normalized": True,
                "count": int(rgba.shape[0]),
                "type": "VEC4",
            }
        )
        attributes["COLOR_0"] = len(accessors) - 1

    accessors.append(
        {
            "bufferView": idx_view,
            "componentType": _UINT,
            "count": int(faces.size),
            "type": "SCALAR",
        }
    )
    index_accessor = len(accessors) - 1

    material: dict = {
        "name": f"{name}_material",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
        "doubleSided": bool(double_sided),
    }
    if colors is not None:
        material["extras"] = {"vertexColors": True}

    gltf = {
        "asset": {"version": "2.0", "generator": "uav3d.dense.glb_export"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
    }

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = b"".join(blobs)

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(out))
    return destination


def write_glb_from_mesh(path: str | Path, mesh, colors: np.ndarray | None = None, **kwargs) -> Path:
    """Convenience wrapper for a :class:`uav3d.dense.mesh.Mesh` instance."""
    return write_glb(
        path,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        colors=colors,
        **kwargs,
    )
