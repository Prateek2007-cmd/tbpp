"""Tests for the GLB exporter that replaced the normals-less trimesh path."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from uav3d.dense.glb_export import vertex_normals, write_glb


def _read_glb(path):
    blob = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total == len(blob)

    offset = 12
    gltf = None
    buffer = b""
    while offset + 8 <= len(blob):
        length, kind = struct.unpack_from("<II", blob, offset)
        offset += 8
        payload = blob[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A:
            gltf = json.loads(payload.decode("utf-8"))
        elif kind == 0x004E4942:
            buffer = payload
    return gltf, buffer


def _accessor(gltf, buffer, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    dtype = {5125: np.uint32, 5126: np.float32, 5121: np.uint8}[accessor["componentType"]]
    flat = np.frombuffer(buffer, dtype=dtype, count=accessor["count"] * width, offset=start)
    return flat.reshape(accessor["count"], width)


@pytest.fixture
def cube():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ]
    )
    return vertices, faces


def test_vertex_normals_are_unit_length(cube):
    vertices, faces = cube
    normals = vertex_normals(vertices, faces)
    assert normals.shape == vertices.shape
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-9)


def test_vertex_normals_point_outward_on_a_cube(cube):
    vertices, faces = cube
    normals = vertex_normals(vertices, faces)
    centre = vertices.mean(axis=0)
    outward = vertices - centre
    # Every cube vertex normal must have a positive dot with its outward radius.
    assert np.all(np.sum(normals * outward, axis=1) > 0)


def test_glb_written_with_normals_and_material(tmp_path, cube):
    vertices, faces = cube
    out = write_glb(tmp_path / "cube.glb", vertices, faces, name="cube")
    assert out.exists()

    gltf, buffer = _read_glb(out)
    primitive = gltf["meshes"][0]["primitives"][0]

    # The two things trimesh's exporter dropped.
    assert "NORMAL" in primitive["attributes"]
    assert len(gltf["materials"]) == 1
    assert gltf["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]

    normals = _accessor(gltf, buffer, primitive["attributes"]["NORMAL"])
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_vertex_colours_survive_a_round_trip(tmp_path, cube):
    vertices, faces = cube
    colours = np.tile(np.array([[0.8, 0.4, 0.2]]), (vertices.shape[0], 1))
    out = write_glb(tmp_path / "cube.glb", vertices, faces, colours, name="cube")

    gltf, buffer = _read_glb(out)
    primitive = gltf["meshes"][0]["primitives"][0]
    assert "COLOR_0" in primitive["attributes"]

    decoded = _accessor(gltf, buffer, primitive["attributes"]["COLOR_0"])
    assert decoded.dtype == np.uint8
    # 0.8 * 255 = 204, 0.4 * 255 = 102, 0.2 * 255 = 51. A direct float->uint8
    # cast would truncate all three to zero, which is the bug this guards.
    np.testing.assert_array_equal(decoded[0, :3], [204, 102, 51])
    np.testing.assert_array_equal(decoded[:, 3], 255)


def test_uint8_colours_are_not_divided_twice(tmp_path, cube):
    vertices, faces = cube
    colours = np.tile(np.array([[200.0, 100.0, 50.0]]), (vertices.shape[0], 1))
    out = write_glb(tmp_path / "cube.glb", vertices, faces, colours, name="cube")

    gltf, buffer = _read_glb(out)
    primitive = gltf["meshes"][0]["primitives"][0]
    decoded = _accessor(gltf, buffer, primitive["attributes"]["COLOR_0"])
    np.testing.assert_array_equal(decoded[0, :3], [200, 100, 50])


def test_material_is_double_sided_by_default(tmp_path, cube):
    vertices, faces = cube
    out = write_glb(tmp_path / "cube.glb", vertices, faces, name="cube")
    gltf, _ = _read_glb(out)
    assert gltf["materials"][0]["doubleSided"] is True


def test_rejects_faces_outside_the_vertex_array(tmp_path, cube):
    vertices, faces = cube
    bad = faces.copy()
    bad[0, 0] = 999
    with pytest.raises(ValueError, match="outside the vertex array"):
        write_glb(tmp_path / "bad.glb", vertices, bad, name="bad")


def test_rejects_mismatched_colour_count(tmp_path, cube):
    vertices, faces = cube
    with pytest.raises(ValueError, match="colors must be"):
        write_glb(tmp_path / "bad.glb", vertices, faces, np.zeros((3, 3)), name="bad")


def test_empty_face_array_still_writes_a_file(tmp_path):
    vertices = np.zeros((3, 3))
    out = write_glb(tmp_path / "empty.glb", vertices, np.zeros((0, 3), dtype=np.int64), name="e")
    gltf, _ = _read_glb(out)
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"] == 0
