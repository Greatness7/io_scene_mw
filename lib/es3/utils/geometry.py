"""Blender-free vertex payload operations shared by the NIF import and export pipelines.

A `VertexPayload` holds numeric arrays that are aligned to one vertex axis of length `N`:

| Field           | Layout                | Selection axis |
| --------------- | --------------------- | -------------- |
| `vertices`      | `(N, 3)`              | 0              |
| `normals`       | `(N, 3)` or `(0, 3)`  | 0              |
| `vertex_colors` | `(N, 4)` or `(0, 4)`  | 0              |
| `uv_sets`       | `(S, N, 2)`           | 1              |
| `weights`       | `(B, N)` or `None`    | 1              |
| `morphs`        | `(M, N, 3)` or `None` | 1              |

Every operation validates these layouts first and returns newly allocated arrays, so a result never
shares memory with its input. Input dtypes are preserved; casting belongs to the NIF format boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from . import meshoptimizer

Array = NDArray[Any]


def _copy(array: Array) -> Array:
    return cast(Array, array.copy())  # pyright: ignore[reportUnknownMemberType]


class GeometryError(ValueError):
    """Malformed or misaligned geometry reached a numeric operation."""


@dataclass(frozen=True, slots=True)
class VertexPayload:
    vertices: Array
    normals: Array
    vertex_colors: Array
    uv_sets: Array
    weights: Array | None = None
    morphs: Array | None = None

    def __len__(self) -> int:
        return len(self.vertices)


def empty_payload(vertices: Array, weights: Array | None = None, morphs: Array | None = None) -> VertexPayload:
    """A payload with no normals, colors, or UV sets."""
    return VertexPayload(
        vertices=vertices,
        normals=np.zeros((0, 3), dtype=vertices.dtype),
        vertex_colors=np.zeros((0, 4), dtype=vertices.dtype),
        uv_sets=np.zeros((0, len(vertices), 2), dtype=vertices.dtype),
        weights=weights,
        morphs=morphs,
    )


def validate_payload(payload: VertexPayload, name: str = "geometry") -> int:
    """Check every field against the documented layout and return the shared vertex count."""
    vertices = payload.vertices
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError(f"{name}: vertices must be shaped (N, 3), got {vertices.shape}")
    count = len(vertices)

    for field, width in (("normals", 3), ("vertex_colors", 4)):
        array: Array = getattr(payload, field)
        if array.ndim != 2 or array.shape[1] != width or len(array) not in (0, count):
            raise GeometryError(f"{name}: {field} must be shaped ({count}, {width}) or (0, {width}), got {array.shape}")

    uv_sets = payload.uv_sets
    if uv_sets.ndim != 3 or uv_sets.shape[2] != 2 or (len(uv_sets) and uv_sets.shape[1] != count):
        raise GeometryError(f"{name}: uv_sets must be shaped (S, {count}, 2), got {uv_sets.shape}")

    weights = payload.weights
    if weights is not None and (weights.ndim != 2 or weights.shape[1] != count):
        raise GeometryError(f"{name}: weights must be shaped (B, {count}), got {weights.shape}")

    morphs = payload.morphs
    if morphs is not None and (morphs.ndim != 3 or morphs.shape[1] != count or morphs.shape[2] != 3):
        raise GeometryError(f"{name}: morphs must be shaped (M, {count}, 3), got {morphs.shape}")

    return count


def validate_triangles(triangles: Array, vertex_count: int, name: str = "geometry") -> None:
    """Check that `triangles` is an integer `(T, 3)` array indexing `[0, vertex_count)`."""
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise GeometryError(f"{name}: triangles must be shaped (T, 3), got {triangles.shape}")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise GeometryError(f"{name}: triangles must hold integer indices, got {triangles.dtype}")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= vertex_count):
        raise GeometryError(
            f"{name}: triangle indices must lie in [0, {vertex_count}), got [{triangles.min()}, {triangles.max()}]"
        )


def serialized_triangles(triangles: Array, name: str = "geometry") -> Array:
    """Cast validated triangle indices to the NIF's little-endian `uint16` storage."""
    if triangles.size and (triangles.min() < 0 or triangles.max() > 0xFFFF):
        raise GeometryError(f"{name}: {triangles.max() + 1} vertices exceed the 65536 a NIF shape can index")
    return triangles.astype("<H")


def select_vertices(payload: VertexPayload, indices: Array, name: str = "geometry") -> VertexPayload:
    """Gather `indices` from every aligned field into newly allocated arrays."""
    count = validate_payload(payload, name)
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise GeometryError(f"{name}: vertex selection must be a one-dimensional integer array, got {indices.shape}")
    if indices.size and (indices.min() < 0 or indices.max() >= count):
        raise GeometryError(f"{name}: vertex selection must lie in [0, {count})")

    def rows(array: Array) -> Array:
        return array[indices] if len(array) else _copy(array)

    uv_sets = payload.uv_sets
    return VertexPayload(
        vertices=payload.vertices[indices],
        normals=rows(payload.normals),
        vertex_colors=rows(payload.vertex_colors),
        uv_sets=uv_sets[:, indices] if len(uv_sets) else np.zeros((0, len(indices), 2), dtype=uv_sets.dtype),
        weights=None if payload.weights is None else payload.weights[:, indices],
        morphs=None if payload.morphs is None else payload.morphs[:, indices],
    )


def unique_rows(*arrays: Array, precision: float) -> tuple[Array, Array]:
    """Group rows that are equal after quantizing every column to `precision`.

    Each nonempty array contributes columns: a `(N,)` array one column, a `(N, K)` array `K` columns. Empty
    arrays are ignored. Row families such as weights `(B, N)` or morphs `(M, N, 3)` must be unpacked by the
    caller (`*weights`) so each member is row-aligned.

    Returns `(indices, inverse)`: `indices` selects one representative row per group, in sorted order of the
    quantized rows, and `inverse` maps every input row to its group.
    """
    if not precision > 0:
        raise GeometryError(f"unique_rows: precision must be positive, got {precision}")

    columns = [array for array in arrays if len(array)]
    if not columns:
        raise GeometryError("unique_rows: at least one nonempty array is required")

    count = len(columns[0])
    for array in columns:
        if array.ndim not in (1, 2):
            raise GeometryError(f"unique_rows: arrays must be one- or two-dimensional, got {array.shape}")
        if len(array) != count:
            raise GeometryError(f"unique_rows: every array must have {count} rows, got {array.shape}")

    rounded = np.round(np.column_stack(columns) / precision) * precision  # pyright: ignore[reportUnknownMemberType]
    _, indices, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)  # pyright: ignore[reportUnknownMemberType]
    return indices, inverse.reshape(-1)  # pyright: ignore[reportUnknownMemberType]


def duplicate_shared_faces(payload: VertexPayload, triangles: Array, name: str = "geometry") -> tuple[VertexPayload, Array]:
    """Give every face whose sorted vertex triple repeats an earlier face its own copies of those vertices.

    The first face with a given triple keeps its vertices; later ones are rewritten to new vertices appended
    after the originals, duplicated across every aligned field.
    """
    count = validate_payload(payload, name)
    validate_triangles(triangles, count, name)

    _, first = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)  # pyright: ignore[reportUnknownMemberType]
    repeated = np.full(len(triangles), True)
    repeated[first] = False

    copies = triangles[repeated].ravel()
    selection = cast(Array, np.concatenate((np.arange(count), copies)))  # pyright: ignore[reportUnknownMemberType]
    result = select_vertices(payload, selection, name)

    triangles = _copy(triangles)
    triangles[repeated] = (np.arange(len(copies)) + count).reshape(-1, 3)  # pyright: ignore[reportUnknownMemberType]
    return result, triangles


def optimize_vertex_cache(payload: VertexPayload, triangles: Array, name: str = "geometry") -> tuple[VertexPayload, Array]:
    """Reorder triangles and vertices for the post-transform cache.

    Requires nonempty geometry whose triangles reference every vertex. Returns a payload with the same remap
    applied to every aligned field, and `uint32` triangles indexing it.
    """
    count = validate_payload(payload, name)
    validate_triangles(triangles, count, name)
    if count == 0 or len(triangles) == 0:
        raise GeometryError(f"{name}: cache optimization requires vertices and faces")
    if np.unique(triangles).size != count:  # pyright: ignore[reportUnknownMemberType]
        raise GeometryError(f"{name}: cache optimization requires every vertex to be referenced by a face")

    vertices = np.ascontiguousarray(payload.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(triangles, dtype=np.uint32)
    result = cast("tuple[Array, Array]", meshoptimizer.optimize(vertices, indices))  # pyright: ignore[reportUnknownMemberType]
    remap, optimized = np.asarray(result[0]), np.asarray(result[1])

    # `remap[old] = new`, so it applies as a scatter. Unreferenced vertices would map to uint32 max.
    if remap.shape != (count,) or np.any(remap >= count) or np.unique(remap).size != count:  # pyright: ignore[reportUnknownMemberType]
        raise GeometryError(f"{name}: cache optimizer returned an invalid vertex remap")
    validate_triangles(optimized, count, name)

    def scatter(array: Array, axis: int) -> Array:
        if array.shape[axis] == 0:
            return _copy(array)
        output = np.empty_like(array)
        if axis == 0:
            output[remap] = array
        else:
            output[:, remap] = array
        return output

    result_payload = VertexPayload(
        vertices=scatter(payload.vertices, 0),
        normals=scatter(payload.normals, 0),
        vertex_colors=scatter(payload.vertex_colors, 0),
        uv_sets=scatter(payload.uv_sets, 1),
        weights=None if payload.weights is None else scatter(payload.weights, 1),
        morphs=None if payload.morphs is None else scatter(payload.morphs, 1),
    )
    return result_payload, optimized
