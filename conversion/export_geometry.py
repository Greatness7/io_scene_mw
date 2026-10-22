"""Blender mesh geometry to NIF shape data.

Export runs in three steps:

1. Capture: author state that export evaluation would disturb (shape-key eligibility, keys, and
   coordinates) is read from the untouched source.
2. Extraction: inside `runtime.export_evaluation_scope`, each evaluated mesh is copied into owned,
   corner-aligned `ExportMeshBuffers` before `runtime.evaluated_mesh` releases it.
3. Emission: after the scope restores the source, each material partition is selected, its equivalent corners
   are welded into vertices, the vertex cache is optimized, and `NiTriShapeData`, skin, and morph data are
   written.

The numeric work goes through `es3.utils.geometry`, so every aligned field shares each selection and remap.
Nothing here keeps an evaluated object, evaluated mesh, or dependency graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.linalg as la
from es3 import nif
from es3.utils import geometry
from es3.utils.geometry import GeometryError, VertexPayload

from . import export_animation
from .blender_compat import get_fcurves
from .options import ExportOptions
from .records import (
    Array,
    ExportMeshBuffers,
    ExportMeshMetadata,
    ExportMorphBuffers,
    ExportMorphCapture,
    ExportSkinBinding,
    ExportSkinBuffers,
    MaterialPartition,
    ResolvedExportSkeleton,
)

# Triangles whose UV area is below this are treated as having no handedness.
UV_HANDEDNESS_EPSILON = 1e-6


def payload(buffers: ExportMeshBuffers) -> VertexPayload:
    return VertexPayload(
        vertices=buffers.vertices,
        normals=buffers.normals,
        vertex_colors=buffers.vertex_colors,
        uv_sets=buffers.uv_sets,
        weights=None if buffers.skin is None else buffers.skin.weights,
        morphs=None if buffers.morphs is None else buffers.morphs.targets,
    )


def with_payload(buffers: ExportMeshBuffers, values: VertexPayload, triangles: Array, **changes: Any) -> ExportMeshBuffers:
    skin = buffers.skin
    morphs = buffers.morphs
    if skin is not None:
        assert values.weights is not None
        skin = replace(skin, weights=values.weights)
    if morphs is not None:
        assert values.morphs is not None
        morphs = replace(morphs, targets=values.morphs)
    return replace(
        buffers,
        vertices=values.vertices,
        normals=values.normals,
        vertex_colors=values.vertex_colors,
        uv_sets=values.uv_sets,
        triangles=triangles,
        skin=skin,
        morphs=morphs,
        **changes,
    )


# -------
# CAPTURE
# -------


def capture_metadata(source: Any, options: ExportOptions) -> ExportMeshMetadata:
    """Read the author state of mesh object `source` before evaluation changes it."""
    return ExportMeshMetadata(
        ignore_normals=bool(source.data.get("ignore_normals")),
        morphs=capture_morphs(source, options),
    )


def capture_morphs(source: Any, options: ExportOptions) -> ExportMorphCapture | None:
    """The basis and every exportable shape-key target, or None when nothing would animate."""
    if not options.export_animations:
        return None

    shape_keys = source.data.shape_keys
    if shape_keys is None or len(shape_keys.key_blocks) == 0:
        return None
    basis = shape_keys.key_blocks[0]

    # a target exports only when it and its F-curve are unmuted and keyed; the engine skips keyless targets
    fcurves = {fc.data_path: fc for fc in get_fcurves(shape_keys.animation_data)}
    targets: list[tuple[Any, Any]] = []
    for block in shape_keys.key_blocks:
        fc = fcurves.get(block.path_from_id("value"))
        if fc and len(fc.keyframe_points) and not (fc.mute or block.mute):
            if block.relative_key != basis:
                raise GeometryError(f"{source.name}: shape key '{block.name}' must be relative to '{basis.name}'")
            targets.append((fc, block))
    if not targets:
        return None

    coordinates = np.empty((1 + len(targets), len(basis.data), 3))
    keys: list[Array] = [np.empty(0)]
    interpolations: list[int] = [nif.NiFloatData.KeyType.LIN_KEY]
    basis.data.foreach_get("co", coordinates[0].ravel())
    for i, (fc, block) in enumerate(targets, start=1):
        key_type = export_animation.get_interpolation_type([fc])
        assert key_type is not None  # eligible targets are keyed
        keys.append(export_animation.collect_keyframe_points([fc], key_type))
        interpolations.append(key_type)
        block.data.foreach_get("co", coordinates[i].ravel())

    return ExportMorphCapture(coordinates=coordinates, keys=tuple(keys), interpolations=tuple(interpolations))


def skin_binding(source: Any, armature: Any, skeleton: ResolvedExportSkeleton) -> ExportSkinBinding | None:
    """Bind the vertex groups of `source` that name a bone of `armature` to that bone's export node."""
    bone_nodes = {node.source: node for node in skeleton.bones}
    groups: list[int] = []
    bones: list[Any] = []
    for i, group in enumerate(source.vertex_groups):
        pose_bone = armature.pose.bones.get(group.name)
        if pose_bone is not None:
            groups.append(i)
            bones.append(bone_nodes[pose_bone])
    if not groups:
        return None
    return ExportSkinBinding(
        root=skeleton.armature,
        groups=tuple(groups),
        bones=tuple(bones),
        bind_relative_to_root=skeleton.bind_relative_to_root,
    )


# ----------
# EXTRACTION
# ----------


def extract_mesh_buffers(
    evaluated: Any,
    mesh: Any,
    metadata: ExportMeshMetadata,
    skin: ExportSkinBinding | None,
    *,
    name: str,
    is_collider: bool,
    baked_vertex_scale: Array | None,
) -> ExportMeshBuffers:
    """Copy `mesh`, the evaluated mesh of object `evaluated`, into owned corner-aligned buffers.

    Every returned array is newly allocated, so the buffers stay valid after the evaluated mesh is released.
    """
    mesh.validate(clean_customdata=False)
    mesh.validate_material_indices()

    corners, triangles = read_topology(mesh, name)
    count = len(mesh.vertices)
    geometry.validate_triangles(triangles, count, name)

    positions = np.empty((count, 3), dtype="<f")
    mesh.vertices.foreach_get("co", positions.ravel())
    if baked_vertex_scale is not None:
        positions *= baked_vertex_scale

    return ExportMeshBuffers(
        vertices=positions[triangles].reshape(-1, 3),
        normals=read_normals(mesh, corners, metadata.ignore_normals or is_collider, baked_vertex_scale),
        vertex_colors=read_vertex_colors(evaluated, mesh, corners, is_collider),
        uv_sets=read_uv_sets(mesh, corners, is_collider),
        triangles=np.arange(len(corners)).reshape(-1, 3),
        partitions=read_partitions(evaluated, mesh),
        skin=None if is_collider else read_skin(mesh, skin, triangles),
        morphs=None if is_collider else read_morphs(metadata.morphs, count, triangles, baked_vertex_scale, name),
    )


def read_topology(mesh: Any, name: str) -> tuple[Array, Array]:
    """The loop index `(3T,)` and vertex indices `(T, 3)` of every triangle corner, in face order."""
    face_count = len(mesh.polygons)
    loop_totals = np.empty(face_count, dtype=np.int32)
    loop_starts = np.empty(face_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", loop_totals)
    mesh.polygons.foreach_get("loop_start", loop_starts)
    if np.any(loop_totals != 3):
        raise GeometryError(f"{name}: the evaluated mesh still has faces that are not triangles")
    if len(mesh.loops) != 3 * face_count:
        raise GeometryError(f"{name}: {len(mesh.loops)} loops cannot form {face_count} triangles")
    if face_count == 0 and len(mesh.vertices):
        raise GeometryError(f"{name}: {len(mesh.vertices)} vertices have no faces")

    loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    corners = (loop_starts[:, None] + np.arange(3)).ravel()
    return corners, loop_vertices[corners].reshape(-1, 3)


def read_normals(mesh: Any, corners: Array, ignore: bool, baked_vertex_scale: Array | None) -> Array:
    if ignore:
        return np.zeros((0, 3), dtype="<f")

    normals = np.empty((len(mesh.loops), 3), dtype="<f")
    if hasattr(mesh, "calc_normals_split"):  # before Blender 4.1
        mesh.calc_normals_split()
        mesh.loops.foreach_get("normal", normals.ravel())
        mesh.free_normals_split()
    else:
        mesh.corner_normals.foreach_get("vector", normals.ravel())

    normals = normals[corners]
    if baked_vertex_scale is not None:
        normals /= baked_vertex_scale
    normals /= la.norm(normals, axis=1, keepdims=True)
    return normals


def read_vertex_colors(evaluated: Any, mesh: Any, corners: Array, is_collider: bool) -> Array:
    # colors only reach the engine through a material
    if is_collider or len(evaluated.material_slots) == 0 or len(mesh.vertex_colors) == 0:
        return np.zeros((0, 4), dtype="<f")

    colors = np.empty((len(mesh.loops), 4), dtype="<f")
    mesh.vertex_colors[0].data.foreach_get("color", colors.ravel())
    return colors[corners]


def read_uv_sets(mesh: Any, corners: Array, is_collider: bool) -> Array:
    if is_collider or not any(uv.data for uv in mesh.uv_layers):
        return np.zeros((0, len(corners), 2), dtype="<f")

    uv_sets = np.empty((len(mesh.uv_layers), len(mesh.loops), 2), dtype="<f")
    for uv_set, layer in zip(uv_sets, mesh.uv_layers):
        layer.data.foreach_get("uv", uv_set.ravel())
    # Blender's V axis points up, the NIF's down
    uv_sets[..., 1] = 1 - uv_sets[..., 1]
    return uv_sets[:, corners]


def read_partitions(evaluated: Any, mesh: Any) -> tuple[MaterialPartition[Any], ...]:
    """One partition per used material; a single-material mesh keeps every corner."""
    material_indices = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", material_indices)

    used = np.unique(material_indices)
    if len(used) <= 1:
        return (MaterialPartition(evaluated.active_material, None),)

    corners = np.arange(3 * len(material_indices)).reshape(-1, 3)
    return tuple(
        MaterialPartition(evaluated.material_slots[i].material, corners[material_indices == i].ravel()) for i in used
    )


def read_skin(mesh: Any, binding: ExportSkinBinding | None, triangles: Array) -> ExportSkinBuffers[Any] | None:
    if binding is None or len(mesh.vertices) == 0:
        return None

    weights = np.zeros((len(binding.groups), len(mesh.vertices)))
    rows = {group: i for i, group in enumerate(binding.groups)}
    for i, vertex in enumerate(mesh.vertices):
        for element in vertex.groups:
            row = rows.get(element.group)
            if row is not None:
                weights[row, i] = element.weight

    # keep the four highest weights of each vertex
    if len(weights) > 4:
        np.put_along_axis(weights, np.argpartition(weights, -4, axis=0)[:-4], 0, axis=0)

    # normalize, assigning unweighted vertices to the first bone
    sums = weights.sum(axis=0)
    unweighted = sums == 0
    weights[0, unweighted] = sums[unweighted] = 1.0
    weights /= sums

    return ExportSkinBuffers(
        root=binding.root,
        bones=binding.bones,
        weights=weights[:, triangles].reshape(len(weights), -1),
        bind_relative_to_root=binding.bind_relative_to_root,
    )


def read_morphs(
    capture: ExportMorphCapture | None,
    vertex_count: int,
    triangles: Array,
    baked_vertex_scale: Array | None,
    name: str,
) -> ExportMorphBuffers | None:
    """The captured basis and relative targets, per corner of the evaluated triangles."""
    if capture is None:
        return None

    source_count = capture.coordinates.shape[1]
    if source_count != vertex_count:
        raise GeometryError(
            f"{name}: its shape keys have {source_count} vertices but the evaluated mesh has {vertex_count}; "
            "shape keys cannot be exported through a modifier that changes the vertex count"
        )

    targets = capture.coordinates.copy()
    targets[1:] -= targets[0]
    if baked_vertex_scale is not None:
        targets *= baked_vertex_scale

    return ExportMorphBuffers(
        targets=targets[:, triangles].reshape(len(targets), -1, 3),
        keys=capture.keys,
        interpolations=capture.interpolations,
    )


# --------
# EMISSION
# --------


def select_partition(buffers: ExportMeshBuffers, partition: MaterialPartition, name: str) -> ExportMeshBuffers:
    """A new single-partition buffer holding copies of the partition's corners."""
    count = geometry.validate_payload(payload(buffers), name)
    corners = np.arange(count) if partition.corner_indices is None else partition.corner_indices
    if len(corners) % 3:
        raise GeometryError(f"{name}: a material partition must select whole triangles, got {len(corners)} corners")
    values = geometry.select_vertices(payload(buffers), corners, name)
    triangles = np.arange(len(corners)).reshape(-1, 3)
    return with_payload(buffers, values, triangles, partitions=(replace(partition, corner_indices=None),))


def uv_handedness(uv_set: Array, triangles: Array) -> Array:
    """The sign of each triangle's UV winding, per corner, so mirrored UV layouts never share vertices."""
    uvs = uv_set[triangles]
    d_uv_1 = uvs[:, 1] - uvs[:, 0]
    d_uv_2 = uvs[:, 2] - uvs[:, 0]
    cross_z = (d_uv_1[:, 0] * d_uv_2[:, 1]) - (d_uv_1[:, 1] * d_uv_2[:, 0])
    cross_z[np.abs(cross_z) < UV_HANDEDNESS_EPSILON] = 0
    handedness = np.empty(len(uv_set), dtype=cross_z.dtype)
    handedness[triangles] = np.sign(cross_z)[:, None]
    return handedness


def weld(buffers: ExportMeshBuffers, precision: float, name: str) -> ExportMeshBuffers:
    """Merge equivalent corners into shared vertices and optimize the result for the vertex cache.

    Corners are equivalent when their positions, normals, colors, every UV set, first-set UV handedness, every
    skin weight, and every morph target agree at `precision`. The returned triangles are serialized `uint16`
    indices.
    """
    values = payload(buffers)
    count = geometry.validate_payload(values, name)
    triangles = buffers.triangles
    if count == 0 and len(triangles) == 0:
        return buffers
    if len(triangles) == 0:
        raise GeometryError(f"{name}: {count} vertices have no faces")
    geometry.validate_triangles(triangles, count, name)
    if count != triangles.size or np.unique(triangles).size != count:
        raise GeometryError(f"{name}: welding requires every triangle corner to be its own vertex")

    columns = [values.vertices, values.normals, values.vertex_colors, *values.uv_sets]
    if len(values.uv_sets):
        columns.append(uv_handedness(values.uv_sets[0], triangles))
    # deformation is part of a vertex: corners that share a position may still move apart
    if values.weights is not None:
        columns.extend(values.weights)
    if values.morphs is not None:
        columns.extend(values.morphs)

    indices, inverse = geometry.unique_rows(*columns, precision=precision)
    welded = geometry.select_vertices(values, indices, name)
    welded, triangles = geometry.optimize_vertex_cache(welded, inverse[triangles], name)
    return with_payload(buffers, welded, geometry.serialized_triangles(triangles, name))


def write_shape_data(data: nif.NiGeometryData, buffers: ExportMeshBuffers) -> None:
    """Assign welded buffers to shape data; absent fields keep the NIF defaults."""
    for field in ("vertices", "normals", "vertex_colors", "uv_sets", "triangles"):
        array = getattr(buffers, field)
        if len(array):
            setattr(data, field, array)


def create_skin(shape: nif.NiTriShape, buffers: ExportMeshBuffers, offset: Array, outputs: Mapping[Any, Any]) -> None:
    """Skin `shape` from welded weights; `offset` is the shape's transform relative to the skin root.

    `outputs` maps each skeleton node to the NIF node already created for it.
    """
    skin_data = buffers.skin
    if skin_data is None:
        return

    missing = [node.name for node in (skin_data.root, *skin_data.bones) if node not in outputs]
    if missing:
        raise RuntimeError(f"Skin references skeleton outputs that have not been created: {', '.join(missing)}")

    skin = shape.skin = nif.NiSkinInstance()
    skin.root, *bones = [outputs[node] for node in (skin_data.root, *skin_data.bones)]
    skin.data = nif.NiSkinData()
    skin.data.matrix = la.inv(offset)

    used_bones: list[Any] = []
    bone_datas: list[nif.NiSkinDataBoneData] = []
    for i, bone in enumerate(bones):
        (j,) = np.nonzero(skin_data.weights[i])
        # bones without weights are left out
        if len(j) == 0:
            continue

        bone_data = nif.NiSkinDataBoneData()
        bone_data.vertex_weights.resize(len(j))
        bone_data.vertex_weights["f0"] = j
        bone_data.vertex_weights["f1"] = skin_data.weights[i, j]

        # skin to bone offset
        matrix_bind = skin_data.bind_relative_to_root[skin_data.bones[i]]
        bone_data.matrix = la.solve(matrix_bind, offset)
        bone_data.update_center_radius(buffers.vertices[j], exact=True)

        used_bones.append(bone)
        bone_datas.append(bone_data)

    skin.bones = used_bones
    skin.data.bone_data = bone_datas


def create_morphs(shape: nif.NiTriShape, morphs: ExportMorphBuffers | None, cycle_type: str) -> None:
    """Attach a morph controller whose relative targets are the welded morph rows."""
    if morphs is None:
        return

    controller = nif.NiGeomMorpherController(cycle_type=cycle_type, target=shape)
    controller.stop_time = max(keys[-1, 0] for keys in morphs.keys if len(keys))
    controller.data = nif.NiMorphData(relative_targets=1)
    controller.data.targets = [nif.NiMorphDataMorphTarget() for _ in morphs.targets]

    for i, target in enumerate(controller.data.targets):
        target.key_type = morphs.interpolations[i]
        target.keys = morphs.keys[i]
        target.vertices = morphs.targets[i]

    shape.controllers.appendleft(controller)
