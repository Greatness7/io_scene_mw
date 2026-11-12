"""NIF shape geometry to Blender mesh data.

Preparation validates a shape's arrays, then builds owned buffers: vertex-aligned positions, skin weights, and
morph targets, welded and with shared faces duplicated through `es3.utils.geometry`; and corner-aligned
normals, UVs, and colors. The source NIF arrays are never modified, so shapes that share `NiTriShapeData` stay
intact. Emission then writes those buffers into a new Blender mesh and its object.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import bpy
import numpy as np
import numpy.linalg as la
from es3 import nif
from es3.utils import geometry
from es3.utils.geometry import GeometryError

from . import import_animation
from .blender_compat import ensure_action, ensure_fcurve
from .records import Array, ImportMeshBuffers, ImportMorphBuffers, ImportSkinBuffers

# Blender meshes hold at most eight UV layers.
MAX_UV_LAYERS = 8


# PREPARATION


def validate_shape(shape: Any, name: str) -> None:
    """Check the geometry, skin, and reachable morph arrays of `shape` against its vertex count."""
    data = shape.data
    count = geometry.validate_payload(
        geometry.VertexPayload(
            vertices=data.vertices,
            normals=data.normals,
            vertex_colors=data.vertex_colors,
            uv_sets=data.uv_sets,
        ),
        name,
    )
    geometry.validate_triangles(data.triangles, count, name)

    skin = shape.skin
    if skin is not None and skin.root is not None and skin.data is not None:
        if len(skin.bones) != len(skin.data.bone_data):
            raise GeometryError(f"{name}: skin has {len(skin.bones)} bones but {len(skin.data.bone_data)} bone weights")
        for bone, bone_data in zip(skin.bones, skin.data.bone_data):
            indices = bone_data.vertex_weights["f0"]
            if len(indices) and indices.max() >= count:
                raise GeometryError(f"{name}: bone '{bone.name}' weights vertex {indices.max()} of {count}")

    controller = shape.controllers.find_type(nif.NiMorpherController)
    if controller is not None and controller.data is not None:
        for i, target in enumerate(controller.data.targets):
            if target.vertices.shape != (count, 3):
                raise GeometryError(f"{name}: morph target {i} has {len(target.vertices)} vertices, the shape {count}")


def prepare_buffers(shape: Any, scale: Array, precision: float, name: str) -> ImportMeshBuffers:
    """Owned Blender-ready buffers for `shape`, whose world `scale` sizes the welding tolerance."""
    validate_shape(shape, name)
    data = shape.data
    triangles = data.triangles

    normals = data.normals
    if len(normals):
        # re-unitize, which fixes landscape meshes
        normals = (normals / la.norm(normals, axis=1, keepdims=True))[triangles].reshape(-1, 3)

    uv_sets = data.uv_sets
    if len(uv_sets):
        uv_sets = uv_sets[:, triangles].reshape(len(uv_sets), -1, 2)
        # the NIF's V axis points down, Blender's up
        uv_sets[..., 1] = 1 - uv_sets[..., 1]

    vertex_colors = data.vertex_colors
    if len(vertex_colors):
        vertex_colors = vertex_colors[triangles].reshape(-1, 4)

    weights, morphs = shape.vertex_weights(), shape.vertex_morphs()
    payload = geometry.empty_payload(data.vertices, weights, morphs)

    # remove doubles, keeping vertices that deform differently apart
    indices, inverse = geometry.unique_rows(data.vertices * scale, *weights, *morphs, precision=precision)
    if len(payload) > len(indices) > 3:
        payload = geometry.select_vertices(payload, indices, name)
        triangles = inverse[triangles]

    # Blender rejects faces that reuse another face's vertices in any order, as double-sided faces do
    payload, triangles = geometry.duplicate_shared_faces(payload, triangles, name)

    weights, morphs = payload.weights, payload.morphs
    assert weights is not None and morphs is not None
    return ImportMeshBuffers(
        vertices=payload.vertices,
        triangles=triangles,
        normals=normals,
        uv_sets=uv_sets,
        vertex_colors=vertex_colors,
        skin=ImportSkinBuffers(weights) if len(weights) else None,
        morphs=ImportMorphBuffers(morphs) if len(morphs) else None,
    )


# EMISSION


def create_vertices(ob: Any, vertices: Array) -> None:
    ob.data.vertices.add(len(vertices))
    ob.data.vertices.foreach_set("co", vertices.ravel())


def create_triangles(ob: Any, triangles: Array) -> None:
    n = len(triangles)
    ob.data.loops.add(3 * n)
    ob.data.loops.foreach_set("vertex_index", triangles.ravel())

    ob.data.polygons.add(n)
    ob.data.polygons.foreach_set("loop_total", [3] * n)
    ob.data.polygons.foreach_set("loop_start", range(0, 3 * n, 3))
    ob.data.polygons.foreach_set("use_smooth", [True] * n)

    ob.data.update()


def create_normals(ob: Any, normals: Array, ignore_custom_normals: bool) -> None:
    if len(normals) == 0:
        ob.data["ignore_normals"] = True
        return

    # Custom normals override each polygon's "use_smooth" flag, but the user may later remove them, which
    # falls back to the flags. So a polygon whose corner normals all agree is marked flat shaded.
    n0, n1, n2 = np.swapaxes(normals.reshape(-1, 3, 3), 0, 1)
    n0__eq__n1 = np.isclose(n0, n1, rtol=0, atol=1e-04)
    n1__eq__n2 = np.isclose(n1, n2, rtol=0, atol=1e-04)
    use_smooth = ~(n0__eq__n1 & n1__eq__n2).all(axis=1)
    ob.data.polygons.foreach_set("use_smooth", use_smooth)

    if ignore_custom_normals:
        return
    if ob.data.validate(verbose=False, clean_customdata=False):
        print(f"Warning: Invalid mesh data, custom normals will be skipped: ({ob.name})")
    else:
        ob.data.normals_split_custom_set(normals)
        if bpy.app.version < (4, 1, 0):
            ob.data.use_auto_smooth = True


def create_uv_sets(ob: Any, uv_sets: Array) -> None:
    for uv_set in uv_sets[:MAX_UV_LAYERS]:
        uv = ob.data.uv_layers.new()
        uv.data.foreach_set("uv", uv_set.ravel())


def create_vertex_colors(ob: Any, vertex_colors: Array) -> None:
    if len(vertex_colors):
        vc = ob.data.vertex_colors.new()
        vc.data.foreach_set("color", vertex_colors.ravel())


def create_vertex_weights(ob: Any, skin: ImportSkinBuffers | None, armature: Any, bone_names: Sequence[str]) -> None:
    """Deform `ob` by `armature`, with one vertex group per bone name in weight-row order."""
    if skin is None:
        return

    ob.modifiers.new("", "ARMATURE").object = armature
    for name, weights in zip(bone_names, skin.weights):
        vg = ob.vertex_groups.new(name=name)
        for j in np.flatnonzero(weights).tolist():
            vg.add((j,), weights[j], "ADD")


def create_vertex_morphs(
    ob: Any, morphs: ImportMorphBuffers | None, targets: Sequence[Any], scene: Any, controller: Any
) -> None:
    """Add a basis and one keyed shape key per morph target, then widen the scene frame range."""
    if morphs is None:
        return

    ob.shape_key_add(name="Basis")
    action = ensure_action(ob.data.shape_keys)

    for coordinates, target in zip(morphs.targets, targets):
        shape_key = ob.shape_key_add(name="")
        shape_key.data.foreach_set("co", coordinates.ravel())

        fc = ensure_fcurve(action, shape_key.path_from_id("value"))
        fc.keyframe_points.add(len(target.keys))
        fc.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
        import_animation.create_interpolation_data(target, fc)
        fc.update()

    import_animation.update_frame_range(scene, controller)
