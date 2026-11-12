"""Resolve exported armature hierarchies and convert bone transforms to NIF axes.

Transform and parenting changes affect export records rather than the Blender source hierarchy.
Bind matrices are kept relative to each armature root for skin export, while axis corrections compensate
child transforms to preserve their placement.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.linalg as la

from es3.utils.transforms import evaluate_world_matrices

from .bone_axes import export_axis_correction
from .records import Array, ExportNode, ResolvedExportSkeleton


def evaluate_node_worlds(nodes: Iterable[ExportNode]) -> dict[ExportNode, Array]:
    return evaluate_world_matrices((node, node.parent, node.matrix_local) for node in nodes)


def set_world_matrix(node: ExportNode, matrix: Array, worlds: dict[ExportNode, Array]) -> None:
    if node.parent is None:
        node.matrix_local = matrix
    else:
        node.matrix_local = la.solve(worlds[node.parent], matrix)
    worlds[node] = np.array(matrix, copy=True)


def resolve_armatures(
    armature_candidates: dict[tuple[Any, Any], tuple[Any, ...]],
    nodes: list[ExportNode],
    get_node: Callable[[Any, Any], ExportNode],
) -> tuple[list[ResolvedExportSkeleton], dict[tuple[Any, Any], ResolvedExportSkeleton]]:
    skeletons: list[ResolvedExportSkeleton] = []
    skeletons_by_source: dict[tuple[Any, Any], ResolvedExportSkeleton] = {}

    for scope, source in armature_candidates:
        armature = get_node(source, scope)
        pose_bones = source.pose.bones
        bones = tuple(get_node(pose_bones[bone.name], scope) for bone in source.data.bones)
        skeleton = ResolvedExportSkeleton(armature, bones, {}, {})
        resolve_armature(skeleton, nodes, get_node)
        skeletons.append(skeleton)
        skeletons_by_source[scope, source] = skeleton

    worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.world_matrices = worlds
    return skeletons, skeletons_by_source


def resolve_armature(
    skeleton: ResolvedExportSkeleton,
    nodes: list[ExportNode],
    get_node: Callable[[Any, Any], ExportNode],
) -> None:
    root_node = skeleton.armature
    root = root_node.source
    worlds = evaluate_node_worlds(nodes)
    root_matrix = worlds[root_node]

    for node in skeleton.bones:
        set_world_matrix(node, root_matrix @ node.matrix_local, worlds)

    for node in list(root_node.children):
        if getattr(node.source, "parent_type", None) == "BONE":
            pose_bone = root.pose.bones[node.source.parent_bone]
            parent = get_node(pose_bone, root_node.scope)
            matrix_world = np.array(node.source.matrix_world)
            node.reparent(parent)
            set_world_matrix(node, matrix_world, worlds)

    skeleton.world_matrices = worlds


def apply_axis_corrections(skeletons: list[ResolvedExportSkeleton], nodes: list[ExportNode]) -> dict[ExportNode, Array]:
    for skeleton in skeletons:
        worlds = evaluate_node_worlds(nodes)
        root_node = skeleton.armature
        root_matrix = worlds[root_node]
        root_inverse = la.inv(root_matrix)
        bind_relative_to_root: dict[ExportNode, Array] = {}

        for node in skeleton.bones:
            prev_matrix_local = node.matrix_local.copy()
            axis = export_axis_correction(node.source.name)
            bind_relative_to_root[node] = root_inverse @ worlds[node] @ axis

            matrix_world = root_matrix @ np.asarray(node.source.matrix) @ axis
            set_world_matrix(node, matrix_world, worlds)

            inverse_transform = la.solve(node.matrix_local, prev_matrix_local)
            for child in node.children:
                child.matrix_local = inverse_transform @ child.matrix_local
                worlds[child] = worlds[node] @ child.matrix_local

        skeleton.bind_relative_to_root = bind_relative_to_root
        skeleton.world_matrices = worlds

    worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.world_matrices = worlds
    return worlds
