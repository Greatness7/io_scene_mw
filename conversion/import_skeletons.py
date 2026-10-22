from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44, compose, decompose, quaternion_from_matrix, quaternion_mul
from es3.utils.transforms import evaluate_world_matrices

from .bone_axes import export_axis_correction, import_axis_correction
from .records import Array, ImportNode, ImportRole, ResolvedImportSkeleton


def evaluate_node_worlds(nodes: Iterable[ImportNode]) -> dict[ImportNode, Array]:
    return evaluate_world_matrices((node, node.parent, node.matrix_local) for node in nodes)


def iter_parents(node: ImportNode) -> Iterator[ImportNode]:
    parent = node.parent
    while parent is not None:
        yield parent
        parent = parent.parent


def set_world_matrix(node: ImportNode, matrix: Array, worlds: dict[ImportNode, Array]) -> None:
    if node.parent is None:
        node.matrix_local = matrix
    else:
        node.matrix_local = la.solve(worlds[node.parent], matrix)


def snap_rotation(m: Array) -> Array:
    """Snap a rotation matrix to the nearest signed axis permutation."""
    m_abs = np.abs(m)
    m_out = np.zeros_like(m)

    for _ in m:
        x, y = np.unravel_index(m_abs.argmax(), m.shape)
        m_out[x, y] = -1 if m[x, y] < 0 else 1
        m_abs[x, :] = m_abs[:, y] = 0

    return m_out


def get_armature_node(
    source: Any,
    get_node: Callable[[Any], ImportNode],
    skeletons_by_source: dict[Any, ResolvedImportSkeleton],
) -> ImportNode | None:
    node = get_node(source)
    return next((candidate for candidate in (node, *iter_parents(node)) if candidate.source in skeletons_by_source), None)


def get_skinned_meshes(
    armature: ImportNode,
    nodes: Iterable[ImportNode],
    get_node: Callable[[Any], ImportNode],
    skeletons_by_source: dict[Any, ResolvedImportSkeleton],
) -> Iterator[ImportNode]:
    for node in nodes:
        skin = getattr(node.source, "skin", None)
        if skin and skin.root and get_armature_node(skin.root, get_node, skeletons_by_source) is armature:
            yield node


def resolve_armatures(
    nodes: list[ImportNode],
    bone_candidates: dict[Any, set[Any]],
    get_node: Callable[[Any], ImportNode],
) -> tuple[list[ResolvedImportSkeleton], dict[Any, ResolvedImportSkeleton]]:
    orphan_bones = bone_candidates.get(None, set())
    candidates = {root: set(bones) for root, bones in bone_candidates.items() if root is not None}
    posed_worlds = evaluate_node_worlds(nodes)

    merged: collections.defaultdict[Any, set[Any]] = collections.defaultdict(set)
    for root, bones in candidates.items():
        outer = [parent.source for parent in iter_parents(get_node(root)) if parent.source in candidates]
        merged[outer[-1] if outer else root].update(bones, {root})

    skeletons: list[ResolvedImportSkeleton] = []
    skeletons_by_source: dict[Any, ResolvedImportSkeleton] = {}
    for node in nodes:
        if node.source in merged and node.source not in skeletons_by_source:
            skeleton = resolve_armature(node.source, merged[node.source], orphan_bones, nodes, get_node, posed_worlds)
            if skeleton is not None:
                skeletons.append(skeleton)
                skeletons_by_source[node.source] = skeleton

    rest_worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.rest_world = rest_worlds
    return skeletons, skeletons_by_source


def resolve_armature(
    root: Any,
    bones: set[Any],
    orphan_bones: set[Any],
    nodes: list[ImportNode],
    get_node: Callable[[Any], ImportNode],
    posed_worlds: dict[ImportNode, Array],
) -> ResolvedImportSkeleton | None:
    root_node = get_node(root)
    ancestors = {node.source for node in (root_node, *iter_parents(root_node))}

    for bone in orphan_bones:
        if any(parent.source is root for parent in iter_parents(get_node(bone))):
            bones.add(bone)
    bones -= ancestors
    if not bones:
        return None

    for root_bone in filter(bones.__contains__, root.children):
        if not isinstance(root_bone, nif.NiAVObject):
            continue
        for child in root_bone.descendants():
            if isinstance(child, nif.NiNode) and child.controllers.find_type(nif.NiKeyframeController):
                bones.add(child)

    for node in list(map(get_node, bones)):
        for parent in iter_parents(node):
            source = parent.source
            if source is root or source in bones:
                break
            bones.add(source)

    ordered_bones = tuple(node for node in nodes if node.source in bones)
    root_bone = ordered_bones[0]
    posed = {node: posed_worlds[node] for node in ordered_bones}

    root.apply_bone_bind_poses()
    root.apply_skins(keep_skins=True)

    for node in ordered_bones:
        node.matrix_local = node.source.matrix
        node.role = ImportRole.BONE
    root_node.role = ImportRole.ARMATURE

    return ResolvedImportSkeleton(root_node, root_bone, ordered_bones, posed, evaluate_node_worlds(nodes))


def correct_rest_positions(
    skeletons: list[ResolvedImportSkeleton],
    nodes: list[ImportNode],
    get_node: Callable[[Any], ImportNode],
    skeletons_by_source: dict[Any, ResolvedImportSkeleton],
) -> None:
    for skeleton in skeletons:
        worlds = evaluate_node_worlds(nodes)
        root_bone = skeleton.root_bone
        t, rotation, scale = decompose(skeleton.posed_world[root_bone])
        corrected_matrix = compose(t, snap_rotation(rotation), scale)
        root_bone_world = worlds[root_bone]
        if np.allclose(root_bone_world, corrected_matrix, rtol=0, atol=1e-6):
            continue

        inverse = la.inv(root_bone_world)
        for node in get_skinned_meshes(skeleton.root, nodes, get_node, skeletons_by_source):
            if root_bone not in set(iter_parents(node)):
                set_world_matrix(node, corrected_matrix @ (inverse @ worlds[node]), worlds)
        set_world_matrix(root_bone, corrected_matrix, worlds)

    worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.rest_world = worlds


def apply_axis_corrections(skeletons: list[ResolvedImportSkeleton], nodes: list[ImportNode]) -> None:
    for skeleton in skeletons:
        bones = skeleton.bones
        bone_set = set(bones)

        for node in reversed(bones):
            axis = import_axis_correction(node.source.name)
            inverse_axis = export_axis_correction(node.source.name)
            skeleton.posed_world[node] = skeleton.posed_world[node] @ axis
            node.matrix_local = node.matrix_local @ axis
            for child in node.children:
                child.matrix_local = inverse_axis @ child.matrix_local

        worlds = evaluate_node_worlds(nodes)
        root_inverse = la.inv(worlds[skeleton.root])
        for node in bones:
            kf_controller = node.source.controllers.find_type(nif.NiKeyframeController)
            if not (kf_controller and kf_controller.data):
                continue

            parent = node.parent
            if parent in bone_set:
                parent_matrix = skeleton.posed_world[parent]
                parent_matrix_uncorrected = parent_matrix @ export_axis_correction(parent.source.name)
            else:
                parent_matrix = worlds[parent] if parent is not None else ID44
                parent_matrix_uncorrected = parent_matrix

            matrix_world = parent_matrix @ node.matrix_local
            matrix_relative_to_root = root_inverse @ matrix_world
            posed_offset = la.solve(matrix_relative_to_root, root_inverse) @ parent_matrix_uncorrected

            translations = kf_controller.data.translations
            if len(translations.values):
                rotation = posed_offset[:3, :3].T
                translation = posed_offset[:3, 3]
                translations.values[:] = translations.values @ rotation + translation
                if translations.key_type.name == "BEZ_KEY":
                    translations.in_tans[:] = translations.in_tans @ rotation
                    translations.out_tans[:] = translations.out_tans @ rotation

            rotations = kf_controller.data.rotations
            if len(rotations.values):
                axis_fix = quaternion_from_matrix(import_axis_correction(node.source.name))
                rotations.values[:] = quaternion_mul(rotations.values, axis_fix)
                to_posed = quaternion_from_matrix(posed_offset)
                rotations.values[:] = quaternion_mul(to_posed, rotations.values)

    worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.rest_world = worlds


def correct_bone_parenting(
    skeletons: list[ResolvedImportSkeleton],
    nodes: list[ImportNode],
    get_node: Callable[[Any], ImportNode],
    skeletons_by_source: dict[Any, ResolvedImportSkeleton],
) -> None:
    for skeleton in skeletons:
        worlds = evaluate_node_worlds(nodes)
        for node in get_skinned_meshes(skeleton.root, nodes, get_node, skeletons_by_source):
            if node.parent is not skeleton.root:
                matrix_world = worlds[node]
                node.reparent(skeleton.root)
                set_world_matrix(node, matrix_world, worlds)

    worlds = evaluate_node_worlds(nodes)
    for skeleton in skeletons:
        skeleton.rest_world = worlds
