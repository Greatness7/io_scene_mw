from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from ..utils.math import ID44, allclose, la
from ..utils.typing import *

from .NiAVObject import NiAVObject
from .NiDynamicEffect import NiDynamicEffect


def _sort_children_key(child: NiAVObject | None) -> tuple[str, bool, int]:
    name = getattr(child, "name", "").lower()
    if name.startswith("tri "):
        prefix, suffix = name.rsplit(" ", 1)
        if suffix.isnumeric():
            return prefix, True, int(suffix)
    return name, False, 0


if TYPE_CHECKING:
    from ..utils.math import FloatArray
    from .NiBinaryStream import NiBinaryStream
    from .NiGeometry import NiGeometry


class NiNode(NiAVObject):
    children: list[NiAVObject | None] = []
    effects: list[NiDynamicEffect | None] = []

    _refs = (*NiAVObject._refs, "children", "effects")

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.children = stream.read_links()
        self.effects = stream.read_links()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_links(self.children)
        stream.write_links(self.effects)

    def sort(self, key: Callable[[Any], Any] = _sort_children_key) -> None:
        """Sort the NiNode's children list.

        If no key is provided children will be sorted by their name, with
        special handling for the "Tri (...) N" naming convention, where N
        is interpreted as an integer rather than a string.
        """
        super().sort()
        self.children.sort(key=key)

    def skinned_meshes(self) -> Iterator[NiGeometry]:
        for mesh in self.descendants():
            if getattr(mesh, "skin", None):
                yield cast("NiGeometry", mesh)

    def apply_skins(self, keep_skins: bool = False) -> None:
        for mesh in self.skinned_meshes():
            mesh.apply_skin(keep_skins)

    def calc_bone_bind_poses(self) -> dict[NiAVObject, FloatArray]:
        """TODO
        handle bones without defined bind poses
        e.g. any Bip01 stuffs unused in rigging
        """
        temps: dict[NiGeometry, dict[NiAVObject | None, FloatArray]] = {}
        binds: dict[NiAVObject | None, FloatArray] = {}

        # collect bind positions
        for mesh in self.skinned_meshes():
            matrix = mesh.matrix_relative_to(self)
            temps[mesh] = {bone: matrix @ la.inv(data.matrix) for bone, data in mesh.bone_influences}

        # bind position priority
        def sorter(items: tuple[NiGeometry, dict[NiAVObject | None, FloatArray]]) -> tuple[int, int]:
            node, data = items
            # give shadow meshes preference as they tend to encompass the entire skeleton
            return int(node.is_shadow), len(data)  # else bones with more users gets prio

        # process bind positions
        for mesh, data in sorted(temps.items(), key=sorter, reverse=True):
            # compare with previously seen bones
            for bone in data.keys() & binds.keys():
                diff = la.solve(data[bone], binds[bone])

                # do the bind positions coincide
                if not allclose(diff, ID44, rtol=0, atol=0.001):
                    print(f"MISMATCHED BINDPOSES: {bone} vs {mesh}")
                    # print(diff)
                    break
            else:
                binds.update(data)

        # use hierarchical order
        return {bone: binds[bone] for bone in self.descendants() if bone in binds}

    def apply_bone_bind_poses(self, lock_children: bool = False) -> None:
        bind_poses = self.calc_bone_bind_poses()

        for bone, bind in bind_poses.items():
            pose = bone.matrix_relative_to(self)

            # calculate difference
            diff = la.solve(pose, bind)

            if not allclose(diff, ID44, rtol=0, atol=1e-6):
                # update bind pose
                bone.matrix = bone.matrix @ diff

                # correct children
                if lock_children:
                    diff_inv = la.inv(diff)
                    for child in bone.children:
                        if child is not None:
                            child.matrix = child.matrix @ diff_inv
