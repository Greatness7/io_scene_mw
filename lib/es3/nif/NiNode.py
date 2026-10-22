from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from ..utils.children import ChildrenView
from ..utils.math import ID44, allclose, la
from ..utils.typing import *

from .NiAVObject import NiAVObject
from .NiDynamicEffect import NiDynamicEffect
from .NiObject import NiObject


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
    effects: list[NiDynamicEffect | None] = []
    _children: list[NiAVObject | None] = []
    _pending_child_indices: list[int] = []

    _refs = (*NiAVObject._refs, "children", "effects")
    _resolve_exclude = ("children",)

    @property
    def children(self) -> Sequence[NiAVObject | None]:
        return ChildrenView(self)

    @children.setter
    def children(self, value: Iterable[NiAVObject | None]) -> None:
        self.replace_children(value)

    @classmethod
    def attributes(cls) -> set[str]:
        runtime = {"_children", "_pending_child_indices"}
        return (super().attributes() - runtime) | {"children"}

    def asdict(self) -> dict[str, object]:
        fields = super().asdict()
        fields["children"] = list(self.children)
        return fields

    def _check_attachable(self, child: object) -> None:
        if not isinstance(child, NiAVObject):
            raise TypeError(f"child must be a NiAVObject, not {type(child).__name__}")
        node: NiAVObject | None = self
        while node is not None:
            if node is child:
                raise ValueError(f"attaching {child!r} to {self!r} would create a cycle")
            node = node._parent

    def attach_child(self, child: NiAVObject, *, first_available: bool = False) -> int:
        """Attach `child`, preserving its local transform, and return its new slot index."""
        self._check_attachable(child)
        if child._parent is not None:
            child._parent.detach_child(child)

        if first_available:
            index = next((i for i, item in enumerate(self._children) if item is None), len(self._children))
        else:
            index = len(self._children)

        if index == len(self._children):
            self._children.append(child)
        else:
            self._children[index] = child
        child._parent = self
        return index

    def set_child(self, index: int, child: NiAVObject | None) -> NiAVObject | None:
        """Set a slot while maintaining ownership; assigning a null tail retains that slot."""
        if type(index) is not int or index < 0:
            raise TypeError("child index must be a non-negative int")
        if index < len(self._children) and self._children[index] is child:
            return child
        if child is not None:
            self._check_attachable(child)

        if index >= len(self._children):
            self._children.extend([None] * (index + 1 - len(self._children)))
        previous = self._children[index]

        if child is not None and child._parent is not None:
            child._parent.detach_child(child)
        if previous is not None:
            previous._parent = None

        self._children[index] = child
        if child is not None:
            child._parent = self
        return previous

    def detach_child(self, child: NiAVObject) -> int:
        """Detach by identity; an interior slot becomes null and a tail removes exactly one slot."""
        for index, item in enumerate(self._children):
            if item is child:
                child._parent = None
                if index == len(self._children) - 1:
                    self._children.pop()
                else:
                    self._children[index] = None
                return index
        raise ValueError(f"{child!r} is not a child of {self!r}")

    def replace_children(self, children: Iterable[NiAVObject | None]) -> None:
        """Replace all slots in order; duplicate children use last-attachment ownership."""
        new = list(children)
        for child in new:
            if child is not None:
                self._check_attachable(child)

        for child in self._children:
            if child is not None:
                child._parent = None
        self._children.clear()
        for index, child in enumerate(new):
            self.set_child(index, child)

    def reorder_children(self, children: Sequence[NiAVObject | None]) -> None:
        """Reorder existing slots without changing ownership or local transforms."""
        new = list(children)
        if len(new) != len(self._children):
            raise ValueError(f"child slot count differs: expected {len(self._children)}, got {len(new)}")

        old_none_count = sum(child is None for child in self._children)
        new_none_count = sum(child is None for child in new)
        if new_none_count != old_none_count:
            raise ValueError(f"null slot count differs: expected {old_none_count}, got {new_none_count}")

        old_ids = sorted(id(child) for child in self._children if child is not None)
        new_ids = sorted(id(child) for child in new if child is not None)
        if new_ids != old_ids:
            raise ValueError("child identities differ; reorder_children cannot add or remove children")

        self._children[:] = new

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self._pending_child_indices = stream.read_links()
        self.effects = stream.read_links()

    def resolve_links(self, objects: dict[int, NiObject]) -> None:
        super().resolve_links(objects)
        pending = self._pending_child_indices
        for index, link in enumerate(pending):
            target = objects.get(link) if link != -1 else None
            if target is not None and not isinstance(target, NiAVObject):
                raise TypeError(f"{self!r}: child link {link} resolved to {target!r}, not a NiAVObject")
            self.set_child(index, target)
        pending.clear()

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
        self.reorder_children(sorted(self.children, key=key))

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
