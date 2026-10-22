from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import TYPE_CHECKING, cast

from ..utils.flags import bool_property, enum_property
from ..utils.math import compose, decompose_uniform, dotproduct, ID33, ID44, ZERO3
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiBoundingVolume import NiBoundingVolume
from .NiObjectNET import NiObjectNET
from .NiProperty import NiProperty

if TYPE_CHECKING:
    from .NiNode import NiNode  # cycle: NiNode imports NiAVObject


class PropagateMode(IntEnum):
    NONE = 0
    USE_TRIANGLES = 1
    USE_OBBS = 2
    CONTINUE = 3


class NiAVObject(NiObjectNET):
    flags: uint16 = 0
    translation: NiPoint3 = ZERO3
    rotation: NiMatrix3 = ID33
    scale: float32 = 1.0
    velocity: NiPoint3 = ZERO3
    properties: list[NiProperty | None] = []
    bounding_volume: NiBoundingVolume | None = None
    _parent: NiNode | None = None

    # provide access to related enums
    PropagateMode = PropagateMode

    # flags access
    app_culled = bool_property(mask=0x0001)
    propagate_mode = enum_property(PropagateMode, mask=0x0006, pos=1)
    visual = bool_property(mask=0x0008)

    _refs = (*NiObjectNET._refs, "properties")

    @property
    def parent(self) -> NiNode | None:
        """The node that owns this object's spatial slot. Only the mutators on `NiNode` write it."""
        return self._parent

    @property
    def children(self) -> Sequence[NiAVObject | None]:
        """An immutable empty sequence because leaf objects own no child slots."""
        return ()

    @classmethod
    def attributes(cls) -> set[str]:
        # `_parent` is runtime ownership, not stream data
        return super().attributes() - {"_parent"}

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.flags = stream.read_ushort()
        self.translation = stream.read_floats(3)
        self.rotation = stream.read_floats(3, 3)
        self.scale = stream.read_float()
        self.velocity = stream.read_floats(3)
        self.properties = stream.read_links()
        has_bounding_volume = stream.read_bool()
        if has_bounding_volume:
            self.bounding_volume = NiBoundingVolume.read(stream)

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_ushort(self.flags)
        stream.write_floats(self.translation)
        stream.write_floats(self.rotation)
        stream.write_float(self.scale)
        stream.write_floats(self.velocity)
        stream.write_links(self.properties)
        stream.write_bool(self.bounding_volume)
        if self.bounding_volume:
            self.bounding_volume.save(stream)

    def sort(self, key: Callable[[Any], Any] = lambda prop: prop.type) -> None:
        super().sort()
        self.properties.sort(key=key)

    def apply_scale(self, scale: float):
        self.translation *= scale
        if self.bounding_volume:
            self.bounding_volume.apply_scale(scale)

    def get_property(self, property_type: type[T]) -> T | None:
        for prop in self.properties:
            if isinstance(prop, property_type):
                return prop

    @property
    def matrix(self) -> ndarray:
        return compose(self.translation, self.rotation, self.scale)

    @matrix.setter
    def matrix(self, value: ndarray):
        self.translation, self.rotation, self.scale = decompose_uniform(value)

    def matrix_relative_to(self, ancestor: NiAVObject) -> ndarray:
        if self is ancestor:
            return cast("ndarray", ID44.copy())  # pyright: ignore[reportUnknownMemberType]
        path = reversed(list(self.find_path(ancestor)))
        return dotproduct([obj.matrix for obj in path])

    @property
    def is_biped(self) -> bool:
        return self.name.lower().startswith("bip01")

    @property
    def is_shadow(self) -> bool:
        return self.name.lower().startswith(("shadow", "tri shadow"))

    @property
    def is_bounding_box(self) -> bool:
        return bool(self.bounding_volume) and self.name.lower().startswith("bounding box")

    def children_of_type(self, cls: type[T]) -> Iterator[T]:
        yield from (child for child in self.children if isinstance(child, cls))

    def descendants(self, breadth_first: bool = False) -> Iterator[NiAVObject]:
        queue: deque[NiAVObject] = deque(child for child in self.children if child is not None)
        while queue:
            node = queue.pop()
            yield node
            if breadth_first:
                queue.extendleft(child for child in node.children if child is not None)
            else:
                queue.extend(child for child in reversed(node.children) if child is not None)

    def descendants_pairs(self, breadth_first: bool = False) -> Iterator[tuple[NiAVObject, NiAVObject]]:
        """Similar to descendants, but yielding pairs of (parent, node)."""

        queue: deque[tuple[NiAVObject, NiAVObject]] = deque((self, child) for child in self.children if child is not None)
        while queue:
            parent, node = queue.pop()
            yield parent, node
            if breadth_first:
                queue.extendleft((node, child) for child in node.children if child is not None)
            else:
                queue.extend((node, child) for child in reversed(node.children) if child is not None)

    def find_path(self, ancestor: NiAVObject) -> Iterator[NiAVObject]:
        """Yield self, then each parent, up to but excluding `ancestor`."""
        node: NiAVObject | None = self
        while node is not ancestor:
            if node is None:
                raise ValueError(f"find_path: no path from {self} to {ancestor} exists")
            yield node
            node = node._parent
