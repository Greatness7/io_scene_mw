from __future__ import annotations

from collections import deque
from enum import IntEnum

from es3.utils.flags import bool_property, enum_property
from es3.utils.math import compose, decompose_uniform, dotproduct, ID33, ZERO3
from .NiBoundingVolume import NiBoundingVolume
from .NiObjectNET import NiObjectNET


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

    # TODO: remove
    children = []  # type: list[NiAVObject | None]

    # provide access to related enums
    PropagateMode = PropagateMode

    # flags access
    app_culled = bool_property(mask=0x0001)
    propagate_mode = enum_property(PropagateMode, mask=0x0006, pos=1)
    visual = bool_property(mask=0x0008)

    _refs = (*NiObjectNET._refs, "properties")

    def load(self, stream):
        super().load(stream)
        self.flags = stream.read_ushort()
        self.translation = stream.read_floats(3)
        self.rotation = stream.read_floats(3, 3)
        self.scale = stream.read_float()
        self.velocity = stream.read_floats(3)
        self.properties = stream.read_links()
        has_bounding_volume = stream.read_bool()
        if has_bounding_volume:
            self.bounding_volume = NiBoundingVolume.load(stream)

    def save(self, stream):
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

    def sort(self, key=lambda prop: prop.type):
        super().sort()
        self.properties.sort(key=key)

    def apply_scale(self, scale: float):
        self.translation *= scale
        if self.bounding_volume:
            self.bounding_volume.apply_scale(scale)

    def get_property(self, property_type: type[T]) -> T:
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

    def descendants(self, breadth_first=False) -> Iterator[NiAVObject]:
        queue = deque(filter(None, self.children))
        extend, iterator = (queue.extendleft, iter) if breadth_first else \
                           (queue.extend, reversed)
        while queue:
            node = queue.pop()
            yield node
            extend(child for child in iterator(node.children) if child)

    def descendants_pairs(self, breadth_first=False) -> Iterator[tuple[NiAVObject, NiAVObject]]:
        """Similar to descendants, but yielding pairs of (parent, node)."""

        queue = deque((self, child) for child in self.children if child)
        extend, iterator = (queue.extendleft, iter) if breadth_first else \
                           (queue.extend, reversed)
        while queue:
            parent, node = queue.pop()
            yield parent, node
            extend((node, child) for child in iterator(node.children) if child)

    def find_path(self, ancestor, breadth_first=True) -> Iterator[NiAVObject]:
        parents = {}
        for parent, node in ancestor.descendants_pairs(breadth_first):
            parents[node] = parent
            if node is self:
                break
        else:
            raise ValueError(f"find_path: no path from {self} to {ancestor} exists")

        while node is not ancestor:
            yield node
            node = parents[node]


if __name__ == "__main__":
    from es3.nif import NiProperty
    from es3.utils.typing import *
