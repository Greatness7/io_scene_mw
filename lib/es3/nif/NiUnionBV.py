from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiBoundingVolume import NiBoundingVolume


class NiUnionBV(NiBoundingVolume):
    bounding_volumes: list[NiBoundingVolume] = []

    bound_type = NiBoundingVolume.BoundType.UNION_BV

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        num_bounding_volumes = stream.read_uint()
        bounding_volumes: list[NiBoundingVolume] = []
        for _ in range(num_bounding_volumes):
            item = NiBoundingVolume.read(stream)
            bounding_volumes.append(item)
        self.bounding_volumes = bounding_volumes

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_uint(len(self.bounding_volumes))
        for item in self.bounding_volumes:
            item.save(stream)

    def apply_scale(self, scale: float) -> None:
        for item in self.bounding_volumes:
            item.apply_scale(scale)
