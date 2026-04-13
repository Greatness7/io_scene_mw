from __future__ import annotations

from ..utils.math import ZERO3
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiBoundingVolume import NiBoundingVolume


class NiSphereBV(NiBoundingVolume):
    center: NiPoint3 = ZERO3
    radius: float32 = 0

    bound_type = NiBoundingVolume.BoundType.SPHERE_BV

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.center = stream.read_floats(3)
        self.radius = stream.read_float()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_floats(self.center)
        stream.write_float(self.radius)

    def apply_scale(self, scale: float) -> None:
        self.center *= scale
        self.radius *= scale
