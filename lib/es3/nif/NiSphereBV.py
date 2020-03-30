from __future__ import annotations

from es3.utils.math import ZERO3
from .NiBoundingVolume import NiBoundingVolume


class NiSphereBV(NiBoundingVolume):
    center: NiPoint3 = ZERO3
    radius: float32 = 0

    bound_type = NiBoundingVolume.BoundType.SPHERE_BV

    def load(self, stream):
        super().load(stream)
        self.center = stream.read_floats(3)
        self.radius = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.center)
        stream.write_float(self.radius)

    def apply_scale(self, scale):
        self.center *= scale
        self.radius *= scale


if __name__ == "__main__":
    from es3.utils.typing import *
