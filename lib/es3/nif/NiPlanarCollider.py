from __future__ import annotations

from es3.utils.math import ZERO3
from .NiParticleCollider import NiParticleCollider


class NiPlanarCollider(NiParticleCollider):
    height: float32 = 0.0
    width: float32 = 0.0
    position: NiPoint3 = ZERO3
    x_axis: NiPoint3 = ZERO3
    y_axis: NiPoint3 = ZERO3
    normal: NiPoint3 = ZERO3
    distance: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.height = stream.read_float()
        self.width = stream.read_float()
        self.position = stream.read_floats(3)
        self.x_axis = stream.read_floats(3)
        self.y_axis = stream.read_floats(3)
        self.normal = stream.read_floats(3)
        self.distance = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.height)
        stream.write_float(self.width)
        stream.write_floats(self.position)
        stream.write_floats(self.x_axis)
        stream.write_floats(self.y_axis)
        stream.write_floats(self.normal)
        stream.write_float(self.distance)


if __name__ == "__main__":
    from es3.utils.typing import *
