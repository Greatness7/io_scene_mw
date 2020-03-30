from __future__ import annotations

from es3.utils.math import ZERO3
from .NiObject import NiObject


class NiPerParticleData(NiObject):  # TODO not NiObject
    velocity: NiPoint3 = ZERO3
    rotation_axis: NiPoint3 = ZERO3
    age: float32 = 0.0
    lifespan: float32 = 0.0
    last_update: float32 = 0.0
    generation: uint16 = 0
    index: uint16 = 0

    def load(self, stream):
        self.velocity = stream.read_floats(3)
        self.rotation_axis = stream.read_floats(3)
        self.age = stream.read_float()
        self.lifespan = stream.read_float()
        self.last_update = stream.read_float()
        self.generation = stream.read_ushort()
        self.index = stream.read_ushort()

    def save(self, stream):
        stream.write_floats(self.velocity)
        stream.write_floats(self.rotation_axis)
        stream.write_float(self.age)
        stream.write_float(self.lifespan)
        stream.write_float(self.last_update)
        stream.write_ushort(self.generation)
        stream.write_ushort(self.index)


if __name__ == "__main__":
    from es3.utils.typing import *
