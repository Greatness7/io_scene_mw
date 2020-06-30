from __future__ import annotations

from es3.utils.math import ZERO3
from .NiParticleModifier import NiParticleModifier


class NiParticleRotation(NiParticleModifier):
    random_initial_axis: uint8 = 0
    initial_axis: NiPoint3 = ZERO3
    rotation_speed: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.random_initial_axis = stream.read_ubyte()
        self.initial_axis = stream.read_floats(3)
        self.rotation_speed = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(self.random_initial_axis)
        stream.write_floats(self.initial_axis)
        stream.write_float(self.rotation_speed)


if __name__ == "__main__":
    from es3.utils.typing import *
