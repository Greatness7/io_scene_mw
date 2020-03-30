from __future__ import annotations

from enum import IntEnum

from es3.utils.math import ZERO3
from .NiParticleModifier import NiParticleModifier


class ForceType(IntEnum):
    FORCE_PLANAR = 0
    FORCE_SPHERICAL = 1


class NiGravity(NiParticleModifier):
    decay: float32 = 0.0
    strength: float32 = 0.0
    force_type: int32 = ForceType.FORCE_PLANAR
    position: NiPoint3 = ZERO3
    direction: NiPoint3 = ZERO3

    # provide access to related enums
    ForceType = ForceType

    def load(self, stream):
        super().load(stream)
        self.decay = stream.read_float()
        self.strength = stream.read_float()
        self.force_type = ForceType(stream.read_int())
        self.position = stream.read_floats(3)
        self.direction = stream.read_floats(3)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.decay)
        stream.write_float(self.strength)
        stream.write_int(self.force_type)
        stream.write_floats(self.position)
        stream.write_floats(self.direction)


if __name__ == "__main__":
    from es3.utils.typing import *
