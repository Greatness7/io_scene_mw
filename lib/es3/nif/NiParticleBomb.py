from __future__ import annotations

from enum import IntEnum

from es3.utils.math import ZERO3
from .NiParticleModifier import NiParticleModifier


class DecayType(IntEnum):
    NONE = 0
    LINEAR = 1
    EXPONENTIAL = 2


class SymmetryType(IntEnum):
    SPHERICAL = 0
    CYLINDRICAL = 1
    PLANAR = 2


class NiParticleBomb(NiParticleModifier):
    decay: float32 = 0.0
    duration: float32 = 0.0
    delta_v: float32 = 0.0  # units/seconds^2
    start_time: float32 = 0.0
    decay_type: int32 = DecayType.NONE
    symmetry_type: int32 = SymmetryType.SPHERICAL
    position: NiPoint3 = ZERO3
    direction: NiPoint3 = ZERO3

    # provide access to related enums
    DecayType = DecayType
    SymmetryType = SymmetryType

    def load(self, stream):
        super().load(stream)
        self.decay = stream.read_float()
        self.duration = stream.read_float()
        self.delta_v = stream.read_float()
        self.start_time = stream.read_float()
        self.decay_type = stream.read_int()
        self.symmetry_type = stream.read_int()
        self.position = stream.read_floats(3)
        self.direction = stream.read_floats(3)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.decay)
        stream.write_float(self.duration)
        stream.write_float(self.delta_v)
        stream.write_float(self.start_time)
        stream.write_int(self.decay_type)
        stream.write_int(self.symmetry_type)
        stream.write_floats(self.position)
        stream.write_floats(self.direction)


if __name__ == "__main__":
    from es3.utils.typing import *
