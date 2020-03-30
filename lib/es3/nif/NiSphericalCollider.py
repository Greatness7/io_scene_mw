from __future__ import annotations

from es3.utils.math import ZERO3
from .NiParticleCollider import NiParticleCollider


class NiSphericalCollider(NiParticleCollider):
    radius: float32 = 0.0
    position: NiPoint3 = ZERO3

    def load(self, stream):
        super().load(stream)
        self.radius = stream.read_float()
        self.position = stream.read_floats(3)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.radius)
        stream.write_floats(self.position)


if __name__ == "__main__":
    from es3.utils.typing import *
