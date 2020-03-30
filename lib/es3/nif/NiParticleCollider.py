from __future__ import annotations

from .NiParticleModifier import NiParticleModifier


class NiParticleCollider(NiParticleModifier):
    bounce: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.bounce = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.bounce)


if __name__ == "__main__":
    from es3.utils.typing import *
