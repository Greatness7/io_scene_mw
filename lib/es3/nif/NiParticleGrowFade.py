from __future__ import annotations

from .NiParticleModifier import NiParticleModifier


class NiParticleGrowFade(NiParticleModifier):
    grow_time: float32 = 0.0
    fade_time: float32 = 0.0

    def load(self, stream):
        super().load(stream)
        self.grow_time = stream.read_float()
        self.fade_time = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.grow_time)
        stream.write_float(self.fade_time)


if __name__ == "__main__":
    from es3.utils.typing import *
