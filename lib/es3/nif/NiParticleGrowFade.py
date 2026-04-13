from __future__ import annotations

from ..utils.typing import *

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

    def apply_time_scale(self, scale: float):
        super().apply_time_scale(scale)
        self.grow_time *= scale
        self.fade_time *= scale
