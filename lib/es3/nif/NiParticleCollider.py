from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiParticleModifier import NiParticleModifier


class NiParticleCollider(NiParticleModifier):
    bounce: float32 = 0.0

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.bounce = stream.read_float()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_float(self.bounce)
