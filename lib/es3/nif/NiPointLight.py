from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiLight import NiLight


class NiPointLight(NiLight):
    constant_attenuation: float32 = 0.0
    linear_attenuation: float32 = 0.0
    quadratic_attenuation: float32 = 0.0

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.constant_attenuation = stream.read_float()
        self.linear_attenuation = stream.read_float()
        self.quadratic_attenuation = stream.read_float()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_float(self.constant_attenuation)
        stream.write_float(self.linear_attenuation)
        stream.write_float(self.quadratic_attenuation)
