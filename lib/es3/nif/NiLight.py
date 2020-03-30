from __future__ import annotations

from es3.utils.math import ZERO3
from .NiDynamicEffect import NiDynamicEffect


class NiLight(NiDynamicEffect):
    dimmer: float32 = 0.0
    ambient_color: NiPoint3 = ZERO3
    diffuse_color: NiPoint3 = ZERO3
    specular_color: NiPoint3 = ZERO3

    def load(self, stream):
        super().load(stream)
        self.dimmer = stream.read_float()
        self.ambient_color = stream.read_floats(3)
        self.diffuse_color = stream.read_floats(3)
        self.specular_color = stream.read_floats(3)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.dimmer)
        stream.write_floats(self.ambient_color)
        stream.write_floats(self.diffuse_color)
        stream.write_floats(self.specular_color)


if __name__ == "__main__":
    from es3.utils.typing import *
