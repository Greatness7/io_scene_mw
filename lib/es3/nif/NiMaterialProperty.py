from __future__ import annotations

from es3.utils.math import ZERO3
from .NiProperty import NiProperty


class NiMaterialProperty(NiProperty):
    ambient_color: NiPoint3 = ZERO3
    diffuse_color: NiPoint3 = ZERO3
    specular_color: NiPoint3 = ZERO3
    emissive_color: NiPoint3 = ZERO3
    shine: float32 = 0.0
    alpha: float32 = 1.0

    def load(self, stream):
        super().load(stream)
        self.ambient_color = stream.read_floats(3)
        self.diffuse_color = stream.read_floats(3)
        self.specular_color = stream.read_floats(3)
        self.emissive_color = stream.read_floats(3)
        self.shine = stream.read_float()
        self.alpha = stream.read_float()

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.ambient_color)
        stream.write_floats(self.diffuse_color)
        stream.write_floats(self.specular_color)
        stream.write_floats(self.emissive_color)
        stream.write_float(self.shine)
        stream.write_float(self.alpha)


if __name__ == "__main__":
    from es3.utils.typing import *
