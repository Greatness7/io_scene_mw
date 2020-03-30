from __future__ import annotations

from es3.utils.math import ZERO3
from .NiProperty import NiProperty


class NiFogProperty(NiProperty):
    fog_depth: float32 = 0.0
    fog_color: NiPoint3 = ZERO3

    def load(self, stream):
        super().load(stream)
        self.fog_depth = stream.read_float()
        self.fog_color = stream.read_floats(3)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.fog_depth)
        stream.write_floats(self.fog_color)


if __name__ == "__main__":
    from es3.utils.typing import *
