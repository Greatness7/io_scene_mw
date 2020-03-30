from __future__ import annotations

from es3.utils.math import ID22
from .NiTexturingPropertyMap import NiTexturingPropertyMap


class NiTexturingPropertyBumpMap(NiTexturingPropertyMap):  # TODO Not NiObject
    luma_scale: float32 = 0.0
    luma_offset: float32 = 0.0
    displacement: NiMatrix2 = ID22

    def load(self, stream):
        super().load(stream)
        self.luma_scale = stream.read_float()
        self.luma_offset = stream.read_float()
        self.displacement = stream.read_floats(2, 2)

    def save(self, stream):
        super().save(stream)
        stream.write_float(self.luma_scale)
        stream.write_float(self.luma_offset)
        stream.write_floats(self.displacement)


if __name__ == "__main__":
    from es3.utils.typing import *
