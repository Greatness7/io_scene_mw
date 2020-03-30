from __future__ import annotations

from es3.utils.math import ZERO3, zeros
from .NiSwitchNode import NiSwitchNode


class NiLODNode(NiSwitchNode):
    lod_center: NiPoint3 = ZERO3
    lod_levels: ndarray = zeros(0, 2)

    def load(self, stream):
        super().load(stream)
        self.lod_center = stream.read_floats(3)
        num_lod_levels = stream.read_uint()
        if num_lod_levels:
            self.lod_levels = stream.read_floats(num_lod_levels, 2)

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.lod_center)
        stream.write_uint(len(self.lod_levels))
        stream.write_floats(self.lod_levels)


if __name__ == "__main__":
    from es3.utils.typing import *
