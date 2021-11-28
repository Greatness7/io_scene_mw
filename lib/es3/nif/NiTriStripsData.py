from __future__ import annotations

from es3.utils.math import np
from .NiTriBasedGeomData import NiTriBasedGeomData


class NiTriStripsData(NiTriBasedGeomData):
    strips: list[ndarray] = []

    def load(self, stream):
        super().load(stream)
        stream.read_ushort()  # num_triangles
        num_strips = stream.read_ushort()
        if num_strips:
            strip_lengths = stream.read_ushorts(num_strips)
            self.strips = [stream.read_ushorts(n) for n in strip_lengths]

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(0)
        stream.write_ushort(len(self.strips))
        strip_lengths = np.array([*map(len, self.strips)])
        stream.write_ushorts(strip_lengths)
        for strip in self.strips:
            stream.write_ushorts(strip)


if __name__ == "__main__":
    from es3.utils.typing import *
