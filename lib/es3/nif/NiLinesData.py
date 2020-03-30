from __future__ import annotations

from es3.utils.math import zeros
from .NiGeometryData import NiGeometryData


class NiLinesData(NiGeometryData):
    lines: ndarray = zeros(0, dtype="<I")

    def load(self, stream):
        super().load(stream)
        num_vertices = len(self.vertices)
        if num_vertices:
            self.lines = stream.read_uints(num_vertices)

    def save(self, stream):
        super().save(stream)
        num_vertices = len(self.vertices)
        if num_vertices:
            stream.write_uints(self.lines)


if __name__ == "__main__":
    from es3.utils.typing import *
