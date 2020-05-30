from __future__ import annotations

from es3.utils.math import zeros
from .NiGeometryData import NiGeometryData


class NiLinesData(NiGeometryData):
    vertex_connectivity_flags: ndarray = zeros(0, dtype="<B")

    def load(self, stream):
        super().load(stream)
        num_vertices = len(self.vertices)
        if num_vertices:
            self.vertex_connectivity_flags = stream.read_ubytes(num_vertices)

    def save(self, stream):
        super().save(stream)
        num_vertices = len(self.vertices)
        if num_vertices:
            stream.write_ubytes(self.vertex_connectivity_flags)


if __name__ == "__main__":
    from es3.utils.typing import *
