from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

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
