from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiParticlesData import NiParticlesData


class NiRotatingParticlesData(NiParticlesData):
    rotations: ndarray = zeros(0, 4)

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        num_vertices = len(self.vertices)
        has_rotations = stream.read_bool()
        if has_rotations:
            self.rotations = stream.read_floats(num_vertices, 4)

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_bool(len(self.rotations))
        if len(self.rotations):
            stream.write_floats(self.rotations)
