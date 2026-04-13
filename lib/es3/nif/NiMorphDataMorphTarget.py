from __future__ import annotations

from ..utils.math import zeros
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiFloatData import KeyType, NiFloatData


class NiMorphDataMorphTarget(NiFloatData):  # TODO Not NiObject
    vertices: ndarray = zeros(0, 3)

    def load(self, stream: NiBinaryStream, num_vertices: int = 0) -> None:
        num_keys = stream.read_uint()
        self.key_type = KeyType(stream.read_int())
        if num_keys:
            self.keys = stream.read_floats(num_keys, self.key_size)
        if num_vertices:
            self.vertices = stream.read_floats(num_vertices, 3)

    def save(self, stream: NiBinaryStream) -> None:
        num_keys = len(self.keys)
        stream.write_uint(num_keys)
        stream.write_int(self.key_type)
        if num_keys:
            stream.write_floats(self.keys)
        if len(self.vertices):
            stream.write_floats(self.vertices)
