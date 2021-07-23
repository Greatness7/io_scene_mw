from __future__ import annotations

from es3.utils.math import zeros
from .NiFloatData import KeyType, NiFloatData


class NiMorphDataMorphTarget(NiFloatData):  # TODO Not NiObject
    vertices: ndarray = zeros(0, 3)

    def load(self, stream, num_vertices=0):
        num_keys = stream.read_uint()
        self.key_type = KeyType(stream.read_int())
        if num_keys:
            self.keys = stream.read_floats(num_keys, self.key_size)
        if num_vertices:
            self.vertices = stream.read_floats(num_vertices, 3)

    def save(self, stream):
        num_keys = len(self.keys)
        stream.write_uint(num_keys)
        stream.write_int(self.key_type)
        if num_keys:
            stream.write_floats(self.keys)
        if len(self.vertices):
            stream.write_floats(self.vertices)


if __name__ == "__main__":
    from es3.utils.typing import *
