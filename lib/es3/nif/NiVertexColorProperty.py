from __future__ import annotations

from .NiProperty import NiProperty


class NiVertexColorProperty(NiProperty):
    vertex_mode: uint32 = 0  # TODO: enum
    lighting_mode: uint32 = 1  # TODO: enum

    def load(self, stream):
        super().load(stream)
        self.vertex_mode = stream.read_uint()
        self.lighting_mode = stream.read_uint()

    def save(self, stream):
        super().save(stream)
        stream.write_uint(self.vertex_mode)
        stream.write_uint(self.lighting_mode)


if __name__ == "__main__":
    from es3.utils.typing import *
