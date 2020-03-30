from __future__ import annotations

from .NiTriShapeData import NiTriShapeData


class NiTriShapeDynamicData(NiTriShapeData):
    active_vertices: uint16 = 0
    active_triangles: uint16 = 0

    def load(self, stream):
        super().load(stream)
        self.active_vertices = stream.read_ushort()
        self.active_triangles = stream.read_ushort()

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(self.active_vertices)
        stream.write_ushort(self.active_triangles)


if __name__ == "__main__":
    from es3.utils.typing import *
