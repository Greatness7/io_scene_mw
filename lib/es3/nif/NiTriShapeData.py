from __future__ import annotations

from es3.utils.math import zeros
from .NiTriBasedGeomData import NiTriBasedGeomData


class NiTriShapeData(NiTriBasedGeomData):
    triangles: ndarray = zeros(0, 3, dtype="<H")
    shared_normals: list[ndarray] = []

    def load(self, stream):
        super().load(stream)
        num_triangles = stream.read_ushort()
        num_triangle_points = stream.read_uint()
        if num_triangles and num_triangle_points:
            self.triangles = stream.read_ushorts(num_triangles, 3)
        num_shared_normals = stream.read_ushort()
        if num_shared_normals:
            self.shared_normals = [
                stream.read_ushorts(stream.read_ushort()) for _ in range(num_shared_normals)
            ]

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(len(self.triangles))
        stream.write_uint(self.triangles.size)
        stream.write_ushorts(self.triangles)
        stream.write_ushort(len(self.shared_normals))
        for index_array in self.shared_normals:
            stream.write_ushort(len(index_array))
            stream.write_ushorts(index_array)


if __name__ == "__main__":
    from es3.utils.typing import *
