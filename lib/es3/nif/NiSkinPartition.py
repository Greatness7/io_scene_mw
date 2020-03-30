from __future__ import annotations

from es3.utils.math import zeros
from .NiObject import NiObject


class NiSkinPartitionData(NiObject):  # TODO Not NiObject
    vertices: ndarray = zeros(0, dtype="<H")
    triangles: ndarray = zeros(0, dtype="<H")
    bones: ndarray = zeros(0, dtype="<H")
    strips: ndarray = zeros(0, dtype="<H")
    bones_per_vertex: ndarray = zeros(0, dtype="<H")
    vertex_map: ndarray = zeros(0, dtype="<H")
    weights: ndarray = zeros(0)
    strip_lengths: ndarray = zeros(0, dtype="<H")
    bone_palette: ndarray = zeros(0, dtype="<H")

    def load(self, stream):
        num_vertices = stream.read_ushort()
        num_triangles = stream.read_ushort()
        num_bones = stream.read_ushort()
        num_strips = stream.read_ushort()
        num_bones_per_vertex = stream.read_ushort()

        self.bones = stream.read_ushorts(num_bones)
        self.vertex_map = stream.read_ushorts(num_vertices)
        self.weights = stream.read_floats(num_bones_per_vertex, num_vertices)
        self.strip_lengths = stream.read_ushorts(num_strips)
        self.triangles = stream.read_ushorts(self.strip_lengths.sum() or (num_triangles * 3))

        has_palette = stream.read_ubyte()
        if has_palette:
            self.bone_palette = stream.read_ubytes(num_bones_per_vertex, num_vertices)

    def save(self, stream):
        stream.write_ushort(len(self.vertices))
        stream.write_ushort(len(self.triangles))
        stream.write_ushort(len(self.bones))
        stream.write_ushort(len(self.strips))
        stream.write_ushort(len(self.bones_per_vertex))

        stream.write_ushorts(self.bones)
        stream.write_ushorts(self.vertex_map)
        stream.write_ushorts(self.weights)
        stream.write_ushorts(self.strip_lengths)
        stream.write_ushorts(self.triangles)

        stream.write_ubyte(len(self.bone_palette))
        if len(self.bone_palette):
            stream.write_ubytes(self.bone_palette)


class NiSkinPartition(NiObject):
    skin_partitions: List[NiSkinPartitionData] = []

    def load(self, stream):
        num_partitions = stream.read_uint()
        self.skin_partitions = [
            stream.read_type(NiSkinPartitionData) for _ in range(num_partitions)
        ]

    def save(self, stream):
        num_partitions = len(self.skin_partitions)
        stream.write_uint(num_partitions)
        for item in self.skin_partitions:
            item.save(stream)


if __name__ == "__main__":
    from es3.utils.typing import *
