from __future__ import annotations

from es3.utils.math import zeros
from .NiObject import NiObject


class NiPartition(NiObject):  # TODO Not NiObject
    bones: ndarray = zeros(0, dtype="<H")
    vertex_map: ndarray = zeros(0, dtype="<H")
    weights: ndarray = zeros(0)
    triangles: ndarray = zeros(0, dtype="<H")
    strip_lengths: ndarray = zeros(0, dtype="<H")
    strips: ndarray = zeros(0, dtype="<H")
    bone_palette: ndarray = zeros(0, dtype="<B")

    def load(self, stream):
        num_vertices = stream.read_ushort()
        num_triangles = stream.read_ushort()
        num_bones = stream.read_ushort()
        num_strip_lengths = stream.read_ushort()
        num_bones_per_vertex = stream.read_ushort()

        self.bones = stream.read_ushorts(num_bones)
        self.vertex_map = stream.read_ushorts(num_vertices)
        self.weights = stream.read_floats(num_bones_per_vertex, num_vertices)

        if num_triangles:
            self.triangles = stream.read_ushorts(num_triangles, 3)
        elif num_strip_lengths:
            self.strip_lengths = stream.read_ushorts(num_strip_lengths)
            self.strips = stream.read_ushorts(self.strip_lengths.sum())

        has_palette = stream.read_ubyte()
        if has_palette:
            self.bone_palette = stream.read_ubytes(num_bones_per_vertex, num_vertices)

    def save(self, stream):
        stream.write_ushort(len(self.vertex_map))
        stream.write_ushort(len(self.triangles))
        stream.write_ushort(len(self.bones))
        stream.write_ushort(len(self.strip_lengths))
        stream.write_ushort(len(self.weights))

        stream.write_ushorts(self.bones)
        stream.write_ushorts(self.vertex_map)
        stream.write_floats(self.weights)

        if len(self.triangles):
            stream.write_ushorts(self.triangles)
        elif len(self.strips):
            stream.write_ushorts(self.strip_lengths)
            stream.write_ushorts(self.strips)

        stream.write_ubyte(len(self.bone_palette))
        if len(self.bone_palette):
            stream.write_ubytes(self.bone_palette)


class NiSkinPartition(NiObject):
    partitions: list[NiPartition] = []

    def load(self, stream):
        num_partitions = stream.read_uint()
        self.partitions = [
            stream.read_type(NiPartition) for _ in range(num_partitions)
        ]

    def save(self, stream):
        num_partitions = len(self.partitions)
        stream.write_uint(num_partitions)
        for item in self.partitions:
            item.save(stream)


if __name__ == "__main__":
    from es3.utils.typing import *
