from __future__ import annotations

from es3.utils.math import compose, decompose_uniform, ID33, ZERO3
from .NiObject import NiObject
from .NiSkinDataBoneData import NiSkinDataBoneData


class NiSkinData(NiObject):
    rotation: NiMatrix3 = ID33
    translation: NiPoint3 = ZERO3
    scale: float32 = 1.0
    skin_partition: NiSkinPartition | None = None
    bone_data: list[NiSkinDataBoneData] = []

    _refs = (*NiObject._refs, "skin_partition")

    def load(self, stream):
        self.rotation = stream.read_floats(3, 3)
        self.translation = stream.read_floats(3)
        self.scale = stream.read_float()
        num_bones = stream.read_uint()
        self.skin_partition = stream.read_link()
        if num_bones:
            self.bone_data = [stream.read_type(NiSkinDataBoneData) for _ in range(num_bones)]

    def save(self, stream):
        stream.write_floats(self.rotation)
        stream.write_floats(self.translation)
        stream.write_float(self.scale)
        stream.write_uint(len(self.bone_data))
        stream.write_link(self.skin_partition)
        for item in self.bone_data:
            item.save(stream)

    def apply_scale(self, scale):
        self.translation *= scale
        for item in self.bone_data:
            item.apply_scale(scale)

    def update_center_radius(self, vertices):
        for item in self.bone_data:
            indices = item.vertex_weights["f0"]
            item.update_center_radius(vertices[indices])

    @property
    def matrix(self) -> ndarray:
        return compose(self.translation, self.rotation, self.scale)

    @matrix.setter
    def matrix(self, value: ndarray):
        self.translation, self.rotation, self.scale = decompose_uniform(value)


if __name__ == "__main__":
    from es3.nif import NiSkinPartition
    from es3.utils.typing import *
