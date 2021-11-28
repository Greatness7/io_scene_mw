from __future__ import annotations

from .NiMorphDataMorphTarget import NiMorphDataMorphTarget
from .NiObject import NiObject


class NiMorphData(NiObject):
    relative_targets: uint8 = 1
    targets: list[NiMorphDataMorphTarget] = []

    def load(self, stream):
        num_targets = stream.read_uint()
        num_vertices = stream.read_uint()
        self.relative_targets = stream.read_ubyte()
        self.targets = [NiMorphDataMorphTarget() for _ in range(num_targets)]
        for item in self.targets:
            item.load(stream, num_vertices)

    def save(self, stream):
        stream.write_uint(len(self.targets))
        stream.write_uint(len(self.basis) if self.targets else 0)
        stream.write_ubyte(self.relative_targets)
        for item in self.targets:
            item.save(stream)

    def apply_scale(self, scale):
        for item in self.targets:
            item.vertices *= scale

    @property
    def basis(self):
        return self.targets[0].vertices


if __name__ == "__main__":
    from es3.utils.typing import *
