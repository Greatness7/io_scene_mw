from __future__ import annotations

from .NiBoundingVolume import NiBoundingVolume


class NiUnionBV(NiBoundingVolume):
    bounding_volumes: list[NiBoundingVolume | None] = []

    bound_type = NiBoundingVolume.BoundType.UNION_BV

    def load(self, stream):
        super().load(stream)
        num_bounding_volumes = stream.read_uint()
        self.bounding_volumes = [
            NiBoundingVolume.load(stream) for _ in range(num_bounding_volumes)
        ]

    def save(self, stream):
        super().save(stream)
        stream.write_uint(len(self.bounding_volumes))
        for item in self.bounding_volumes:
            item.save(stream)

    def apply_scale(self, scale):
        for item in self.bounding_volumes:
            item.apply_scale(scale)


if __name__ == "__main__":
    from es3.utils.typing import *
