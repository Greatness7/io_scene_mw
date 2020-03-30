from __future__ import annotations

from es3.utils.math import ZERO4, ZERO6, zeros
from .NiAVObject import NiAVObject


class NiCamera(NiAVObject):
    view_frustum: NiFrustum = ZERO6
    view_port: NiRect = ZERO4
    lod_adjust: float32 = 0.0
    scene: Optional[NiNode] = None
    screen_polygons: ndarray = zeros(0, dtype="<I")  # TODO NiScreenPolygon

    _refs = (*NiAVObject._refs, "scene")

    def load(self, stream):
        super().load(stream)
        self.view_frustum = stream.read_floats(6)
        self.view_port = stream.read_floats(4)
        self.lod_adjust = stream.read_float()
        self.scene = stream.read_link()
        num_screen_polygons = stream.read_uint()
        if num_screen_polygons:
            self.screen_polygons = stream.read_uints(num_screen_polygons)

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.view_frustum)
        stream.write_floats(self.view_port)
        stream.write_float(self.lod_adjust)
        stream.write_link(self.scene)
        stream.write_uint(len(self.screen_polygons))
        stream.write_uints(self.screen_polygons)


if __name__ == "__main__":
    from es3.nif import NiNode
    from es3.utils.typing import *
