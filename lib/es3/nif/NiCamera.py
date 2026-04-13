from __future__ import annotations

from ..utils.math import ZERO4, ZERO6
from ..utils.typing import *

from .NiAVObject import NiAVObject
from .NiNode import NiNode
from .NiScreenPolygon import NiScreenPolygon


class NiCamera(NiAVObject):
    view_frustum: NiFrustum = ZERO6
    view_port: NiRect = ZERO4
    lod_adjust: float32 = 0.0
    scene: NiNode | None = None
    screen_polygons: list[NiScreenPolygon | None] = []

    _refs = (*NiAVObject._refs, "scene")

    def load(self, stream):
        super().load(stream)
        self.view_frustum = stream.read_floats(6)
        self.view_port = stream.read_floats(4)
        self.lod_adjust = stream.read_float()
        self.scene = stream.read_link()
        self.screen_polygons = stream.read_links()

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.view_frustum)
        stream.write_floats(self.view_port)
        stream.write_float(self.lod_adjust)
        stream.write_link(self.scene)
        stream.write_links(self.screen_polygons)
