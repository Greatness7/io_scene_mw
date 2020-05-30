from __future__ import annotations

from enum import IntEnum

from .NiProperty import NiProperty


class SourceVertexMode(IntEnum):
    SOURCE_IGNORE = 0
    SOURCE_EMISSIVE = 1
    SOURCE_AMB_DIFF = 2
    SOURCE_NUM_MODES = 3


class LightingMode(IntEnum):
    LIGHTING_E = 0
    LIGHTING_E_A_D = 1
    LIGHTING_NUM_MODE = 2


class NiVertexColorProperty(NiProperty):
    source_vertex_mode: uint32 = SourceVertexMode.SOURCE_IGNORE
    lighting_mode: uint32 = LightingMode.LIGHTING_E_A_D

    # provide access to related enums
    SourceVertexMode = SourceVertexMode
    LightingMode = LightingMode

    def load(self, stream):
        super().load(stream)
        self.source_vertex_mode = SourceVertexMode(stream.read_uint())
        self.lighting_mode = LightingMode(stream.read_uint())

    def save(self, stream):
        super().save(stream)
        stream.write_uint(self.source_vertex_mode)
        stream.write_uint(self.lighting_mode)


if __name__ == "__main__":
    from es3.utils.typing import *
