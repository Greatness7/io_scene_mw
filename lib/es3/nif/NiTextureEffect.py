from __future__ import annotations

from es3.utils.math import ID33, ZERO3, ZERO4
from .NiDynamicEffect import NiDynamicEffect


class NiTextureEffect(NiDynamicEffect):
    model_projection_matrix: NiMatrix3 = ID33
    model_projection_translation: NiPoint3 = ZERO3
    texture_filtering: uint32 = 2  # TODO: enum
    texture_clamping: uint32 = 3  # TODO: enum
    texture_type: uint32 = 2  # TODO: enum
    coordinate_generation_type: uint32 = 2  # TODO: enum
    source_texture: Optional[NiSourceTexture] = None
    clipping_plane_enable: uint8 = 0
    clipping_plane: NiPlane = ZERO4
    ps2_l: int16 = 0
    ps2_k: int16 = -75
    unknown_byte1: uint8 = 0  # TODO unknown
    unknown_byte2: uint8 = 0  # TODO unknown

    _refs = (*NiDynamicEffect._refs, "source_texture")

    def load(self, stream):
        super().load(stream)
        self.model_projection_matrix = stream.read_floats(3, 3)
        self.model_projection_translation = stream.read_floats(3)
        self.texture_filtering = stream.read_uint()
        self.texture_clamping = stream.read_uint()
        self.texture_type = stream.read_uint()
        self.coordinate_generation_type = stream.read_uint()
        self.source_texture = stream.read_link()
        self.clipping_plane_enable = stream.read_ubyte()
        self.clipping_plane = stream.read_floats(4)
        self.ps2_l = stream.read_short()
        self.ps2_k = stream.read_short()
        self.unknown_byte1 = stream.read_ubyte()
        self.unknown_byte2 = stream.read_ubyte()

    def save(self, stream):
        super().save(stream)
        stream.write_floats(self.model_projection_matrix)
        stream.write_floats(self.model_projection_translation)
        stream.write_uint(self.texture_filtering)
        stream.write_uint(self.texture_clamping)
        stream.write_uint(self.texture_type)
        stream.write_uint(self.coordinate_generation_type)
        stream.write_link(self.source_texture)
        stream.write_ubyte(self.clipping_plane_enable)
        stream.write_floats(self.clipping_plane)
        stream.write_short(self.ps2_l)
        stream.write_short(self.ps2_k)
        stream.write_ubyte(self.unknown_byte1)
        stream.write_ubyte(self.unknown_byte2)


if __name__ == "__main__":
    from es3.nif import NiSourceTexture
    from es3.utils.typing import *
