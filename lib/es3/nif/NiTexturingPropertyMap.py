from __future__ import annotations

from enum import IntEnum

from .NiObject import NiObject


class ClampMode(IntEnum):
    CLAMP_S_CLAMP_T = 0
    CLAMP_S_WRAP_T = 1
    WRAP_S_CLAMP_T = 2
    WRAP_S_WRAP_T = 3


class FilterMode(IntEnum):
    FILTER_NEAREST = 0
    FILTER_BILERP = 1
    FILTER_TRILERP = 2
    FILTER_NEAREST_MIPNEAREST = 3
    FILTER_NEAREST_MIPLERP = 4
    FILTER_BILERP_MIPNEAREST = 5


class NiTexturingPropertyMap(NiObject):  # TODO Not NiObject
    source: NiSourceTexture | None = None
    clamp_mode: int32 = ClampMode.WRAP_S_WRAP_T
    filter_mode: int32 = FilterMode.FILTER_TRILERP
    uv_set: uint32 = 0
    ps2_l: int16 = 0
    ps2_k: int16 = -75
    unknown_byte1: int8 = 0
    unknown_byte2: int8 = 0

    # provide access to related enums
    ClampMode = ClampMode
    FilterMode = FilterMode

    def load(self, stream):
        self.source = stream.read_link()
        self.clamp_mode = ClampMode(stream.read_int())
        self.filter_mode = FilterMode(stream.read_int())
        self.uv_set = stream.read_uint()
        self.ps2_l = stream.read_short()
        self.ps2_k = stream.read_short()
        self.unknown_byte1 = stream.read_byte()
        self.unknown_byte2 = stream.read_byte()

    def save(self, stream):
        stream.write_link(self.source)
        stream.write_int(self.clamp_mode)
        stream.write_int(self.filter_mode)
        stream.write_uint(self.uv_set)
        stream.write_short(self.ps2_l)
        stream.write_short(self.ps2_k)
        stream.write_byte(self.unknown_byte1)
        stream.write_byte(self.unknown_byte2)


if __name__ == "__main__":
    from es3.nif import NiSourceTexture
    from es3.utils.typing import *
