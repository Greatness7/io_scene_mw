from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import bool_property, enum_property
from .NiProperty import NiProperty


class AlphaBlendFunction(IntEnum):
    ONE = 0
    ZERO = 1
    SRC_COLOR = 2
    INV_SRC_COLOR = 3
    DST_COLOR = 4
    INV_DST_COLOR = 5
    SRC_ALPHA = 6
    INV_SRC_ALPHA = 7
    DST_ALPHA = 8
    INV_DST_ALPHA = 9
    SRC_ALPHA_SAT = 10


class AlphaTestFunction(IntEnum):
    ALWAYS = 0
    LESS = 1
    EQUAL = 2
    LESS_EQUAL = 3
    GREATER = 4
    NOT_EQUAL = 5
    GREATER_EQUAL = 6
    NEVER = 7


class NiAlphaProperty(NiProperty):
    test_ref: uint8 = 0  # [0, 255]

    # provide access to related enums
    AlphaBlendFunction = AlphaBlendFunction
    AlphaTestFunction = AlphaTestFunction

    # convenience properties
    alpha_blending = bool_property(mask=0x0001)
    src_blend_mode = enum_property(AlphaBlendFunction, mask=0x001E, pos=1)
    dst_blend_mode = enum_property(AlphaBlendFunction, mask=0x01E0, pos=5)

    alpha_testing = bool_property(mask=0x0200)
    test_mode = enum_property(AlphaTestFunction, mask=0x1C00, pos=10)

    no_sort = bool_property(mask=0x2000)

    def load(self, stream):
        super().load(stream)
        self.test_ref = stream.read_ubyte()

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(self.test_ref)


if __name__ == "__main__":
    from es3.utils.typing import *
