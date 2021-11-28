from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import bool_property, enum_property
from .NiTimeController import NiTimeController


class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2


class NiLookAtController(NiTimeController):
    look_at: NiAVObject | None = None

    _ptrs = (*NiTimeController._ptrs, "look_at")

    # provide access to related enums
    Axis = Axis

    # convenience properties
    flip = bool_property(mask=0x0010)
    axis = enum_property(Axis, mask=0x0060, pos=5)

    def load(self, stream):
        super().load(stream)
        self.look_at = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.look_at)


if __name__ == "__main__":
    from es3.nif import NiAVObject
    from es3.utils.typing import *
