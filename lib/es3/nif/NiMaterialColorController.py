from __future__ import annotations

from enum import IntEnum

from es3.utils.flags import enum_property
from .NiTimeController import NiTimeController


class ColorField(IntEnum):
    AMBIENT = 0
    DIFFUSE = 1
    SPECULAR = 2
    EMISSIVE = 3


class NiMaterialColorController(NiTimeController):
    data: NiPosData | None = None

    # provide access to related enums
    ColorField = ColorField

    # convenience properties
    color_field = enum_property(ColorField, mask=0x0070, pos=4)

    _refs = (*NiTimeController._refs, "data")

    def load(self, stream):
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.data)


if __name__ == "__main__":
    from es3.nif import NiPosData
    from es3.utils.typing import *
