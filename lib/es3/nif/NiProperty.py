from __future__ import annotations

from ..utils.typing import *

from .NiObjectNET import NiObjectNET


class NiProperty(NiObjectNET):
    flags: uint16 = 0

    def load(self, stream):
        super().load(stream)
        self.flags = stream.read_ushort()

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(self.flags)
