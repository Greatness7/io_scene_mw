from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiObjectNET import NiObjectNET


class NiProperty(NiObjectNET):
    flags: uint16 = 0

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.flags = stream.read_ushort()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_ushort(self.flags)
