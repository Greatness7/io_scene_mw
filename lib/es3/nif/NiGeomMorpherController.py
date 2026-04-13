from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiMorpherController import NiMorpherController


class NiGeomMorpherController(NiMorpherController):
    always_update: uint8 = 0

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.always_update = stream.read_ubyte()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_ubyte(self.always_update)
