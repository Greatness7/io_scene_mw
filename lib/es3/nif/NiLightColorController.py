from __future__ import annotations

from ..utils.flags import bool_property
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiPosData import NiPosData
from .NiTimeController import NiTimeController


class NiLightColorController(NiTimeController):
    data: NiPosData | None = None

    _refs = (*NiTimeController._refs, "data")

    ambient = bool_property(mask=0x0010)

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_link(self.data)
