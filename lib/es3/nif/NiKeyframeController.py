from __future__ import annotations

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiKeyframeData import NiKeyframeData
from .NiTimeController import NiTimeController


class NiKeyframeController(NiTimeController):
    data: NiKeyframeData | None = None

    _refs = (*NiTimeController._refs, "data")

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_link(self.data)
