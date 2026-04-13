from __future__ import annotations

from ..utils.typing import *

from .NiFloatData import NiFloatData
from .NiTimeController import NiTimeController


class NiFloatController(NiTimeController):
    data: NiFloatData | None = None

    _refs = (*NiTimeController._refs, "data")

    def load(self, stream):
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.data)
