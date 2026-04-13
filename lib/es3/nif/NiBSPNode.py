from __future__ import annotations

from ..utils.math import ZERO4
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiNode import NiNode


class NiBSPNode(NiNode):
    model_plane: NiPlane = ZERO4

    def load(self, stream: NiBinaryStream) -> None:
        super().load(stream)
        self.model_plane = stream.read_floats(4)

    def save(self, stream: NiBinaryStream) -> None:
        super().save(stream)
        stream.write_floats(self.model_plane)
