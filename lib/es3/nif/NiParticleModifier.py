from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiObject import NiObject

if TYPE_CHECKING:
    from .NiParticleSystemController import NiParticleSystemController


class NiParticleModifier(NiObject):
    next: NiParticleModifier | None = None
    controller: NiParticleSystemController | None = None

    _refs = (*NiObject._refs, "next")
    _ptrs = (*NiObject._refs, "controller")

    def load(self, stream: NiBinaryStream) -> None:
        self.next = stream.read_link()
        self.controller = stream.read_link()

    def save(self, stream: NiBinaryStream) -> None:
        stream.write_link(self.next)
        stream.write_link(self.controller)
