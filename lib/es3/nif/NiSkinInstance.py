from __future__ import annotations

from ..utils.typing import *

from .NiAVObject import NiAVObject
from .NiBinaryStream import NiBinaryStream
from .NiObject import NiObject
from .NiSkinData import NiSkinData


class NiSkinInstance(NiObject):
    data: NiSkinData | None = None
    root: NiAVObject | None = None
    bones: list[NiAVObject | None] = []

    _refs = (*NiObject._refs, "data")
    _ptrs = (*NiObject._ptrs, "root", "bones")

    def load(self, stream: NiBinaryStream) -> None:
        self.data = stream.read_link()
        self.root = stream.read_link()
        self.bones = stream.read_links()

    def save(self, stream: NiBinaryStream) -> None:
        stream.write_link(self.data)
        stream.write_link(self.root)
        stream.write_links(self.bones)
