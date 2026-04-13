from __future__ import annotations

from ..utils.linked_list import LinkedListHelper
from ..utils.typing import *

from .NiBinaryStream import NiBinaryStream
from .NiExtraData import NiExtraData
from .NiObject import NiObject
from .NiTimeController import NiTimeController


class NiObjectNET(NiObject):
    name: str = ""
    controller: NiTimeController | None = None
    extra_data: NiExtraData | None = None

    _refs = (*NiObject._refs, "extra_data", "controller")

    controllers = LinkedListHelper["NiTimeController"](name="controller")
    extra_datas = LinkedListHelper["NiExtraData"](name="extra_data")

    def __repr__(self):
        return f"{self.type}<'{self.name}'>"

    def load(self, stream: NiBinaryStream) -> None:
        self.name = stream.read_str()
        self.extra_data = stream.read_link()
        self.controller = stream.read_link()

    def save(self, stream: NiBinaryStream) -> None:
        stream.write_str(self.name)
        stream.write_link(self.extra_data)
        stream.write_link(self.controller)
