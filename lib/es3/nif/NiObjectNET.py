from __future__ import annotations

from es3.utils.linked_list import LinkedListHelper
from .NiObject import NiObject


class NiObjectNET(NiObject):
    name: str = ""
    controller: NiTimeController | None = None
    extra_data: NiExtraData | None = None

    _refs = (*NiObject._refs, "extra_data", "controller")

    controllers = LinkedListHelper(name="controller")
    extra_datas = LinkedListHelper(name="extra_data")

    def __repr__(self):
        return f"{self.type}<'{self.name}'>"

    def load(self, stream):
        self.name = stream.read_str()
        self.extra_data = stream.read_link()
        self.controller = stream.read_link()

    def save(self, stream):
        stream.write_str(self.name)
        stream.write_link(self.extra_data)
        stream.write_link(self.controller)


if __name__ == "__main__":
    from es3.nif import NiExtraData, NiTimeController
    from es3.utils.typing import *
