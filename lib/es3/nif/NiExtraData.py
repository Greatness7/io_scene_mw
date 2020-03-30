from __future__ import annotations

from .NiObject import NiObject


class NiExtraData(NiObject):
    next: Optional[NiExtraData] = None

    _refs = (*NiObject._refs, "next")

    def load(self, stream):
        self.next = stream.read_link()
        stream.read_uint()  # bytes remaining  TODO only used if not subclassed

    def save(self, stream):
        stream.write_link(self.next)
        stream.write_uint(0)  # bytes remaining


if __name__ == "__main__":
    from es3.utils.typing import *
