from __future__ import annotations

from .NiObject import NiObject


class NiExtraData(NiObject):
    next: NiExtraData | None = None
    bytes_remaining: uint32 = 0

    _refs = (*NiObject._refs, "next")

    def load(self, stream):
        self.next = stream.read_link()
        self.bytes_remaining = stream.read_uint()

    def save(self, stream):
        stream.write_link(self.next)
        stream.write_uint(self.bytes_remaining)


if __name__ == "__main__":
    from es3.utils.typing import *
