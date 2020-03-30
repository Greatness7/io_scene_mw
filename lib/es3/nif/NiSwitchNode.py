from __future__ import annotations

from .NiNode import NiNode


class NiSwitchNode(NiNode):
    active_index: uint32 = 0

    def load(self, stream):
        super().load(stream)
        self.active_index = stream.read_uint()

    def save(self, stream):
        super().save(stream)
        stream.write_uint(self.active_index)


if __name__ == "__main__":
    from es3.utils.typing import *
