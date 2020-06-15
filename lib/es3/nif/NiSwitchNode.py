from __future__ import annotations

from es3.utils.flags import bool_property
from .NiNode import NiNode


class NiSwitchNode(NiNode):
    active_index: uint32 = 0

    # flags access
    update_only_active = bool_property(mask=0x0020)

    def load(self, stream):
        super().load(stream)
        self.active_index = stream.read_uint()

    def save(self, stream):
        super().save(stream)
        stream.write_uint(self.active_index)


if __name__ == "__main__":
    from es3.utils.typing import *
