from __future__ import annotations

from .NiTimeController import NiTimeController


class NiLookAtController(NiTimeController):
    look_at: Optional[NiAVObject] = None

    _ptrs = (*NiTimeController._ptrs, "look_at")

    def load(self, stream):
        super().load(stream)
        self.look_at = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.look_at)


if __name__ == "__main__":
    from es3.nif import NiAVObject
    from es3.utils.typing import *
