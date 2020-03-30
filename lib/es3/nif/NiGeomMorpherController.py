from __future__ import annotations

from .NiMorpherController import NiMorpherController


class NiGeomMorpherController(NiMorpherController):
    always_update: uint8 = 0

    def load(self, stream):
        super().load(stream)
        self.always_update = stream.read_ubyte()

    def save(self, stream):
        super().save(stream)
        stream.write_ubyte(self.always_update)


if __name__ == "__main__":
    from es3.utils.typing import *
