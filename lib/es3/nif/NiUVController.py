from __future__ import annotations

from .NiTimeController import NiTimeController


class NiUVController(NiTimeController):
    texture_set: uint16 = 0
    data: NiUVData | None = None

    _refs = (*NiTimeController._refs, "data")

    def load(self, stream):
        super().load(stream)
        self.texture_set = stream.read_ushort()
        self.data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_ushort(self.texture_set)
        stream.write_link(self.data)


if __name__ == "__main__":
    from es3.nif import NiUVData
    from es3.utils.typing import *
