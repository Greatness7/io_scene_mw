from __future__ import annotations

from es3.utils.flags import bool_property

from .NiTimeController import NiTimeController


class NiLightColorController(NiTimeController):
    data: NiPosData | None = None

    _refs = (*NiTimeController._refs, "data")

    ambient = bool_property(mask=0x0010)

    def load(self, stream):
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.data)


if __name__ == "__main__":
    from es3.nif import NiPosData
    from es3.utils.typing import *
