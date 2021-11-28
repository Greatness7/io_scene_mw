from __future__ import annotations

from .NiTimeController import NiTimeController


class NiVisController(NiTimeController):
    data: NiVisData | None = None

    _refs = (*NiTimeController._refs, "data")

    def load(self, stream):
        super().load(stream)
        self.data = stream.read_link()

    def save(self, stream):
        super().save(stream)
        stream.write_link(self.data)


if __name__ == "__main__":
    from es3.nif import NiVisData
    from es3.utils.typing import *
